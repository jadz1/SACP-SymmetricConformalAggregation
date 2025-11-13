import numpy as np
import pandas as pd
from math import ceil
import time as ts


from joblib import Parallel, delayed
from numba import njit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from modelscifar import resnet56, VGG, ShuffleNetV2, EfficientNetB0, DLA

from utilis_def import predict_proba_model, score_function, generate_simplex_grid

def conformal_prediction_classification(models, X_calib, y_calib, X_test, y_test, alpha=0.05, method='higher'):
    """
        Compute conformal prediction using:
        - Split conformal prediction for individual base learners
        - SACP aggregation (ours)
        - Conformal aggregation via majority vote (CR, CM), reference: https://arxiv.org/abs/2401.09379
    """
    q_levels = {}
    individual_results = []
    n_calib = X_calib.shape[0]
    q_idx = np.ceil(((n_calib + 1) * (1 - alpha))) / n_calib

    is_image = X_calib.ndim > 2  # check if input is images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_probs_calib = []
    for name, m in models.items():
        probs_calib = predict_proba_model(m, X_calib, is_image, device=device)
        scores_calib = score_function(probs_calib, y_calib, False)
        q = np.quantile(scores_calib, q_idx, method=method)
        q_levels[name] = q

        probs_test = predict_proba_model(m, X_test, is_image, device=device)
        scores_test = score_function(probs_test, y_test, False)
        accuracy = np.mean(np.argmax(probs_test, axis=1) == y_test)
        should_log = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if should_log:
            print(f'model: {name}, oos accuracy: {accuracy}')
        mask = scores_test <= q
        coverage = np.mean(mask)
        pred_sets = probs_test >= (1.0 - q)
        set_sizes = pred_sets.sum(axis=1)

        # average set size
        avg_size = set_sizes.mean()
        individual_results.append({
            'Model': name,
            'Coverage': coverage,
            'Avg Length': avg_size
        })
        all_probs_calib.append(probs_calib)

    # 1‑bis) AGGREGATION CALIBRATION FOR CR & CM AT 1 - alpha/2
    # build a second dictionary of “half‑alpha” quantiles
    q2_levels = {}
    half = alpha / 2
    for name, m in models.items():
        probs_calib = predict_proba_model(m, X_calib, is_image, device=device)
        scores_calib = score_function(probs_calib, y_calib, False)
        q2 = np.quantile(
                scores_calib,
                np.ceil((len(X_calib) + 1) * (1 - half)) / len(X_calib),
                method=method
            )
        q2_levels[name] = q2


    # Ensemble conformal
    m_list = list(models.values())
    classes = np.arange(all_probs_calib[0].shape[1])
    K = len(classes)

    s_calib = np.stack([score_function(p, y_calib, False) for p in all_probs_calib], axis=1)  # (n_calib, n_models)
    probs_test_all = np.stack([predict_proba_model(m, X_test, is_image, device=device) for m in m_list], axis=1)  # (n_test, n_models, K)

    sums = s_calib.sum(axis=0)
    sums_sq = (s_calib ** 2).sum(axis=0)

    n_test = len(y_test)
    covered, lengths = np.zeros(n_test, dtype=bool), np.zeros(n_test, dtype=int)
    cr_covered, cr_lengths, cm_covered, cm_lengths = [], [], [], []

    # SACP, CR & CM methods
    for i in range(n_test):
        s_test_grid = 1.0 - probs_test_all[i].T  # shape (K, n_models)
        denom = sums[None, :] + s_test_grid
        e_calib = (s_calib[None, :, :] / denom[:, None, :]).mean(axis=2)
        e_test = (s_test_grid / denom).mean(axis=1)

        q_grid = np.quantile(e_calib, q_idx, axis=1, method=method)
        mask = (e_test <= q_grid)
        true_k = y_test[i]
        covered[i] = mask[true_k]
        lengths[i] = mask.sum()

        vote_mask = (s_test_grid <= np.array(list(q2_levels.values()))).astype(float)
        vote_frac = vote_mask.mean(axis=1)
        U = np.random.rand()
        cr = vote_frac >= (0.5 + U / 2)
        cm = vote_frac >= 0.5
        cr_covered.append(cr[true_k]); cr_lengths.append(cr.sum())
        cm_covered.append(cm[true_k]); cm_lengths.append(cm.sum())

    agg_coverage = covered.mean()
    agg_avg_length = lengths.mean()
    if (not dist.is_initialized()) or (dist.get_rank() == 0):
        print(f'Aggregate coverage: {agg_coverage:.4f}, avg length: {agg_avg_length:.4f}')
        print(f'CR coverage: {np.mean(cr_covered):.4f}, avg length: {np.mean(cr_lengths):.4f}')
        print(f'CM coverage: {np.mean(cm_covered):.4f}, avg length: {np.mean(cm_lengths):.4f}')
    return (
        individual_results,
        agg_coverage, agg_avg_length,
        np.mean(cr_covered), np.mean(cr_lengths),
        np.mean(cm_covered), np.mean(cm_lengths)
    )


def score_function(probs, y_true):
    return 1.0 - probs[np.arange(len(y_true)), y_true]

def run_CSA_class(models, X_train, y_train, X_calib, y_calib, X_test, y_test, alpha=0.05, M=50, pct_split=0.5, method='higher'):
    """
        Implementation of Conformal Prediction for Ensembles: Improving Efficiency via Score-Based Aggregation (CSA)
        Reference: https://arxiv.org/abs/2405.16246
    """
    is_image = X_calib.ndim > 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_probs_calib = np.stack([predict_proba_model(m, X_calib, is_image, device=device) for m in list(models.values())], axis=1)  # (n_test, n_models, K)

    S_calib = np.stack([score_function(all_probs_calib[:, m_idx, :], y_calib, False) for m_idx in range(all_probs_calib.shape[1])], axis=1)

    n1 = int(len(S_calib) * pct_split)
    S1, S2 = S_calib[:n1], S_calib[n1:]

    # Sample M random directions in positive orthant
    U = np.abs(np.random.randn(M, S1.shape[1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    # Binary search for beta*
    beta_lo, beta_hi = alpha / M, alpha
    for _ in range(20):
        beta = (beta_lo + beta_hi) / 2
        q_tilde = np.quantile(S1 @ U.T, 1 - beta, axis=0, method=method)
        coverage = ((S1 @ U.T <= q_tilde).all(axis=1)).mean()
        if coverage > 1 - alpha:
            beta_lo = beta
        else:
            beta_hi = beta

    beta_star = beta_lo
    q_tilde = np.quantile(S1 @ U.T, 1 - beta_star, axis=0, method=method)

    # Calibrate t_hat using S2
    t_star = np.max(S2 @ U.T / q_tilde, axis=1)
    t_hat = np.quantile(t_star, 1 - alpha, method=method)

    # Test set evaluation
    all_probs_test = np.stack([predict_proba_model(m, X_test, is_image, device=device) for m in list(models.values())], axis=1)  # (n_test, n_models, K)
    S_test = np.stack([score_function(all_probs_test[:, m_idx, :], y_test, False) for m_idx in range(all_probs_test.shape[1])], axis=1)
    g_test = np.max(S_test @ U.T / q_tilde, axis=1)

    # Compute coverage (whether true label is in prediction set)
    coverage = (g_test <= t_hat).mean()

    # Approximate average set size (predicted sets)
    probs_all = np.transpose(all_probs_test, (1, 0, 2))
    avg_set_sizes = []
    pred_sets=[]

    for i in range(len(X_test)):
        probs = probs_all[:, i, :]  # shape: (num_models, n_classes)
        s_class = 1 - probs  # score = 1 - predicted probability
        g_class = np.max(U @ s_class / q_tilde[:, None], axis=0)
        pred_set = np.where(g_class <= t_hat)[0]
        avg_set_sizes.append(len(pred_set))
        pred_sets.append(pred_set)

    meanL = np.mean(avg_set_sizes)
    should_log = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if should_log:
        print(f'CSA cov: {coverage:.4f}, meanL: {meanL:.4f}')
    return coverage, meanL, avg_set_sizes, pred_sets



def run_weighted_agg_classification(models, X_calib, y_calib, X_test, y_test, alpha=0.05, eps=0.01, max_grid=200, seed=0, method='higher'):
    """
        Weighted Aggregation of Conformity Scores for Classification (Wagg)
        Validity-First Conformal Prediction (VFCP) for data splitting, which guarantees (1 - alpha) marginal coverage

    Reference: https://arxiv.org/pdf/2407.10230
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl_list = list(models.values())
    d = len(mdl_list)

    # Stack calibration probabilities: (n_calib, d, K)
    P_cal = np.stack([
        predict_proba_model(m, X_calib, X_calib.ndim > 2, device=device)
        for m in mdl_list
    ], axis=1)
    K = P_cal.shape[2]

    # Scores: 1 - probability of true label
    scores = 1 - np.take_along_axis(P_cal, y_calib[:, None, None], axis=2).squeeze(-1)

    # Generate weights
    grid = generate_simplex_grid(d, eps, max_grid, seed)
    best = {'size': np.inf, 'w': None, 'q': None}
    nC = len(y_calib)
    q_idx = np.ceil((nC + 1) * (1 - alpha)) / nC

    # Search weight grid
    for w in grid:
        # ensemble score on calib
        ens_scores = scores.dot(w)
        q = np.quantile(ens_scores, q_idx, method=method)

        # ensemble probabilities on calib: (nC, K)
        ens_P = (P_cal * w[None, :, None]).sum(axis=1)
        mask = (1 - ens_P) <= q  # shape (nC, K)

        avg_size = mask.sum(axis=1).mean()
        cov = np.mean(mask[np.arange(nC), y_calib])
        if cov >= 1 - alpha and avg_size < best['size']:
            best.update({'size': avg_size, 'w': w, 'q': q})

    # Apply to test
    P_test = np.stack([
        predict_proba_model(m, X_test, X_test.ndim > 2, device=device)
        for m in mdl_list
    ], axis=1)
    ens_Pt = (P_test * best['w'][None, :, None]).sum(axis=1)
    mask_t = (1 - ens_Pt) <= best['q']  # (n_test, K)

    pred_sets = [np.where(mask_t[i])[0] for i in range(len(y_test))]
    coverage = np.mean([y_test[i] in s for i, s in enumerate(pred_sets)])
    avg_set_size = np.mean([len(s) for s in pred_sets])
    if (not dist.is_initialized()) or (dist.get_rank() == 0):
        print(best['w'])
    return avg_set_size, coverage, best['w'], best['q']



@njit
def quantile_higher(x, q_idx):
    """
        Numba-compatible function to return the q_idx-th order statistic
    """
    x_sorted = np.sort(x)
    return x_sorted[max(q_idx, 0)]
@njit
def L_p(x, p, axis):
    if axis == -1:
        axis = x.ndim - 1
    return np.sum(np.abs(x) ** p, axis=axis) ** (1.0 / p)

@njit
def get_length_jit(p_value, probs_test_all, s_calib, sums, y_test, q_idx2):
    n_calib, n_test = s_calib.shape[0], probs_test_all.shape[0]
    covered_mask = np.zeros(n_test, dtype=np.bool_)
    lengths = np.zeros(n_test, dtype=np.int32)
    q_idx2 = int(q_idx2)
    K = probs_test_all.shape[2]
    for i in range(n_test):
        s_test_grid = 1.0 - probs_test_all[i].T  # shape (K, n_models)
        denom_grid = (sums[None, :] + s_test_grid) / n_calib
        e_test = L_p(s_test_grid / denom_grid, p=p_value, axis=1)
        e_calib = np.empty((K, n_calib), dtype=np.float32)
        for k in range(K):

            e_calib[k, :] = L_p(s_calib / denom_grid[k, :], p=p_value, axis=1)
        
        q_grid = np.empty(K, dtype=np.float32)
        for k in range(K):
            # q_grid[k] = np.quantile(e_calib[k, :], q_idx) # Method 'higher' is implicitly handled by q_idx logic
            q_grid[k] = quantile_higher(e_calib[k, :], q_idx2)

        mask = (e_test <= q_grid)
        true_k = y_test[i]
        covered_mask[i] = mask[true_k]
        lengths[i] = mask.sum()

    mean_length = lengths.mean()
    coverage = covered_mask.mean()
    return p_value, coverage, mean_length, covered_mask

def find_p_optimized(models, X_calib, y_calib, X_test, y_test, name, alpha=0.05, method='higher', p_min=1.0, p_max=20.0, num_points=70, n_jobs=-1, cond_cov=False, verbose=True):
    """
        SACP++ method, implemented with parallelization for the grid search over p
    """
    # Use float32 to save memory
    X_calib = X_calib.astype(np.float32)
    y_calib = y_calib.astype(np.int64)
    X_test  = X_test.astype(np.float32)
    y_test  = y_test.astype(np.int64)

    m_list = list(models.values())
    n_calib, n_test = X_calib.shape[0], X_test.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_probs_calib = np.stack([predict_proba_model(m, X_calib, X_calib.ndim > 2, device=device) for m in list(models.values())], axis=1)  # (n_test, n_models, K)
    all_probs_test = np.stack([predict_proba_model(m, X_test, X_test.ndim > 2, device=device) for m in list(models.values())], axis=1)  # (n_test, n_models, K)
    
    s_calib = np.stack([
        score_function(all_probs_calib[:, m_idx, :], y_calib)
        for m_idx in range(all_probs_calib.shape[1])
    ], axis=1)
    sums = s_calib.sum(axis=0)
    q_idx = ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_idx2 = int(ceil((n_calib + 1) * (1 - alpha)))

    p_values = np.linspace(p_min, p_max, num_points, dtype=np.float32)

    # Warm up JIT compiler
    _ = get_length_jit(p_values[0], all_probs_test, s_calib, sums, y_test, q_idx2)

    print(f"Launching grid search over {num_points} values of p using {n_jobs} threads...\n")
    t0 = ts.time()

    # Optionally track timing for each p-value
    def timed_get_length(p):
        if verbose:
            t1 = ts.time()
            result = get_length_jit(p, all_probs_test, s_calib, sums, y_test, q_idx2)
            return result
        else:
            return get_length_jit(p, all_probs_test, s_calib, sums, y_test, q_idx2)
    results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=10)(delayed(timed_get_length)(p) for p in p_values)

    n_test = X_test.shape[0]

    # Add min & max symmetric functions
    for m in ['max', 'min']:
        covered = np.zeros(n_test, dtype=bool)
        lengths = np.zeros(n_test)
        for i in range(n_test):
            s_test_grid = 1.0 - all_probs_test[i].T 
            denom = sums[None, :] + s_test_grid

            # compute e scores
            e_calib = (s_calib[None, :, :] / denom[:, None, :]).max(axis=2) if m == 'max' else (s_calib[None, :, :] / denom[:, None, :]).min(axis=2)
            e_test = (s_test_grid / denom).max(axis=1) if m == 'max' else (s_test_grid / denom).min(axis=1)
            
            q_grid = np.quantile(e_calib, q_idx, axis=1, method=method)

            mask = (e_test <= q_grid)
            true_k = y_test[i]
            covered[i] = mask[true_k]
            lengths[i] = mask.sum()
        results.append((m, covered.mean(), lengths.mean(), covered))

    total_time = ts.time() - t0
    n_success = len(results)

    results.sort(key=lambda x: x[2])
    p_opt, coverage_opt, length_opt, covered = results[0]


    print(f"\nGrid search complete in {total_time:.2f}s")
    print(f"Number of p-values computed: {n_success}")
    print(f"Optimal p*:       {p_opt}")
    print(f"Mean length*:     {length_opt:.4f}")
    print(f"Coverage:         {coverage_opt:.4f} (target ≥ {1-alpha})")

    return p_opt, coverage_opt, length_opt