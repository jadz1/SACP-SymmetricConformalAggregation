import numpy as np
from math import ceil
import time as ts


from joblib import Parallel, delayed
from numba import njit

from utilis_def import score_function, compute_scores, generate_simplex_grid


def run_baselines_and_sacp(models, X_calib, y_calib, X_test, y_test, alpha=0.05, K=255, method='higher'):
    """
        Compute conformal prediction using:
        - Split conformal prediction for individual base learners
        - SACP aggregation (ours)
        - Conformal aggregation via majority vote (CR, CM), reference: https://arxiv.org/abs/2401.09379
    """
    
    m_list = list(models.values())

    # 1) Individual-model intervals based on quantile of calibration residuals
    q_levels = {}
    individual_results = []
    for name, model in models.items():
        calib_pred = model.predict(X_calib)
        residuals = score_function(calib_pred, y_calib)
        q = np.quantile(residuals, np.ceil((len(X_calib) + 1) * (1 - alpha)) / len(X_calib), method=method)
        q_levels[name] = q
        # Evaluate on test
        y_pred = model.predict(X_test)
        lower = y_pred - q
        upper = y_pred + q
        coverages = (y_test >= lower) & (y_test <= upper)
        coverage = coverages.mean()
        avg_length = np.mean(upper - lower)
       
        individual_results.append({
            'Model': name,
            'Coverage': coverage,
            'Avg Length': avg_length,
        })

        # Conformal aggregation with CR & CM at level alpha/2 for fairness
        # (build a second dictionary of “half‑alpha” quantiles)
        q2_levels = {}
        half = alpha / 2
        for name, model in models.items():
            res = score_function(model.predict(X_calib), y_calib)
            q2 = np.quantile(
                res,
                np.ceil((len(X_calib) + 1) * (1 - half)) / len(X_calib),
                method=method
            )
            q2_levels[name] = q2

    # e aggreg
    s_calib = np.stack([np.abs(mj.predict(X_calib) - y_calib) for mj in m_list], axis=1)
    sums = s_calib.sum(axis=0)
    
    # Uniform grid
    y_min = y_calib.min()
    y_max = y_calib.max()
    y_grid = np.linspace(y_min, y_max, K)
    step = y_grid[1] - y_grid[0]

    # quantile index for e-calibration
    n_calib = X_calib.shape[0]
    q_idx = ceil((n_calib + 1) * (1 - alpha)) / (n_calib)

    n_test = X_test.shape[0]
    covered = np.zeros(n_test, dtype=bool)
    lengths = np.zeros(n_test)

    preds_test = np.stack([mj.predict(X_test) for mj in m_list], axis=1)
    cr_covered, cr_lengths, cm_covered, cm_lengths = [], [], [], []
    for i in range(n_test):
        s_test_grid = np.abs(y_grid[:, None] - preds_test[i][None, :])  # (K, m)
        denom = sums[None, :] + s_test_grid

        # compute e-values
        e_calib = (s_calib[None, :, :] / denom[:, None, :]).mean(axis=2)
        e_test = (s_test_grid / denom).mean(axis=1)
        
        q_grid = np.quantile(e_calib, q_idx, axis=1, method=method)

        mask = (e_test <= q_grid)
        idx_closest = np.abs(y_grid - y_test[i]).argmin()
        covered[i] = mask[idx_closest]
        lengths[i] = mask.sum() * step

        # CR (randomized threshold with U ~ Uniform[0,1])
        vote_mask = (s_test_grid <= np.array(list(q2_levels.values()))).astype(float)
        vote_frac = vote_mask.mean(axis=1)
        U = np.random.rand()
        cr_mask = vote_frac > (0.5 + U / 2)
        cm_mask = vote_frac > 0.5
        cm_covered.append(cm_mask[idx_closest])
        cm_lengths.append(cm_mask.sum() * step)
        cr_covered.append(cr_mask[idx_closest])
        cr_lengths.append(cr_mask.sum() * step)
    
    agg_coverage = covered.mean()
    agg_avg_length = lengths.mean()

    return individual_results, agg_coverage, agg_avg_length, np.mean(cr_covered), np.mean(cr_lengths), np.mean(cm_covered), np.mean(cm_lengths)



def run_CSA(models, X_calib, y_calib, X_test, y_test, alpha=0.05, K=255, M=50, pct_split=0.5, method='higher'):
    """
        Implementation of Conformal Prediction for Ensembles: Improving Efficiency via Score-Based Aggregation (CSA)

        Reference: https://arxiv.org/abs/2405.16246
    """
    # Split calibration set into two halves
    S_calib = compute_scores(models, X_calib, y_calib)
    n1 = int(len(S_calib) * pct_split)
    S1, S2 = S_calib[:n1], S_calib[n1:]

    # Sample M directions in positive orthant
    U = np.abs(np.random.randn(M, S1.shape[1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    # Binary search for beta* such that (1 - alpha) coverage is achieved
    beta_lo, beta_hi = alpha/M, alpha
    for _ in range(20):
        beta = (beta_lo + beta_hi) / 2
        q_tilde = np.quantile(S1.dot(U.T), 1 - beta, axis=0, method=method)
        coverage = ((S1.dot(U.T) <= q_tilde).all(axis=1)).mean()
        if coverage > 1 - alpha:
            beta_lo = beta
        else:
            beta_hi = beta

    beta_star = beta_lo
    q_tilde = np.quantile(S1.dot(U.T), 1 - beta_star, axis=0, method=method)

    # Calibrate t_hat on S2
    t_star = np.max(S2.dot(U.T) / q_tilde, axis=1)
    t_hat = np.quantile(t_star, 1 - alpha, method=method)

    # Evaluate on test set
    S_test = compute_scores(models, X_test, y_test)
    g_test = np.max(S_test.dot(U.T) / q_tilde, axis=1)
    # Compute empirical coverage
    mask_coverage = g_test <= t_hat
    coverage_test = mask_coverage.mean()

    # Uniform grid
    y_min = y_calib.min()
    y_max = y_calib.max()
    y_grid = np.linspace(y_min, y_max, K)
    step = y_grid[1] - y_grid[0]
    lengths = []
    preds_all = np.stack([m.predict(X_test) for m in models.values()], axis=0)

    for i in range(len(X_test)):
        preds = preds_all[:, i]
        s_grid = np.abs(preds[:, None] - y_grid[None, :])
        
        # Aggregate using random projections and normalization
        g_grid = np.max(U.dot(s_grid) / q_tilde[:, None], axis=0)
        lengths.append((g_grid <= t_hat).sum() * step)

    avg_length = np.mean(lengths)

    return coverage_test, avg_length


def run_WAgg(models, X_calib, y_calib, X_test, y_test, alpha=0.05, K=255, eps=0.01, method='higher'):
    """
        Weighted Aggregation of Conformity Scores for Classification (Wagg) - direct extension for regression.
        Validity-First Conformal Prediction (VFCP) for data splitting, which guarantees (1 - alpha) marginal coverage

        Reference: https://arxiv.org/pdf/2407.10230
    """

    n_test = X_test.shape[0]
    d = len(list(models.values()))
    weights = generate_simplex_grid(d, eps=0.01)

    S_cal = compute_scores(models, X_calib, y_calib)
    n_split = int(len(S_cal) * 0.5)
    S_cal1, S_cal2 = S_cal[:n_split], S_cal[n_split:]

    best_w = None
    min_length = np.inf
    best_q = None

    y_min = y_calib.min()
    y_max = y_calib.max()
    y_grid = np.linspace(y_min, y_max, K)
    step = y_grid[1] - y_grid[0]

    for w in weights:
        scores_cal = S_cal1 @ w.T
        q_w = np.quantile(scores_cal, np.ceil((n_split + 1) * (1 - alpha)) / n_split, method=method)
        preds_cal1 = np.column_stack([m.predict(X_calib[:n_split]) for m in models.values()]) @ w
        error_grid = np.abs(preds_cal1[:, None] - y_grid[None, :])
        mask = (error_grid <= q_w)
        lengths = mask.sum(axis=1) * step
        avg_length = lengths.mean()
        if avg_length < min_length:
            min_length = avg_length
            best_w = w
            best_q = q_w

    print("Optimal weights (grid):", best_w)
    new_S_cal = S_cal2 @ best_w
    q = np.quantile(new_S_cal, np.ceil((len(S_cal2) + 1) * (1 - alpha)) / len(S_cal2), method=method)
    print(f"Optimal quantile threshold: {q:.3f}")

    # Eval
    y_min, y_max = min(y_calib.min(), y_test.min()), max(y_calib.max(), y_test.max())
    y_grid = np.linspace(y_min, y_max, K)
    step = y_grid[1] - y_grid[0]

    lengths = []
    preds_test = np.column_stack([m.predict(X_test) for m in models.values()]) @ best_w
    error_grid = np.abs(preds_test[:, None] - y_grid[None, :])
    mask = (error_grid <= q)
    lengths = mask.sum(axis=1) * step 
    avg_length = lengths.mean()
    idx = np.abs(y_test[:, None] - y_grid[None, :]).argmin(axis=1)
    covered = mask[np.arange(n_test), idx]
    coverage = covered.mean()
    
    
    return avg_length, coverage, best_w, best_q


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
def get_length_jit(p_value, preds_test, y_grid, s_calib, sums, y_test, q_idx2, step, method):
    n_test, m = preds_test.shape
    n_calib = s_calib.shape[0]
    K = y_grid.shape[0]
    
    # Pre-allocate arrays for results
    lengths = np.zeros(n_test, dtype=np.float32)
    covered_mask = np.zeros(n_test, dtype=np.bool_)
    q_idx2 = int(q_idx2)
    # Loop over each test point to keep memory usage low
    for i in range(n_test):
        s_test_grid = np.abs(preds_test[i, :] - y_grid[:, None]) # (K, m)
        denom_grid = (sums + s_test_grid) / n_calib            # (K, m)
        e_test = L_p(s_test_grid / denom_grid, p=p_value, axis=1) # (K,)

        e_calib = np.empty((K, n_calib), dtype=np.float32)
        for k in range(K):
            # This is the calculation that is now done efficiently
            e_calib[k, :] = L_p(s_calib / denom_grid[k, :], p=p_value, axis=1)

        # Numba doesn't support the 'method' argument as a string literal directly in np.quantile
        # but works when passed as a variable.
        q_grid = np.empty(K, dtype=np.float32)
        for k in range(K):
            # q_grid[k] = np.quantile(e_calib[k, :], q_idx) # Method 'higher' is implicitly handled by q_idx logic
            q_grid[k] = quantile_higher(e_calib[k, :], q_idx2)
        mask = (e_test <= q_grid)
        lengths[i] = mask.sum() * step

        idx_true = np.abs(y_grid - y_test[i]).argmin()
        covered_mask[i] = mask[idx_true]

    mean_length = lengths.mean()
    coverage = covered_mask.mean()
    return p_value, coverage, mean_length, covered_mask

def run_SACP_pp(models, X_calib, y_calib, X_test, y_test, K=255, alpha=0.05, method='higher', p_min=1.0, p_max=20.0, num_points=70, n_jobs=-1, verbose=True):
    """
        SACP++ method, implemented with parallelization for the grid search over p
    """
    
    # Use float32 to save memory
    X_calib = X_calib.astype(np.float32)
    y_calib = y_calib.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    m_list = list(models.values())
    n_calib = X_calib.shape[0]

    s_calib = np.stack([m.predict(X_calib) for m in m_list], axis=1).astype(np.float32)
    s_calib = np.abs(s_calib - y_calib[:, None])
    sums = s_calib.sum(axis=0)

    y_min, y_max = y_calib.min(), y_calib.max()
    y_grid = np.linspace(y_min, y_max, K, dtype=np.float32)
    step = y_grid[1] - y_grid[0]
    q_idx = ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_idx2 = int(ceil((n_calib + 1) * (1 - alpha)))
    preds_test = np.stack([m.predict(X_test) for m in m_list], axis=1).astype(np.float32)
    p_values = np.linspace(p_min, p_max, num_points, dtype=np.float32)

    # Warm up JIT compiler
    _ = get_length_jit(p_values[0], preds_test, y_grid, s_calib, sums, y_test, q_idx2, step, method)

    t0 = ts.time()

    # Optionally track timing for each p-value
    def timed_get_length(p):
        if verbose:
            t1 = ts.time()
            result = get_length_jit(p, preds_test, y_grid, s_calib, sums, y_test, q_idx2, step, method)
            print(f"→ p = {p:.2f} done in {ts.time() - t1:.2f}s")
            return result
        else:
            return get_length_jit(p, preds_test, y_grid, s_calib, sums, y_test, q_idx2, step, method)

    results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=10)(delayed(timed_get_length)(p) for p in p_values)

    n_test = X_test.shape[0]

    for m in ['max', 'min']:
        covered = np.zeros(n_test, dtype=bool)
        lengths = np.zeros(n_test)
        for i in range(n_test):
            s_test_grid = np.abs(y_grid[:, None] - preds_test[i][None, :])  # (K, m)
            denom = sums[None, :] + s_test_grid

            # compute e scores
            e_calib = (s_calib[None, :, :] / denom[:, None, :]).max(axis=2) if m == 'max' else (s_calib[None, :, :] / denom[:, None, :]).min(axis=2)
            e_test = (s_test_grid / denom).max(axis=1) if m == 'max' else (s_test_grid / denom).min(axis=1)
            
            q_grid = np.quantile(e_calib, q_idx, axis=1, method=method)

            mask = (e_test <= q_grid)
            idx_closest = np.abs(y_grid - y_test[i]).argmin()
            covered[i] = mask[idx_closest]
            lengths[i] = mask.sum() * step
        results.append((m, covered.mean(), lengths.mean(), covered))

    total_time = ts.time() - t0
    n_success = len(results)

    results.sort(key=lambda x: x[2])
    p_opt, coverage_opt, length_opt, covered = results[0]


    print(f"\n Grid search complete in {total_time:.2f}s")
    print(f"Number of p-values computed: {n_success}")
    print(f"Optimal p*:       {p_opt}")
    print(f"Mean length*:     {length_opt:.4f}")
    print(f"Coverage:         {coverage_opt:.4f} (target ≥ {1-alpha})")

    
    return p_opt, coverage_opt, length_opt
