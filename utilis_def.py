import os
import random
import math
import numpy as np
import math
from math import ceil
import time as ts
from itertools import combinations

import pickle
import joblib

from joblib import Parallel, delayed
from numba import njit

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, SGDRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import openml

from modelscifar import resnet56, VGG, ShuffleNetV2, EfficientNetB0, DLA


MODEL_DIR = "./models"
SKLEARN_MODELS  = ['Logistic', 'RandomForest', 'HistGB', 'MLP']
PYTORCH_MODELS = ['ResNet56', 'ShuffleNetV2', 'VGG16_BN', 'DLA', 'EfficientNet']


def set_seed(seed):
    """
        Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    return g


def load_dataset(name_or_id):
    """Load dataset from OpenML by name or ID."""
    task = openml.tasks.get_task(name_or_id)
    features, targets = task.get_X_and_y(dataset_format='dataframe')
    X, y = features.values, targets.values
    return X, y


def split_data(X, y, train_size=0.8, test_size=0.1, random_state=42):
    """
        Split & scale data in train / calib / test
    """
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=random_state)

    categorical_cols = []
    for i in range(X_train.shape[1]):
        if isinstance(X_train[0, i], str):
            categorical_cols.append(i)
    X_train_clean = np.delete(X_train, categorical_cols, axis=1)
    X_temp_clean = np.delete(X_temp, categorical_cols, axis=1)
    te = test_size / (1 - train_size)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp_clean, y_temp, test_size=te, random_state=random_state)
    
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train_clean)
    X_calib = X_scaler.transform(X_calib)
    X_test  = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_calib = y_scaler.transform(y_calib.reshape(-1, 1)).ravel()
    y_test  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
   
    print(f"X_train shape: {X_train.shape}, calib shape: {X_calib.shape}, test shape: {X_test.shape}")
    return (X_train, y_train, X_calib, y_calib, X_test, y_test)


def score_function(y_pred, y_true, regression=True):
    """
        Computes nonconformity scores (NCS) for regression (absolute error) or classification (1 - predicted probability).
    """
    if regression:
        return np.abs(y_true - y_pred)
    return 1.0 - y_pred[np.arange(len(y_true)), y_true]

def compute_scores(models, X, y):
    return np.stack([score_function(m.predict(X), y) for m in models.values()], axis=1)

def split_image_data(X, y, calib_frac=0.5, scale=False, random_state=42):
    """
        Split image data into train/calib/test splits.
        Optionally standardize input features
    """

    # Split into train + temp (calib + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Split temp into calib and test
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=calib_frac, random_state=random_state, stratify=y_temp)

    if scale:
        # Scale X only if using non-image models (e.g., sklearn)
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_calib_flat = X_calib.reshape(len(X_calib), -1)
        X_test_flat  = X_test.reshape(len(X_test), -1)

        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_calib_scaled = scaler.transform(X_calib_flat)
        X_test_scaled  = scaler.transform(X_test_flat)

        return X_train_scaled, y_train, X_calib_scaled, y_calib, X_test_scaled, y_test, scaler
    else:
        return X_train, y_train, X_calib, y_calib, X_test, y_test, None

def _get_model_path(dataset_name, seed, name):
    """
        Get model path to load / save model
    """
    ext = ".pt" if name in PYTORCH_MODELS else ".pkl"
    return os.path.join(MODEL_DIR, f"{dataset_name}_{seed}_{name}{ext}")


def predict_proba_model(model, X, is_image=False, batch_size=128, device=None):
    """
    Returns predicted probabilities for both sklearn and PyTorch models.
    If is_image=True, X should be shaped as (N, C, H, W) and passed through a torch model.
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if not is_image:
        # Reshape flat input to image if needed
        if X.shape[1] == 784:  # FashionMNIST
            X_tensor = X_tensor.view(-1, 1, 28, 28)
        elif X.shape[1] == 3072:  # CIFAR10
            X_tensor = X_tensor.view(-1, 3, 32, 32)
    X_tensor = X_tensor.to(device)

    with torch.no_grad():
        probs = []
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            logits = model(batch)
            batch_probs = torch.softmax(logits, dim=1)
            probs.append(batch_probs.cpu().numpy())
        return np.vstack(probs)


def get_and_fit_models(dataset_name, X_train, y_train, seed=42, epochs=1, batch_size=64, lr=1e-2):
    """
        Loads or trains base models for classificaiton.
    """
    models = {}

    if X_train.ndim == 2:
        sklearn_constructors = {
            'Logistic'    : LogisticRegression(solver='saga', penalty='l2', C=1.0, max_iter=500, n_jobs=-1, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=20, max_features='sqrt', min_samples_leaf=1, n_jobs=-1, random_state=42),
            'HistGB'      : HistGradientBoostingClassifier(max_iter=100, random_state=42, min_samples_leaf=10, l2_regularization=0.1, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20),
            'MLP'         : MLPClassifier(
                            hidden_layer_sizes=(512, 256), activation='relu', solver='adam', learning_rate_init=1e-3,
                            batch_size=128, max_iter=100, random_state=42, verbose=False
                        )
        }

        for name, clf in sklearn_constructors.items():
            path = _get_model_path(dataset_name, seed, name)
            if os.path.exists(path):
                print(f"[sklearn] Loading {name}")
                clf = joblib.load(path)
            else:
                _ts = ts.time()
                print(f"[sklearn] Training {name}")
                clf.fit(X_train, y_train)
                print(ts.time() - _ts)
                joblib.dump(clf, path)
            models[name] = clf

        return models
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_cls = int(y_train.max()) + 1
    
    if X_train.ndim == 4:
        X_t = torch.from_numpy(X_train).float()
    else:
        N, D = X_train.shape
        C = 1 if D == 28*28 else 3
        side = int((D / C) ** 0.5)
        X_t = torch.from_numpy(X_train.reshape(N, C, side, side)).float()
    y_t = torch.from_numpy(y_train).long()

    loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    for name in PYTORCH_MODELS:
        path = _get_model_path(dataset_name, seed, name)
        if name == 'ResNet56':
            model = resnet56(num_classes=num_cls, in_channels=X_t.shape[1])
        elif name == 'ShuffleNetV2':
            model = ShuffleNetV2(net_size=0.5, num_classes=num_cls)
        elif name == 'VGG16_BN':
            model = VGG('VGG16', num_classes=num_cls)
        elif name == 'EfficientNet':
            model = EfficientNetB0(num_classes=num_cls)
        elif name == 'DLA':
            model = DLA(num_classes=num_cls)
        else:
            continue

        model.to(device)

        if os.path.exists(path):
            print(f"[pytorch] Loading {name}")
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"[pytorch] Training {name}")
            crit = nn.CrossEntropyLoss()
            opt  = optim.Adam(model.parameters(), lr=lr)
            model.train()
            for ep in range(1, epochs+1):
                total_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    out = model(xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                print(f"  {name} Ep{ep}/{epochs} loss={total_loss/len(loader):.4f}")
            torch.save(model.state_dict(), path)

        models[name] = model

    return models


def generate_simplex_grid(d, eps=0.1, max_points=200, random_seed=None):
    """
    Generate up to max_points weight vectors w in the d‑simplex on a grid of step eps.
    If max_points is None, returns all grid points (potentially very large).
    Otherwise, uniformly samples max_points points from the full grid without repetition.
    """
    M = int(round(1/eps))
    # total number of grid points = comb(M + d - 1, d - 1)
    total = int(math.comb(M + d - 1, d - 1))

    # if we want all points or total <= max_points, generate exhaustively
    if max_points is None or total <= max_points:
        positions = range(M + d - 1)
        W = []
        for separators in combinations(positions, d - 1):
            coords = (-1,) + separators + (M + d - 1,)
            ks = [coords[i+1] - coords[i] - 1 for i in range(d)]
            W.append(np.array(ks) / M)
        return np.stack(W, axis=0)

    # otherwise, sample uniformly at random from the grid
    rng = np.random.default_rng(random_seed)
    positions = np.arange(M + d - 1)
    samples = set()
    W = []
    while len(W) < max_points:
        # draw d-1 unique separators (stars-and-bars)
        separators = np.sort(rng.choice(positions, size=d-1, replace=False))
        key = tuple(separators.tolist())
        if key in samples:
            continue
        samples.add(key)
        coords = (-1,) + tuple(separators) + (M + d - 1,)
        ks = [coords[i+1] - coords[i] - 1 for i in range(d)]
        W.append(np.array(ks) / M)
    return np.stack(W, axis=0)


## The 20 seeds considered: [42, 0, 1, 7, 10, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]