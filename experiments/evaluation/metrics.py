from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def compute_pearson(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        r = pearsonr(y_pred, y_true)[0]
    if np.isnan(r):
        return -1.0  # Constant predictions: no correlation; treat as worst for early stopping.
    return float(r)


def compute_spearman(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        r = spearmanr(y_pred, y_true)[0]
    if np.isnan(r):
        return -1.0
    return float(r)


def compute_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(((y_pred - y_true) ** 2).mean())


def compute_cosine_error(
    z1: torch.Tensor, z2: torch.Tensor, y_true: np.ndarray
) -> float:
    cos_sim = torch.nn.functional.cosine_similarity(z1, z2).cpu().numpy()
    return float(np.mean((cos_sim - y_true) ** 2))


def summarize_metrics(
    scores: Dict[str, float],
) -> Dict[str, float]:
    return scores

