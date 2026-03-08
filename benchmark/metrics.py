"""Evaluation metrics for scaling law benchmark."""

import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score clipped to [-1, 1]. Handles multi-output by flattening."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-30:
        r2 = 1.0 if ss_res < 1e-30 else -1.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return float(np.clip(r2, -1.0, 1.0))


def log_auc_r2(r2_values: np.ndarray) -> float:
    """Mean R² across log-spaced budget checkpoints."""
    r2_values = np.asarray(r2_values, dtype=np.float64)
    return float(r2_values.mean())
