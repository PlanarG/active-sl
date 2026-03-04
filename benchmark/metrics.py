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


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error. Handles multi-output by flattening."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def auc_r2(budget_fractions: np.ndarray, r2_values: np.ndarray) -> float:
    """Area under the R²-vs-budget curve using trapezoidal rule."""
    budget_fractions = np.asarray(budget_fractions, dtype=np.float64)
    r2_values = np.asarray(r2_values, dtype=np.float64)
    return float(np.trapezoid(r2_values, budget_fractions))


def budget_to_reach(
    fracs: np.ndarray,
    mse_vals: np.ndarray,
    oracle_mse: float,
    alpha: float,
) -> float:
    """Find min budget fraction where mse <= oracle_mse * (1 + alpha).

    Uses linear interpolation. Returns 1.0 if threshold is never reached.
    """
    threshold = oracle_mse * (1.0 + alpha)
    fracs = np.asarray(fracs, dtype=np.float64)
    mse_vals = np.asarray(mse_vals, dtype=np.float64)

    for i in range(len(mse_vals)):
        if mse_vals[i] <= threshold:
            if i == 0:
                return float(fracs[0])
            f0, f1 = fracs[i - 1], fracs[i]
            v0, v1 = mse_vals[i - 1], mse_vals[i]
            # If previous was inf or values equal, snap to current checkpoint
            if not np.isfinite(v0) or abs(v0 - v1) < 1e-30:
                return float(f1)
            t = (v0 - threshold) / (v0 - v1)
            return float(f0 + t * (f1 - f0))
    return 1.0
