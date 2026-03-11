"""Parameter fitting via scipy optimizers with random restarts."""

import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, least_squares


def _random_within_bounds(
    rng: np.random.Generator,
    n_params: int,
    bounds: Optional[List[Tuple[float, float]]],
) -> np.ndarray:
    """Sample a random initial point uniformly within bounds."""
    if bounds is None:
        return rng.standard_normal(n_params) * 0.1
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return lo + (hi - lo) * rng.random(n_params)


# ── Top-level worker functions (must be picklable for ProcessPoolExecutor) ──


def _lbfgsb_one_start(x0, model_fn, X, y, bounds, maxiter, multi_output):
    """Run one L-BFGS-B restart. Returns (loss, theta) or (inf, None)."""
    def objective(theta_flat):
        theta_2d = theta_flat.reshape(1, -1)
        pred = model_fn(theta_2d, X)[0]
        residuals = (pred - y).ravel() if multi_output else pred - y
        return float(np.sum(residuals ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10},
            )
            return (res.fun, res.x.copy())
        except Exception:
            return (np.inf, None)


def _lm_one_start(x0, model_fn, X, y, max_nfev, multi_output):
    """Run one Levenberg-Marquardt restart. Returns (cost, theta) or (inf, None)."""
    def residuals_fn(theta_flat):
        theta_2d = theta_flat.reshape(1, -1)
        pred = model_fn(theta_2d, X)[0]
        return (pred - y).ravel() if multi_output else pred - y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = least_squares(
                residuals_fn,
                x0,
                method="lm",
                max_nfev=max_nfev,
            )
            return (res.cost, res.x.copy())
        except Exception:
            return (np.inf, None)


class LBFGSBFitter:
    """L-BFGS-B optimizer with multiple random restarts."""

    def __init__(self, n_restarts: int = 5, maxiter: int = 2000, seed: int = 42,
                 max_workers: int = 1):
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.seed = seed
        self.max_workers = max_workers

    def fit(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        n_params: int,
        theta0: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        theta0s: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        # Build starting-point list.
        if theta0s is not None:
            starts = [np.asarray(t, dtype=np.float64) for t in theta0s]
        else:
            rng = np.random.default_rng(self.seed)
            starts = []
            if theta0 is not None:
                starts.append(theta0.copy())
            while len(starts) < self.n_restarts:
                starts.append(_random_within_bounds(rng, n_params, bounds))

        multi_output = y.ndim == 2
        best_theta = None
        best_loss = np.inf

        if self.max_workers > 1 and len(starts) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        _lbfgsb_one_start, x0, model_fn, X, y,
                        bounds, self.maxiter, multi_output,
                    ): i
                    for i, x0 in enumerate(starts)
                }
                for future in as_completed(futures):
                    loss, theta = future.result()
                    if theta is not None and loss < best_loss:
                        best_loss = loss
                        best_theta = theta
        else:
            for x0 in starts:
                loss, theta = _lbfgsb_one_start(
                    x0, model_fn, X, y, bounds, self.maxiter, multi_output,
                )
                if theta is not None and loss < best_loss:
                    best_loss = loss
                    best_theta = theta

        if best_theta is None:
            rng = np.random.default_rng(self.seed)
            return _random_within_bounds(rng, n_params, bounds)
        return best_theta


class LMFitter:
    """Levenberg-Marquardt least squares with multiple random restarts."""

    def __init__(self, n_restarts: int = 5, max_nfev: int = 5000, seed: int = 42,
                 max_workers: int = 1):
        self.n_restarts = n_restarts
        self.max_nfev = max_nfev
        self.seed = seed
        self.max_workers = max_workers

    def fit(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        n_params: int,
        theta0: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        theta0s: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        if theta0s is not None:
            starts = [np.asarray(t, dtype=np.float64) for t in theta0s]
        else:
            rng = np.random.default_rng(self.seed)
            starts = []
            if theta0 is not None:
                starts.append(theta0.copy())
            while len(starts) < self.n_restarts:
                starts.append(_random_within_bounds(rng, n_params, bounds))

        multi_output = y.ndim == 2
        best_theta = None
        best_cost = np.inf

        if self.max_workers > 1 and len(starts) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        _lm_one_start, x0, model_fn, X, y,
                        self.max_nfev, multi_output,
                    ): i
                    for i, x0 in enumerate(starts)
                }
                for future in as_completed(futures):
                    cost, theta = future.result()
                    if theta is not None and cost < best_cost:
                        best_cost = cost
                        best_theta = theta
        else:
            for x0 in starts:
                cost, theta = _lm_one_start(
                    x0, model_fn, X, y, self.max_nfev, multi_output,
                )
                if theta is not None and cost < best_cost:
                    best_cost = cost
                    best_theta = theta

        if best_theta is None:
            rng = np.random.default_rng(self.seed)
            return _random_within_bounds(rng, n_params, bounds)
        return best_theta


FITTER_REGISTRY = {
    "lbfgsb": LBFGSBFitter,
    "lm": LMFitter,
}
