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


class LBFGSBFitter:
    """L-BFGS-B optimizer with multiple random restarts."""

    def __init__(self, n_restarts: int = 5, maxiter: int = 2000, seed: int = 42):
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.seed = seed

    def fit(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        n_params: int,
        theta0: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        best_theta = None
        best_loss = np.inf

        multi_output = y.ndim == 2

        def objective(theta_flat):
            theta_2d = theta_flat.reshape(1, -1)
            pred = model_fn(theta_2d, X)
            if multi_output:
                residuals = (pred - y).ravel()
            else:
                residuals = pred - y
            return float(np.sum(residuals ** 2))

        for i in range(self.n_restarts):
            if i == 0 and theta0 is not None:
                x0 = theta0.copy()
            else:
                x0 = _random_within_bounds(rng, n_params, bounds)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = minimize(
                        objective,
                        x0,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": self.maxiter, "ftol": 1e-15, "gtol": 1e-10},
                    )
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_theta = res.x.copy()
                except Exception:
                    continue

        if best_theta is None:
            return _random_within_bounds(rng, n_params, bounds)
        return best_theta


class LMFitter:
    """Levenberg-Marquardt least squares with multiple random restarts."""

    def __init__(self, n_restarts: int = 5, max_nfev: int = 5000, seed: int = 42):
        self.n_restarts = n_restarts
        self.max_nfev = max_nfev
        self.seed = seed

    def fit(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        n_params: int,
        theta0: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        best_theta = None
        best_cost = np.inf

        multi_output = y.ndim == 2

        def residuals_fn(theta_flat):
            theta_2d = theta_flat.reshape(1, -1)
            pred = model_fn(theta_2d, X)
            if multi_output:
                return (pred - y).ravel()
            return pred - y

        for i in range(self.n_restarts):
            if i == 0 and theta0 is not None:
                x0 = theta0.copy()
            else:
                x0 = _random_within_bounds(rng, n_params, bounds)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = least_squares(
                        residuals_fn,
                        x0,
                        method="lm",
                        max_nfev=self.max_nfev,
                    )
                    if res.cost < best_cost:
                        best_cost = res.cost
                        best_theta = res.x.copy()
                except Exception:
                    continue

        if best_theta is None:
            return _random_within_bounds(rng, n_params, bounds)
        return best_theta


FITTER_REGISTRY = {
    "lbfgsb": LBFGSBFitter,
    "lm": LMFitter,
}
