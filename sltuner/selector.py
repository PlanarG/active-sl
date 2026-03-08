from .parameter import ParameterSpec
from typing import List, Callable, Tuple
from dataclasses import dataclass
from itertools import product

import numpy as np
import math

@dataclass
class Candidate:
    point: np.ndarray
    cost: float

class Selector:
    def __init__(
        self,
        parameters:  List[ParameterSpec],
        candidates:  List[Candidate],
        predict_fn:  Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
        num_samples: int = 1024,
        sigma2:      float = 1e-4,
        rng:         np.random.Generator = np.random.default_rng()
    ):
        self.parameters     = parameters
        self.candidates     = candidates
        self.rng            = rng
        self.num_samples    = num_samples
        self.num_params     = len(parameters)
        self.num_candidates = len(candidates)
        self.trials         = []
        self.initial_trails = None
        self.step           = 0
        self.sigma2         = sigma2
        self.last_scores    = None  # scores / cost for the most recent select() call

        if self.num_candidates < self.num_params:
            raise NotImplementedError

        # objective and jacobian
        def predict(inner_theta: np.ndarray, data: np.ndarray) -> Tuple[float, np.ndarray]:
            """Predict the cost at the given internal point."""
            theta = np.column_stack([param.to_external(inner_theta[:, i]) for i, param in enumerate(self.parameters)])
            objectives, ext_jacs = predict_fn(theta, data)

            for i, param in enumerate(self.parameters):
                _, ext_jacs[:, :, i] = param.to_internal(theta[:, i], ext_jacs[:, :, i])

            return objectives, ext_jacs

        self.predict_fn = predict

    def _make_initial_trials(self) -> List[int]:
        X     = np.array([candidate.point for candidate in self.candidates])
        costs = np.array([candidate.cost  for candidate in self.candidates])

        samples_per_param = math.ceil(self.num_samples ** (1 / self.num_params))
        thetas     = np.array(list(product(*[param.sample(self.rng, samples_per_param) for param in self.parameters])))
        num_thetas = thetas.shape[0]

        all_grads = self.predict_fn(thetas, X)[1] # shape (num_thetas, num_candidates, num_params)
        self._init_all_grads = all_grads
        self._init_num_thetas = num_thetas
        self._init_costs = costs
        basis     = np.zeros((num_thetas, self.num_params, 0))
        self._init_basis = basis
        avail_idx = [i for i in range(self.num_candidates)]
        selected  = []
        self._init_scores_per_step = []

        for _ in range(self.num_params):
            avg_scores = np.zeros(self.num_candidates)
            for i in range(num_thetas):
                G = all_grads[i]
                residual = G - G @ basis[i] @ basis[i].T if basis.shape[2] > 0 else G
                avg_scores += np.linalg.norm(residual, axis=1)

            avg_scores = avg_scores / num_thetas / costs
            self._init_scores_per_step.append(avg_scores.copy())

            best_idx = avail_idx[np.argmax(avg_scores[avail_idx])]
            selected.append(best_idx)
            avail_idx.remove(best_idx)

            for i in range(num_thetas):
                g = all_grads[i][best_idx]
                residual = g - basis[i] @ basis[i].T @ g if basis.shape[2] > 0 else g
                norm = np.linalg.norm(residual)
                if norm > 1e-9:
                    basis[i] = np.column_stack([basis[i], residual / norm])

        return selected
    
    def _estimate_trust_radii(
        self,
        theta: np.ndarray,
        eigvecs: np.ndarray,
        X_obs: np.ndarray,
        tol: float = 1e-2,
        h_init: float = 1e-4,
        max_doublings: int = 100,
    ) -> np.ndarray:
        """Estimate trust radius per eigen-direction by measuring FIM stability.

        Criterion: find largest δ such that the relative change in FIM satisfies
            ‖FIM(θ + δ·e_i) - FIM(θ)‖_F  /  ‖FIM(θ)‖_F  <  tol

        This is the right metric because:
        - Frobenius norm of FIM = ‖JᵀJ‖_F directly bounds the change in
            all eigenvalues (Weyl's inequality);
        - it ignores J changes in the FIM null-space;
        - a pure rotation of J (‖ΔJ‖=0 but FIM changes) is correctly caught.

        Uses geometric search: start from h_init, double until tol is violated,
        then bisect once.
        """
        p = self.num_params
        theta_batch = theta.reshape(1, p)
        J0 = self.predict_fn(theta_batch, X_obs)[1][0]   # (N, p)
        FIM0 = J0.T @ J0                                  # (p, p)
        FIM0_norm = np.linalg.norm(FIM0, 'fro')

        if FIM0_norm < 1e-15:
            return np.full(p, 1.0)

        radii = np.empty(p)

        for i in range(p):
            e_i = eigvecs[:, i]
            h = h_init

            # Geometric growth: find h where FIM changes too much
            for _ in range(max_doublings):
                theta_pert = theta + h * e_i
                J_pert = self.predict_fn(theta_pert.reshape(1, p), X_obs)[1][0]
                FIM_pert = J_pert.T @ J_pert
                rel_change = np.linalg.norm(FIM_pert - FIM0, 'fro') / FIM0_norm

                if rel_change > tol:
                    # Overshoot — trust radius is between h/2 and h.
                    # Linear interpolation: δ ≈ h * tol / rel_change
                    radii[i] = h * tol / rel_change
                    break
                h *= 2.0
            else:
                radii[i] = h  # still linear at max step

        return radii

    def _get_optimal_idx(self, theta: np.ndarray = None) -> int:
        selected_set = set(self.trials[:self.step])
        avail_idx = [i for i in range(self.num_candidates) if i not in selected_set]
        if theta is None:
            # fall back to random selection with probability proportional to 1/cost
            costs = np.array([self.candidates[i].cost for i in avail_idx])
            probabilities = 1 / costs
            probabilities /= probabilities.sum()
            self.last_scores = np.full(self.num_candidates, np.nan)
            for j, i in enumerate(avail_idx):
                self.last_scores[i] = probabilities[j]
            return self.rng.choice(avail_idx, p=probabilities)

        observed_idx = self.trials[:self.step]
        X_all = np.array([c.point for c in self.candidates])
        costs = np.array([c.cost for c in self.candidates])

        theta_1 = theta.reshape(1, self.num_params)
        G_obs = self.predict_fn(theta_1, X_all[observed_idx])[1][0]      # (C, p)
        FIM = G_obs.T @ G_obs / self.sigma2
        FIM = 0.5 * (FIM + FIM.T) 

        eigvals, eigvecs = np.linalg.eigh(FIM)

        # Reparameterization: theta_s = theta + V diag(1 / sqrt(lambda)) z, z ~ N(0, I)
        X_obs = np.array([self.candidates[i].point for i in observed_idx])
        trust_radii = self._estimate_trust_radii(theta, eigvecs, X_obs)
        
        scale = np.minimum(1 / np.sqrt(np.maximum(eigvals, 1e-5)), trust_radii)
        z = self.rng.standard_normal((self.num_samples, self.num_params))
        sampled_thetas = theta[None, :] + (z * scale[None, :]) @ eigvecs.T
        lo = np.array([p.internal_bounds()[0] for p in self.parameters])
        hi = np.array([p.internal_bounds()[1] for p in self.parameters])
        sampled_thetas = np.clip(sampled_thetas, lo, hi)

        all_grads = self.predict_fn(sampled_thetas, X_all)[1]
        scores = np.zeros(self.num_candidates)

        for s in range(self.num_samples):
            G_s = all_grads[s][observed_idx]
            FIM_s = G_s.T @ G_s / self.sigma2 + 1e-5 * np.eye(self.num_params)
            FIM_s = 0.5 * (FIM_s + FIM_s.T)  # ensure symmetry

            evals, evecs = np.linalg.eigh(FIM_s)
            mask = evals < 1e5
            if not mask.any():
                continue

            V   = evecs[:, mask]
            lam = evals[mask]
            inv_lam = 1.0 / lam
            G_proj = all_grads[s] @ V
            gains = np.sum(G_proj ** 2 * inv_lam[None, :], axis=1) / self.sigma2
            scores += np.log1p(gains)

        self.last_scores = scores / costs
        best_idx = max(avail_idx, key=lambda i: self.last_scores[i])
        return best_idx

    def select(self, theta: np.ndarray = None) -> int:
        if self.initial_trails is None:
            self.initial_trails = self._make_initial_trials()

        if self.step < len(self.initial_trails):
            self.last_scores = self._init_scores_per_step[self.step]
            idx = self.initial_trails[self.step]
        else:
            idx = self._get_optimal_idx(theta)

        self.trials.append(idx)
        self.step += 1
        return idx