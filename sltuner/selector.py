from __future__ import annotations

from .parameter import ParameterSpec
from typing import Callable, List, Tuple
from dataclasses import dataclass
from itertools import combinations, product

import math
import numpy as np


@dataclass
class Candidate:
    point: np.ndarray
    cost: float


class Selector:
    """
    Coordinate systems used inside Selector

    1) external:
       Physical parameters consumed by the user-supplied predict_fn.

    2) internal:
       Per-parameter transformed coordinates defined by ParameterSpec
       (currently LINEAR / LOG). This is the coordinate in which `theta`
       is passed into select(...).

    3) search:
       A fixed linear reparameterization of internal increments, learned
       after the pilot design. It is used only inside Selector to stabilize
       FIM geometry, posterior proposals, and batch scoring.

       If delta_search is a step in search coordinates, then
           delta_internal = delta_search @ search_to_internal.T
       and a row-gradient transforms as
           g_search = g_internal @ search_to_internal
    """

    def __init__(
        self,
        parameters: List[ParameterSpec],
        candidates: List[Candidate],
        predict_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
        num_samples: int = 1024,
        sigma2: float = 1e-4,
        rng: np.random.Generator = np.random.default_rng(),
        shortlist_size: int = 20,
        max_enumerated_subsets: int = 5000,
        pilot_ridge_rel: float = 5e-2,
        local_ridge_rel: float = 1e-8,
        trust_tol: float = 5e-2,
    ):
        self.parameters = parameters
        self.candidates = candidates
        self.rng = rng

        self.num_samples = num_samples
        self.sigma2 = sigma2
        self.num_params = len(parameters)
        self.num_candidates = len(candidates)

        self.shortlist_size = shortlist_size
        self.max_enumerated_subsets = max_enumerated_subsets
        self.pilot_ridge_rel = pilot_ridge_rel
        self.local_ridge_rel = local_ridge_rel
        self.trust_tol = trust_tol

        self.candidate_points = np.array([c.point for c in candidates], dtype=float)
        self.candidate_costs = np.array([c.cost for c in candidates], dtype=float)

        self.selected_indices: List[int] = []
        self.pilot_indices: List[int] | None = None
        self.last_scores: np.ndarray | None = None

        # Filled after pilot design is constructed
        self._pilot_gradient_samples: np.ndarray | None = None
        self._pilot_candidate_scores_by_step: List[np.ndarray] | None = None

        # Fixed linear map learned from pilot FIM shape
        self.search_to_internal = np.eye(self.num_params, dtype=float)

        if self.num_candidates < self.num_params:
            raise ValueError("Need at least num_params candidate points.")

        def predict_internal(internal_theta: np.ndarray, data: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Evaluate the model at internal coordinates.

            Parameters
            ----------
            internal_theta : (S, p) or (p,)
                Internal coordinates.
            data : array-like
                Candidate points / observed points.

            Returns
            -------
            objectives : model outputs
            internal_jacs : (S, n_points, p)
                Jacobian with respect to internal coordinates.
            """
            internal_theta = np.asarray(internal_theta, dtype=float)
            if internal_theta.ndim == 1:
                internal_theta = internal_theta[None, :]

            theta_external = np.column_stack([
                param.to_external(internal_theta[:, j]) for j, param in enumerate(self.parameters)
            ])

            objectives, external_jacs = predict_fn(theta_external, data)
            external_jacs = np.asarray(external_jacs, dtype=float).copy()

            for j, param in enumerate(self.parameters):
                _, external_jacs[:, :, j] = param.to_internal(theta_external[:, j], external_jacs[:, :, j])

            return objectives, external_jacs

        self.predict_internal = predict_internal

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _symmetrize(M: np.ndarray) -> np.ndarray:
        return 0.5 * (M + M.T)

    def _matrix_scale(self, M: np.ndarray) -> float:
        scale = float(np.trace(M) / M.shape[0])
        if not np.isfinite(scale) or scale <= 0:
            return 1.0
        return scale

    def _stable_cholesky(
        self,
        M: np.ndarray,
        rel_ridge: float | None = None,
        max_tries: int = 8,
    ) -> Tuple[np.ndarray, float]:
        """
        Cholesky with adaptive relative ridge:
            ridge = rel_ridge * tr(M)/p
        and automatic escalation if needed.
        """
        M = self._symmetrize(np.asarray(M, dtype=float))
        p = M.shape[0]
        rel_ridge = self.local_ridge_rel if rel_ridge is None else rel_ridge

        ridge = max(rel_ridge * self._matrix_scale(M), np.finfo(float).eps)
        I = np.eye(p, dtype=float)

        for _ in range(max_tries):
            try:
                L = np.linalg.cholesky(M + ridge * I)
                return L, ridge
            except np.linalg.LinAlgError:
                ridge *= 10.0

        evals, _ = np.linalg.eigh(M)
        ridge = max(ridge, -float(evals.min()) + rel_ridge * self._matrix_scale(M))
        L = np.linalg.cholesky(M + ridge * I)
        return L, ridge

    def _internal_bounds_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        low = np.array([p.internal_bounds()[0] for p in self.parameters], dtype=float)
        high = np.array([p.internal_bounds()[1] for p in self.parameters], dtype=float)
        return low, high

    def _as_internal_vector(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.shape[0] != self.num_params:
            raise ValueError(f"theta must have shape ({self.num_params},), got {theta.shape}")
        return theta

    def _to_search_grads(self, grads_internal: np.ndarray) -> np.ndarray:
        """
        Convert row-gradients from internal coordinates to search coordinates.
        Works for arrays with shape (..., p).
        """
        return np.asarray(grads_internal, dtype=float) @ self.search_to_internal

    def _search_delta_to_internal(self, delta_search: np.ndarray) -> np.ndarray:
        """
        Convert search-coordinate row-steps to internal-coordinate row-steps.
        Works for arrays with shape (..., p).
        """
        return np.asarray(delta_search, dtype=float) @ self.search_to_internal.T

    # ------------------------------------------------------------------
    # Pilot design
    # ------------------------------------------------------------------

    def _build_pilot_indices(self) -> List[int]:
        """
        Build the initial pivot / pilot design.

        This uses prior samples over internal parameters and greedily picks
        points whose gradients add the most new directional coverage,
        measured by squared residual norm after projection onto the
        already-selected basis.
        """
        samples_per_param = math.ceil(self.num_samples ** (1 / self.num_params))
        theta_samples = np.array(list(product(*[
            param.sample(self.rng, samples_per_param) for param in self.parameters
        ])), dtype=float)

        gradient_samples = np.asarray(
            self.predict_internal(theta_samples, self.candidate_points)[1],
            dtype=float
        )  # (S, C, p)

        num_theta_samples = gradient_samples.shape[0]
        pilot_basis = np.zeros((num_theta_samples, self.num_params, self.num_params), dtype=float)
        pilot_ranks = np.zeros(num_theta_samples, dtype=int)

        available = list(range(self.num_candidates))
        chosen: List[int] = []
        candidate_scores_by_step: List[np.ndarray] = []

        for _ in range(self.num_params):
            mean_scores = np.zeros(self.num_candidates, dtype=float)

            for s in range(num_theta_samples):
                G = gradient_samples[s]  # (C, p)
                rank = pilot_ranks[s]

                if rank > 0:
                    Q = pilot_basis[s, :, :rank]     # (p, rank)
                    residual = G - (G @ Q) @ Q.T
                else:
                    residual = G

                residual_sq = np.sum(residual * residual, axis=1)
                mean_scores += residual_sq

            mean_scores /= max(num_theta_samples, 1)
            mean_scores /= np.maximum(self.candidate_costs, np.finfo(float).eps)
            candidate_scores_by_step.append(mean_scores.copy())

            best_idx = available[int(np.argmax(mean_scores[available]))]
            chosen.append(best_idx)
            available.remove(best_idx)

            for s in range(num_theta_samples):
                g = gradient_samples[s, best_idx]
                rank = pilot_ranks[s]

                if rank > 0:
                    Q = pilot_basis[s, :, :rank]
                    residual = g - Q @ (Q.T @ g)
                else:
                    residual = g

                norm = np.linalg.norm(residual)
                if norm > 1e-12 and rank < self.num_params:
                    pilot_basis[s, :, rank] = residual / norm
                    pilot_ranks[s] += 1

        self._pilot_gradient_samples = gradient_samples
        self._pilot_candidate_scores_by_step = candidate_scores_by_step
        return chosen

    def _fit_search_preconditioner(self) -> None:
        """
        Learn a fixed linear map from pilot FIM shape:

            search_to_internal = (M_pilot + lambda I)^(-1/2)

        but with per-sample trace normalization before averaging, so this map
        captures average geometric shape rather than being dominated by
        absolute gradient amplitudes.
        """
        pilot_sel = np.asarray(self.pilot_indices, dtype=int)
        G_pilot = self._pilot_gradient_samples[:, pilot_sel, :]  # (S, p, p) if len(pilot)==p, but keep general

        pilot_fims = np.einsum("snp,snq->spq", G_pilot, G_pilot, optimize=True) / self.sigma2
        pilot_fims = 0.5 * (pilot_fims + np.swapaxes(pilot_fims, 1, 2))

        # Normalize each sample by its own average eigen-scale
        sample_scales = np.trace(pilot_fims, axis1=1, axis2=2) / self.num_params
        sample_scales = np.maximum(sample_scales, np.finfo(float).eps)

        mean_pilot_fim = np.mean(pilot_fims / sample_scales[:, None, None], axis=0)
        mean_pilot_fim = self._symmetrize(mean_pilot_fim)

        evals, evecs = np.linalg.eigh(mean_pilot_fim)
        ridge = self.pilot_ridge_rel * self._matrix_scale(mean_pilot_fim)

        # Regularized whitening:
        #   lambda_i -> lambda_i / (lambda_i + ridge)
        scales = 1.0 / np.sqrt(np.maximum(evals, 0.0) + ridge)
        self.search_to_internal = evecs @ np.diag(scales) @ evecs.T

        self._pilot_fim_search = mean_pilot_fim
        self._pilot_fim_eigenvalues = evals
        self._pilot_fim_ridge = ridge

    # ------------------------------------------------------------------
    # Local proposal geometry around current theta
    # ------------------------------------------------------------------

    def _estimate_search_step_limits(
        self,
        theta_internal: np.ndarray,
        search_basis_directions_internal: np.ndarray,
        observed_points: np.ndarray,
        tol: float | None = None,
        initial_step: float = 0.01,
        max_doublings: int = 30,
    ) -> np.ndarray:
        """
        Estimate, for each search-basis direction, how far we can move before
        the observed FIM geometry changes too much.

        search_basis_directions_internal[:, i] is the internal-coordinate
        direction corresponding to a unit move along search-basis axis i.
        """
        tol = self.trust_tol if tol is None else tol
        p = self.num_params

        J0_internal = np.asarray(
            self.predict_internal(theta_internal.reshape(1, p), observed_points)[1][0],
            dtype=float
        )
        J0_search = self._to_search_grads(J0_internal)
        M0_search = self._symmetrize(J0_search.T @ J0_search / self.sigma2)

        ref_norm = max(np.linalg.norm(M0_search, ord="fro"), self._matrix_scale(M0_search))
        low, high = self._internal_bounds_arrays()
        step_limits = np.empty(p, dtype=float)

        for i in range(p):
            direction_internal = search_basis_directions_internal[:, i]
            step = initial_step

            for _ in range(max_doublings):
                worst_relative_change = 0.0

                for sign in (+1.0, -1.0):
                    theta_trial = np.clip(theta_internal + sign * step * direction_internal, low, high)

                    J_trial_internal = np.asarray(
                        self.predict_internal(theta_trial.reshape(1, p), observed_points)[1][0],
                        dtype=float
                    )
                    J_trial_search = self._to_search_grads(J_trial_internal)
                    M_trial_search = self._symmetrize(J_trial_search.T @ J_trial_search / self.sigma2)

                    relative_change = np.linalg.norm(M_trial_search - M0_search, ord="fro") / ref_norm
                    worst_relative_change = max(worst_relative_change, relative_change)

                if worst_relative_change > tol:
                    step_limits[i] = step * tol / max(worst_relative_change, 1e-12)
                    break

                step *= 2.0
            else:
                step_limits[i] = step

        return step_limits

    def _build_posterior_cache(
        self,
        theta_internal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the adaptive-design cache around the current theta.

        Returns
        -------
        single_point_gain : (C,)
            Posterior-averaged single-point D-gain surrogate:
                E[ log(1 + ||u_i||^2) ]
            where u_i is the candidate gradient whitened by the current
            observed information at each sampled theta.

        whitened_gradients : (S, C, p)
            For each sampled theta and each candidate, the gradient whitened
            by the sampled observed-information Cholesky factor.
        """
        observed_indices = np.asarray(self.selected_indices, dtype=int)
        observed_points = self.candidate_points[observed_indices]

        # 1) Build a local Gaussian-like proposal in search coordinates
        J_obs_internal = np.asarray(
            self.predict_internal(theta_internal.reshape(1, self.num_params), observed_points)[1][0],
            dtype=float
        )
        J_obs_search = self._to_search_grads(J_obs_internal)
        proposal_fim_search = self._symmetrize(J_obs_search.T @ J_obs_search / self.sigma2)

        _, proposal_ridge = self._stable_cholesky(proposal_fim_search)
        proposal_evals, proposal_evecs = np.linalg.eigh(
            proposal_fim_search + proposal_ridge * np.eye(self.num_params)
        )

        # A unit move along search eigenvector v corresponds to internal move S v
        search_basis_directions_internal = self.search_to_internal @ proposal_evecs
        step_limits = self._estimate_search_step_limits(
            theta_internal=theta_internal,
            search_basis_directions_internal=search_basis_directions_internal,
            observed_points=observed_points,
        )

        gaussian_std = 1.0 / np.sqrt(np.maximum(proposal_evals, np.finfo(float).eps))
        search_step_std = np.minimum(gaussian_std, step_limits)

        standard_normal = self.rng.standard_normal((self.num_samples, self.num_params))
        delta_search = (standard_normal * search_step_std[None, :]) @ proposal_evecs.T

        low, high = self._internal_bounds_arrays()
        theta_samples_internal = theta_internal[None, :] + self._search_delta_to_internal(delta_search)
        theta_samples_internal = np.clip(theta_samples_internal, low, high)

        # 2) At each sampled theta, whiten every candidate gradient by that theta's
        #    sampled observed information.
        all_grads_internal = np.asarray(
            self.predict_internal(theta_samples_internal, self.candidate_points)[1],
            dtype=float
        )  # (S, C, p)
        all_grads_search = self._to_search_grads(all_grads_internal)

        whitened_gradients = np.empty_like(all_grads_search)
        single_point_gain = np.zeros(self.num_candidates, dtype=float)

        for s in range(self.num_samples):
            G_obs_search = all_grads_search[s, observed_indices, :]
            sampled_fim_search = self._symmetrize(G_obs_search.T @ G_obs_search / self.sigma2)

            chol, _ = self._stable_cholesky(sampled_fim_search)

            # u_i = L^{-1} g_i / sqrt(sigma2)
            U = np.linalg.solve(chol, (all_grads_search[s].T / np.sqrt(self.sigma2))).T
            whitened_gradients[s] = U

            single_point_gain += np.log1p(np.einsum("ij,ij->i", U, U, optimize=True))

        single_point_gain /= self.num_samples
        return single_point_gain, whitened_gradients

    # ------------------------------------------------------------------
    # Shortlist + exact small-batch enumeration
    # ------------------------------------------------------------------

    def _build_shortlist(
        self,
        available_indices: List[int],
        whitened_gradients: np.ndarray,
        batch_size: int,
    ) -> List[int]:
        """
        Build a diversified shortlist.

        This is NOT "take top-L single-point scores".
        Instead, it greedily adds points whose marginal posterior-averaged
        logdet gain, given the points already in the shortlist, is largest
        per unit cost.

        That makes the shortlist itself diversity-aware, so it is much less
        likely to collapse onto many near-duplicate points around the current
        best single-point location.
        """
        if batch_size > len(available_indices):
            raise ValueError("batch_size exceeds the number of available candidates.")

        shortlist_size = min(max(batch_size, self.shortlist_size), len(available_indices))
        while shortlist_size > batch_size and math.comb(shortlist_size, batch_size) > self.max_enumerated_subsets:
            shortlist_size -= 1

        if shortlist_size == len(available_indices):
            return list(available_indices)

        available = np.asarray(available_indices, dtype=int)
        chosen_local: List[int] = []

        # For each sampled theta, maintain:
        #   H^{-1} = (I + sum_{j in shortlist} u_j u_j^T)^(-1)
        # Start with H = I.
        Hinv = np.repeat(np.eye(self.num_params, dtype=float)[None, :, :], self.num_samples, axis=0)

        remaining_local = np.arange(len(available), dtype=int)

        for _ in range(shortlist_size):
            candidate_ids = available[remaining_local]                          # (m,)
            U = whitened_gradients[:, candidate_ids, :]                         # (S, m, p)
            HU = np.einsum("spq,smq->smp", Hinv, U, optimize=True)              # (S, m, p)
            quad = np.einsum("smp,smp->sm", U, HU, optimize=True)               # (S, m)

            marginal_gain = np.mean(np.log1p(np.maximum(quad, 0.0)), axis=0)    # (m,)
            marginal_score = marginal_gain / np.maximum(self.candidate_costs[candidate_ids], np.finfo(float).eps)

            best_pos = int(np.argmax(marginal_score))
            best_local = int(remaining_local[best_pos])
            chosen_local.append(best_local)

            chosen_id = available[best_local]
            u = whitened_gradients[:, chosen_id, :]                             # (S, p)
            Hu = np.einsum("spq,sq->sp", Hinv, u, optimize=True)                # (S, p)
            denom = 1.0 + np.einsum("sp,sp->s", u, Hu, optimize=True)           # (S,)

            Hinv -= np.einsum("sp,sq->spq", Hu, Hu, optimize=True) / denom[:, None, None]
            remaining_local = remaining_local[remaining_local != best_local]

        return available[chosen_local].tolist()

    def _score_candidate_batch(
        self,
        candidate_batch: List[int],
        whitened_gradients: np.ndarray,
    ) -> float:
        """
        Exact objective for a small candidate batch:
            average_s log det(I + U_s^T U_s) / total_cost
        where U_s stacks the whitened gradients of the batch at sampled theta s.
        """
        subset = np.asarray(candidate_batch, dtype=int)
        U = whitened_gradients[:, subset, :]  # (S, b, p)

        H = np.eye(self.num_params, dtype=float)[None, :, :] + np.einsum(
            "sbp,sbq->spq", U, U, optimize=True
        )  # (S, p, p)

        sign, logdet = np.linalg.slogdet(H)
        if np.any(sign <= 0):
            return -np.inf

        mean_joint_gain = float(np.mean(logdet))
        total_cost = float(np.sum(self.candidate_costs[subset]))
        return mean_joint_gain / max(total_cost, np.finfo(float).eps)

    def _choose_adaptive_batch(
        self,
        theta_internal: np.ndarray,
        batch_size: int,
    ) -> List[int]:
        available_indices = [i for i in range(self.num_candidates) if i not in set(self.selected_indices)]
        if batch_size > len(available_indices):
            raise ValueError("batch_size exceeds the number of available candidates.")

        if batch_size == len(available_indices):
            return available_indices

        single_point_gain, whitened_gradients = self._build_posterior_cache(theta_internal)

        # Keep this for inspection / plotting; it remains a useful 1-point diagnostic.
        single_point_score = single_point_gain / np.maximum(self.candidate_costs, np.finfo(float).eps)
        self.last_scores = np.full(self.num_candidates, np.nan, dtype=float)
        self.last_scores[available_indices] = single_point_score[available_indices]

        shortlist = self._build_shortlist(
            available_indices=available_indices,
            whitened_gradients=whitened_gradients,
            batch_size=batch_size,
        )

        best_batch = shortlist[:batch_size]
        best_score = -np.inf

        for batch_indices in combinations(shortlist, batch_size):
            score = self._score_candidate_batch(
                candidate_batch=list(batch_indices),
                whitened_gradients=whitened_gradients,
            )
            if score > best_score:
                best_score = score
                best_batch = list(batch_indices)

        return best_batch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, theta: np.ndarray | None = None, batch: int = 1) -> int | List[int]:
        """
        Select the next point(s).

        Parameters
        ----------
        theta : (p,), optional
            Current INTERNAL-parameter estimate. Required after the pilot phase.
        batch : int
            Number of points to select.
            - batch=1: returns a single int
            - batch>1: returns a list[int]

        Important assumption
        --------------------
        This class assumes all previously selected points have already been
        observed and are included in the theta used here.
        Therefore a single call is NOT allowed to cross from pilot phase into
        adaptive phase.
        """
        
        if self.pilot_indices is None:
            self.pilot_indices = self._build_pilot_indices()
            self._fit_search_preconditioner()

        num_selected = len(self.selected_indices)
        num_pilot = len(self.pilot_indices)
        remaining_pilot = num_pilot - num_selected

        # Pilot phase: only return pilot points. Do not cross into adaptive
        # phase inside the same call; that would treat not-yet-observed pilot
        # points as if they were already observed.
        if remaining_pilot > 0:
            if batch > remaining_pilot:
                raise ValueError(
                    "batch crosses the pilot/adaptive boundary. "
                    "Request at most the remaining pilot points, observe them, "
                    "refit theta, then call select(...) again."
                )

            chosen = self.pilot_indices[num_selected:num_selected + batch]
            self.last_scores = self._pilot_candidate_scores_by_step[num_selected]
            self.selected_indices.extend(chosen)

            return chosen[0] if batch == 1 else chosen

        # Adaptive phase
        if theta is None:
            raise ValueError("theta must be provided after the pilot phase.")

        theta_internal = self._as_internal_vector(theta)
        chosen = self._choose_adaptive_batch(theta_internal=theta_internal, batch_size=batch)
        self.selected_indices.extend(chosen)

        return chosen[0] if batch == 1 else chosen