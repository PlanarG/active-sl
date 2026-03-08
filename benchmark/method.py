"""Selection methods for the scaling law benchmark."""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class SelectionState:
    candidate_indices: np.ndarray   # indices into training set not yet selected
    observed_indices: np.ndarray    # indices into training set already selected
    current_theta: Optional[np.ndarray]  # current fitted parameters
    spent_budget: float             # budget consumed so far
    total_budget: float             # total budget cap
    cost_per_point: np.ndarray      # cost of each training point (full array)
    rng: np.random.Generator
    method_state: dict = field(default_factory=dict)
    X_train: Optional[np.ndarray] = None
    model_fn: Optional[Callable] = None
    n_params: int = 0
    param_bounds: Optional[list] = None


# ── Shared helpers ────────────────────────────────────────────────────────────

def _affordable_candidates(state: SelectionState):
    """Return (affordable_candidates, affordable_costs) arrays."""
    remaining = state.total_budget - state.spent_budget
    costs = state.cost_per_point[state.candidate_indices]
    mask = costs <= remaining + 1e-12
    return state.candidate_indices[mask], costs[mask]


def _random_theta(state):
    lo = np.array([b[0] for b in state.param_bounds])
    hi = np.array([b[1] for b in state.param_bounds])
    return lo + (hi - lo) * state.rng.random(state.n_params)


def _inverse_cost_fallback(state, affordable_candidates, affordable_costs):
    """Pick one point with prob ~ 1/cost (used as fallback)."""
    inv_c = 1.0 / np.maximum(affordable_costs, 1e-30)
    probs = inv_c / inv_c.sum()
    idx = state.rng.choice(len(affordable_candidates), p=probs)
    return np.array([affordable_candidates[idx]])

def _jac_batch_fd(model_fn, theta, X, eps=1e-5):
    """Finite-difference Jacobians for all rows of X.  Returns (M, D, P)."""
    n_p = len(theta)
    partials = []
    for i in range(n_p):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            fp = np.asarray(model_fn(tp.reshape(1, -1), X), dtype=np.float64)
            fm = np.asarray(model_fn(tm.reshape(1, -1), X), dtype=np.float64)
        partials.append((fp - fm) / (2.0 * eps))
    stacked = np.stack(partials, axis=-1)       # (M, P) or (M, D, P)
    if stacked.ndim == 2:
        stacked = stacked[:, np.newaxis, :]     # (M, 1, P)
    return np.where(np.isfinite(stacked), stacked, 0.0)


# Cache of JIT-compiled forward-mode Jacobian functions, keyed by model identity.
# jacfwd needs P JVPs (one per parameter); jacobian (reverse) needs M*D VJPs.
# Since P << M, forward-mode is much faster for this problem.



def _jac_batch(model_fn, theta, X, eps=1e-5):
    """Compute Jacobians for all rows of X via finite differences.  Returns (M, D, P)."""
    return _jac_batch_fd(model_fn, theta, X, eps)

def _build_fim(jac_all, sel_idx, n_p, reg=1e-6, inv_s2=1.0):
    """FIM = reg*I + inv_s2 * sum J_i^T J_i."""
    F = np.eye(n_p) * reg
    for idx in sel_idx:
        J = jac_all[int(idx)]
        F += inv_s2 * (J.T @ J)
    return F


# ── RandomMethod ──────────────────────────────────────────────────────────────

class RandomMethod:
    """Randomly select one affordable candidate point."""

    def propose(self, state: SelectionState) -> np.ndarray:
        remaining = state.total_budget - state.spent_budget
        affordable = state.cost_per_point[state.candidate_indices] <= remaining + 1e-12
        affordable_candidates = state.candidate_indices[affordable]
        if len(affordable_candidates) == 0:
            return np.array([], dtype=int)
        idx = state.rng.choice(len(affordable_candidates))
        return np.array([affordable_candidates[idx]])


# ── InverseCostMethod ─────────────────────────────────────────────────────────

class InverseCostMethod:
    """Select one affordable candidate with probability proportional to 1/cost."""

    def propose(self, state: SelectionState) -> np.ndarray:
        remaining = state.total_budget - state.spent_budget
        costs = state.cost_per_point[state.candidate_indices]
        affordable = costs <= remaining + 1e-12
        affordable_candidates = state.candidate_indices[affordable]
        if len(affordable_candidates) == 0:
            return np.array([], dtype=int)
        inv_costs = 1.0 / np.maximum(costs[affordable], 1e-30)
        probs = inv_costs / inv_costs.sum()
        idx = state.rng.choice(len(affordable_candidates), p=probs)
        return np.array([affordable_candidates[idx]])


# ── FIMMethod (greedy delta-log-det-FIM / cost) ──────────────────────────────

class FIMMethod:
    """Select candidate maximising delta log det FIM / cost.

    Falls back to inverse-cost sampling when no fitted theta is available yet.
    """

    def __init__(self, jac_eps: float = 1e-5, reg: float = 1e-6):
        self.jac_eps = jac_eps
        self.reg = reg

    def _rebuild_cache(self, state, theta):
        """Recompute Jacobians for all training points at the current theta."""
        ms = state.method_state
        ms["jac_all"] = _jac_batch(state.model_fn, theta, state.X_train, self.jac_eps)
        ms["jac_theta"] = theta.copy()

    def propose(self, state: SelectionState) -> np.ndarray:
        aff_cand, aff_costs = _affordable_candidates(state)
        if len(aff_cand) == 0:
            return np.array([], dtype=int)
        if state.current_theta is None or state.model_fn is None:
            return _inverse_cost_fallback(state, aff_cand, aff_costs)

        theta = state.current_theta
        ms = state.method_state
        prev = ms.get("jac_theta", None)
        if prev is None or not np.array_equal(prev, theta):
            self._rebuild_cache(state, theta)

        jac_all = ms["jac_all"]

        # Always recompute FIM from scratch using all currently observed points.
        # Jacobians are nonlinear in theta, so any cached partial sum is invalid
        # once theta changes; recomputing ensures correctness at the current theta.
        n_p = len(theta)
        F = _build_fim(jac_all, state.observed_indices, n_p, self.reg)

        try:
            F_inv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            F_inv = np.linalg.pinv(F)

        J_cand = jac_all[aff_cand]
        K, D, P = J_cand.shape
        if D == 1:
            J2d = J_cand[:, 0, :]
            JFJt = np.sum((J2d @ F_inv) * J2d, axis=1)
            deltas = np.log1p(np.maximum(JFJt, 0.0))
        else:
            I_D = np.eye(D)
            deltas = np.empty(K)
            for i in range(K):
                M = J_cand[i] @ F_inv @ J_cand[i].T
                sign, logdet = np.linalg.slogdet(I_D + M)
                deltas[i] = logdet if sign > 0 else 0.0

        scores = deltas / np.maximum(aff_costs, 1e-30)
        best = int(np.argmax(scores))
        return np.array([aff_cand[best]])


# ── EnsembleDOptMethod (3-phase, from claude.py) ─────────────────────────────

class EnsembleDOptMethod:
    """Three-phase ensemble D-optimal selection.

    Phase 1  (step < phase1_steps): gradient-volume init with QR projections
    Phase 2  (step < phase2_steps): ensemble D-opt + 2-step look-ahead
    Phase 3  (step >= phase2_steps): ensemble D-opt, smaller ensemble, no look-ahead
    """

    def __init__(
        self,
        sigma2: float = 0.01,
        reg: float = 1e-6,
        jac_eps: float = 1e-5,
        phase1_steps: int = 10,
        phase2_steps: int = 50,
        k_init: int = 20,
        k_ens: int = 5,
        k_ens_late: int = 3,
        n_lookahead: int = 5,
    ):
        self.sigma2 = sigma2
        self.inv_s2 = 1.0 / sigma2
        self.reg = reg
        self.jac_eps = jac_eps
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.k_init = k_init
        self.k_ens = k_ens
        self.k_ens_late = k_ens_late
        self.n_lookahead = n_lookahead

    # ── ensemble sampling from N(θ̂, FIM⁻¹) ──────────────────────────────

    def _sample_ensemble(self, theta, fim, rng, k, param_bounds=None):
        n_p = len(theta)
        try:
            fim_inv = np.linalg.inv(fim)
            fim_inv = (fim_inv + fim_inv.T) / 2.0
            eigvals = np.linalg.eigvalsh(fim_inv)
            if eigvals.min() < 0:
                fim_inv -= (eigvals.min() - 1e-10) * np.eye(n_p)
            samples = rng.multivariate_normal(theta, fim_inv, size=k)
        except np.linalg.LinAlgError:
            samples = np.array(
                [theta + rng.normal(0, 0.01, n_p) for _ in range(k)]
            )
        if param_bounds is not None:
            lb = np.array([b[0] for b in param_bounds])
            ub = np.array([b[1] for b in param_bounds])
            samples = np.clip(samples, lb, ub)
        return list(samples)

    # ── ensemble D-opt scores (avg information-gain / cost) ──────────────

    def _ensemble_scores(self, state, thetas_ens, aff_cand, aff_costs):
        n_rem = len(aff_cand)
        n_p = state.n_params
        avg = np.zeros(n_rem)
        n_valid = 0

        for th_k in thetas_ens:
            jac_k = _jac_batch(state.model_fn, th_k, state.X_train, self.jac_eps)
            F_k = _build_fim(jac_k, state.observed_indices, n_p,
                             self.reg, self.inv_s2)
            try:
                F_inv = np.linalg.inv(F_k)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(F_inv)):
                continue

            J_rem = jac_k[aff_cand]
            K, D, P = J_rem.shape
            if D == 1:
                G = J_rem[:, 0, :]
                quads = np.einsum("ij,ij->i", G @ F_inv, G)
            else:
                quads = np.array(
                    [np.trace(J_rem[i] @ F_inv @ J_rem[i].T) for i in range(K)]
                )
            gains = np.log1p(self.inv_s2 * np.clip(quads, 0, None))
            gains = np.where(np.isfinite(gains), gains, 0.0)
            avg += gains
            n_valid += 1

        if n_valid > 0:
            avg /= n_valid
        return avg / np.maximum(aff_costs, 1e-30)

    # ── 2-step look-ahead via Woodbury ───────────────────────────────────

    def _lookahead(self, state, thetas_ens, aff_cand, aff_costs, top_pos):
        best_combined = np.zeros(len(top_pos))
        n_p = state.n_params
        n_valid = 0

        for th_k in thetas_ens:
            jac_k = _jac_batch(state.model_fn, th_k, state.X_train, self.jac_eps)
            F_k = _build_fim(jac_k, state.observed_indices, n_p,
                             self.reg, self.inv_s2)
            try:
                F_inv = np.linalg.inv(F_k)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(F_inv)):
                continue

            J_rem = jac_k[aff_cand]
            K_rem, D, P = J_rem.shape
            n_valid += 1

            for t, cpos in enumerate(top_pos):
                J1 = J_rem[cpos]                   # (D, P)
                c1 = aff_costs[cpos]

                if D == 1:
                    g1 = J1[0]
                    Fg1 = F_inv @ g1
                    q1 = max(float(g1 @ Fg1), 0.0)
                    d1 = np.log1p(self.inv_s2 * q1)
                    if not np.isfinite(d1):
                        continue
                    denom = self.sigma2 + q1
                    if denom < 1e-30:
                        continue
                    F_inv_new = F_inv - np.outer(Fg1, Fg1) / denom
                    if not np.all(np.isfinite(F_inv_new)):
                        continue
                    G2 = J_rem[:, 0, :]
                    q2 = np.einsum("ij,ij->i", G2 @ F_inv_new, G2)
                else:
                    Fg1 = F_inv @ J1.T                 # (P, D)
                    M = J1 @ Fg1                       # (D, D)
                    sign, logdet = np.linalg.slogdet(
                        np.eye(D) + self.inv_s2 * M)
                    d1 = logdet if sign > 0 else 0.0
                    if not np.isfinite(d1):
                        continue
                    S = np.eye(D) * self.sigma2 + M
                    try:
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        continue
                    F_inv_new = F_inv - Fg1 @ S_inv @ Fg1.T
                    if not np.all(np.isfinite(F_inv_new)):
                        continue
                    q2 = np.array(
                        [np.trace(J_rem[i] @ F_inv_new @ J_rem[i].T)
                         for i in range(K_rem)]
                    )

                d2 = np.log1p(self.inv_s2 * np.clip(q2, 0, None))
                d2 = np.where(np.isfinite(d2), d2, 0.0)
                combined = (d1 + d2) / (c1 + aff_costs)
                combined[cpos] = -np.inf
                best_combined[t] += np.max(combined)

        if n_valid > 0:
            best_combined /= n_valid
        return best_combined

    # ── Phase 1: gradient-volume / cost with QR projections ──────────────

    def _phase1_propose(self, state, aff_cand, aff_costs):
        ms = state.method_state
        n_p = state.n_params

        # First call: build K random thetas and cache their Jacobians
        if "p1_jacs" not in ms:
            thetas = [_random_theta(state) for _ in range(self.k_init)]
            ms["p1_jacs"] = [
                _jac_batch(state.model_fn, th, state.X_train, self.jac_eps)
                for th in thetas
            ]
            ms["p1_Q"] = [np.zeros((n_p, 0)) for _ in range(self.k_init)]

        jacs = ms["p1_jacs"]
        Q_list = ms["p1_Q"]

        avg_scores = np.zeros(len(aff_cand))
        for k in range(self.k_init):
            J_cand = jacs[k][aff_cand]             # (K, D, P)
            K_c, D, P = J_cand.shape
            Q_k = Q_list[k]                        # (P, q)

            # Project each output row, then Frobenius norm of residual
            G = J_cand.reshape(K_c * D, P)
            if Q_k.shape[1] > 0:
                residual = G - (G @ Q_k) @ Q_k.T
            else:
                residual = G
            residual = residual.reshape(K_c, D, P)
            norms = np.sqrt(np.sum(residual ** 2, axis=(1, 2)))
            avg_scores += norms

        avg_scores /= self.k_init
        scores = avg_scores / np.maximum(aff_costs, 1e-30)
        best = int(np.argmax(scores))
        selected = aff_cand[best]

        # Update Q basis for each ensemble member
        for k in range(self.k_init):
            J_sel = jacs[k][selected]              # (D, P)
            Q_k = Q_list[k]
            for d in range(J_sel.shape[0]):
                g = J_sel[d]
                res = g - Q_k @ (Q_k.T @ g) if Q_k.shape[1] > 0 else g.copy()
                norm = np.linalg.norm(res)
                if norm > 1e-12:
                    Q_k = np.column_stack([Q_k, res / norm])
            Q_list[k] = Q_k

        return np.array([selected])

    # ── propose (dispatcher) ─────────────────────────────────────────────

    def propose(self, state: SelectionState) -> np.ndarray:
        aff_cand, aff_costs = _affordable_candidates(state)
        if len(aff_cand) == 0:
            return np.array([], dtype=int)
        if state.model_fn is None or state.X_train is None:
            return _inverse_cost_fallback(state, aff_cand, aff_costs)

        ms = state.method_state
        step = ms.get("step", 0)
        ms["step"] = step + 1

        # Phase 1: gradient-volume initialisation
        if step < self.phase1_steps:
            return self._phase1_propose(state, aff_cand, aff_costs)

        # Need fitted theta for Phases 2 & 3
        theta = state.current_theta
        if theta is None:
            return _inverse_cost_fallback(state, aff_cand, aff_costs)

        n_p = state.n_params
        use_la = step < self.phase2_steps
        k = self.k_ens if use_la else self.k_ens_late

        # Build FIM at current theta and sample ensemble
        jac_main = _jac_batch(state.model_fn, theta, state.X_train, self.jac_eps)
        fim = _build_fim(jac_main, state.observed_indices, n_p,
                         self.reg, self.inv_s2)
        thetas_ens = self._sample_ensemble(theta, fim, state.rng, k, state.param_bounds)

        scores = self._ensemble_scores(state, thetas_ens, aff_cand, aff_costs)

        if use_la and len(aff_cand) > self.n_lookahead:
            top_pos = np.argsort(scores)[-self.n_lookahead:][::-1]
            la = self._lookahead(state, thetas_ens, aff_cand, aff_costs, top_pos)
            best_pos = top_pos[int(np.argmax(la))]
        else:
            best_pos = int(np.argmax(scores))

        return np.array([aff_cand[best_pos]])


# ── Registry ──────────────────────────────────────────────────────────────────

METHOD_REGISTRY = {
    "random": RandomMethod,
    "inverse_cost": InverseCostMethod,
    "fim": FIMMethod,
    "ensemble_dopt": EnsembleDOptMethod,
}
