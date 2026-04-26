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
    y_train: Optional[np.ndarray] = None


# ── Shared helpers ────────────────────────────────────────────────────────────

def _affordable_candidates(state: SelectionState):
    """Return (affordable_candidates, affordable_costs) arrays."""
    remaining = state.total_budget - state.spent_budget
    costs = state.cost_per_point[state.candidate_indices]
    mask = costs <= remaining + 1e-12
    return state.candidate_indices[mask], costs[mask]


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


# ── GreedyCheapestMethod ─────────────────────────────────────────────────────

class GreedyCheapestMethod:
    """Always select the cheapest affordable candidate (ties broken randomly)."""

    def propose(self, state: SelectionState) -> np.ndarray:
        aff_cand, aff_costs = _affordable_candidates(state)
        if len(aff_cand) == 0:
            return np.array([], dtype=int)
        min_cost = aff_costs.min()
        ties = np.where(np.abs(aff_costs - min_cost) < 1e-12)[0]
        idx = state.rng.choice(ties)
        return np.array([aff_cand[idx]])



# ── Registry ──────────────────────────────────────────────────────────────────

METHOD_REGISTRY = {
    "random": RandomMethod,
    "inverse_cost": InverseCostMethod,
    "greedy_cheapest": GreedyCheapestMethod,
}
