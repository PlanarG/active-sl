"""Pipeline runner: budget-controlled active learning loop."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from benchmark.metrics import r_squared, log_auc_r2
from benchmark.method import SelectionState
from benchmark.task import ScalingLawTask, GroupData

BUDGET_CHECKPOINTS = np.logspace(np.log10(0.0001), np.log10(1.0), 10).tolist()


@dataclass
class RunResult:
    task_id: str
    seed: int
    r2_at_checkpoints: Dict[float, float]
    log_auc: float


def _init_state(gd: GroupData, task: ScalingLawTask, seed: int) -> SelectionState:
    rng = np.random.default_rng(seed)
    n_train = gd.X_train.shape[0]
    total_cost = float(np.sum(gd.cost_train))
    return SelectionState(
        candidate_indices=np.arange(n_train),
        observed_indices=np.array([], dtype=int),
        current_theta=None,
        spent_budget=0.0,
        total_budget=total_cost,
        cost_per_point=gd.cost_train,
        rng=rng,
        X_train=gd.X_train,
        model_fn=task.model_fn,
        n_params=task.n_params,
        param_bounds=task.param_bounds,
    )


def _evaluate_global(task: ScalingLawTask, group_thetas: Dict[str, np.ndarray]) -> float:
    """Compute global R² by concatenating predictions across all groups.

    Groups without a valid theta contribute predictions of NaN and are excluded.
    Returns -1.0 on failure.
    """
    all_pred = []
    all_true = []
    for gd in task.groups:
        theta = group_thetas.get(gd.group)
        if theta is None:
            continue
        theta_2d = theta.reshape(1, -1)
        try:
            pred = task.model_fn(theta_2d, gd.X_test)
            pred_arr = np.asarray(pred, dtype=np.float64)
            if not np.all(np.isfinite(pred_arr)):
                continue
            all_pred.append(pred_arr.ravel())
            all_true.append(np.asarray(gd.y_test, dtype=np.float64).ravel())
        except Exception:
            continue

    if len(all_pred) == 0:
        return -1.0

    pred_cat = np.concatenate(all_pred)
    true_cat = np.concatenate(all_true)
    r2 = r_squared(true_cat, pred_cat)
    return r2 if np.isfinite(r2) else -1.0


def run_single(
    task: ScalingLawTask,
    method,
    fitter,
    seed: int,
) -> RunResult:
    # Use different seed per group to avoid correlated randomness
    base_seed = seed * 100_000
    states = {}
    for i, gd in enumerate(task.groups):
        states[gd.group] = _init_state(gd, task, base_seed + i)

    r2_at_checkpoints = {}
    checkpoint_r2_vals = []

    for checkpoint in BUDGET_CHECKPOINTS:
        group_thetas = {}

        for gd in task.groups:
            state = states[gd.group]
            target = checkpoint * state.total_budget

            # Selection phase
            while state.spent_budget < target - 1e-12 and len(state.candidate_indices) > 0:
                selected = method.propose(state)
                if len(selected) == 0:
                    break
                for idx in selected:
                    state.spent_budget += state.cost_per_point[idx]
                state.observed_indices = np.concatenate([state.observed_indices, selected])
                mask = np.isin(state.candidate_indices, selected, invert=True)
                state.candidate_indices = state.candidate_indices[mask]

            # Fitting phase
            obs_idx = state.observed_indices.astype(int)
            if len(obs_idx) < task.n_params:
                continue

            X_obs = gd.X_train[obs_idx]
            y_obs = gd.y_train[obs_idx]
            n_random = max(fitter.n_restarts - 1, 1)
            theta0s = []
            if state.current_theta is not None:
                theta0s.append(state.current_theta.copy())
            bounds = task.param_bounds
            lo = np.array([b[0] for b in bounds], dtype=np.float64)
            hi = np.array([b[1] for b in bounds], dtype=np.float64)
            for _ in range(n_random):
                theta0s.append(lo + (hi - lo) * state.rng.random(task.n_params))
            theta = fitter.fit(
                task.model_fn, X_obs, y_obs, task.n_params,
                bounds=task.param_bounds, theta0s=theta0s,
            )
            state.current_theta = theta
            group_thetas[gd.group] = theta

        # Global evaluation across all groups
        r2_val = _evaluate_global(task, group_thetas)
        r2_at_checkpoints[checkpoint] = r2_val
        checkpoint_r2_vals.append(r2_val)

    r2_vals = np.array(checkpoint_r2_vals)
    log_auc_val = log_auc_r2(r2_vals)

    return RunResult(
        task_id=task.task_id,
        seed=seed,
        r2_at_checkpoints=r2_at_checkpoints,
        log_auc=log_auc_val,
    )


def run_repeat(
    task: ScalingLawTask,
    method,
    fitter,
    seeds: List[int],
    max_workers: int = 1,
) -> List[RunResult]:
    """Run the same task across multiple seeds, optionally in parallel."""
    if max_workers <= 1:
        return [run_single(task, method, fitter, s) for s in seeds]

    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = [None] * len(seeds)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(run_single, task, method, fitter, s): i
            for i, s in enumerate(seeds)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results
