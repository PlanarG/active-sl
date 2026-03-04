"""Pipeline runner: budget-controlled active learning loop."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from benchmark.metrics import r_squared, mse, auc_r2, budget_to_reach
from benchmark.method import SelectionState
from benchmark.task import ScalingLawTask

BUDGET_CHECKPOINTS = np.logspace(np.log10(0.0001), np.log10(1.0), 10).tolist()
ALPHA_LEVELS = np.logspace(np.log10(0.0001), np.log10(10.0), 10).tolist()


@dataclass
class RunResult:
    task_id: str
    seed: int
    r2_at_checkpoints: Dict[float, float]
    mse_at_checkpoints: Dict[float, float]
    auc: float
    oracle_r2: float
    oracle_mse: float
    budget_to_reach_values: Dict[float, float]


def _init_state(task: ScalingLawTask, seed: int) -> SelectionState:
    rng = np.random.default_rng(seed)
    n_train = task.X_train.shape[0]
    total_cost = float(np.sum(task.cost_train))
    return SelectionState(
        candidate_indices=np.arange(n_train),
        observed_indices=np.array([], dtype=int),
        current_theta=None,
        spent_budget=0.0,
        total_budget=total_cost,
        cost_per_point=task.cost_train,
        rng=rng,
        X_train=task.X_train,
        model_fn=task.model_fn,
        n_params=task.n_params,
        param_bounds=task.param_bounds,
    )


def _evaluate(task: ScalingLawTask, theta: np.ndarray) -> tuple:
    """Compute test R² and MSE. Returns (-1.0, inf) on numerical failure."""
    theta_2d = theta.reshape(1, -1)
    try:
        pred = task.model_fn(theta_2d, task.X_test)
        pred_arr = np.asarray(pred, dtype=np.float64)
        if not np.all(np.isfinite(pred_arr)):
            return -1.0, float("inf")
        r2_val = r_squared(task.y_test, pred)
        mse_val = mse(task.y_test, pred)
        if not np.isfinite(r2_val):
            r2_val = -1.0
        if not np.isfinite(mse_val):
            mse_val = float("inf")
        return r2_val, mse_val
    except Exception:
        return -1.0, float("inf")


def run_single(
    task: ScalingLawTask,
    method,
    fitter,
    seed: int,
) -> RunResult:
    state = _init_state(task, seed)
    total_cost = state.total_budget

    r2_at_checkpoints = {}
    mse_at_checkpoints = {}
    checkpoint_fracs = []
    checkpoint_r2_vals = []
    checkpoint_mse_vals = []

    for checkpoint in BUDGET_CHECKPOINTS:
        target = checkpoint * total_cost
        while state.spent_budget < target - 1e-12 and len(state.candidate_indices) > 0:
            selected = method.propose(state)
            if len(selected) == 0:
                break
            for idx in selected:
                state.spent_budget += state.cost_per_point[idx]
            state.observed_indices = np.concatenate([state.observed_indices, selected])
            mask = np.isin(state.candidate_indices, selected, invert=True)
            state.candidate_indices = state.candidate_indices[mask]

        if len(state.observed_indices) < task.n_params:
            r2_val = -1.0
            mse_val = float("inf")
        else:
            obs_idx = state.observed_indices.astype(int)
            X_obs = task.X_train[obs_idx]
            y_obs = task.y_train[obs_idx]
            theta = fitter.fit(
                task.model_fn, X_obs, y_obs, task.n_params,
                theta0=state.current_theta,
                bounds=task.param_bounds,
            )
            state.current_theta = theta
            r2_val, mse_val = _evaluate(task, theta)

        r2_at_checkpoints[checkpoint] = r2_val
        mse_at_checkpoints[checkpoint] = mse_val
        checkpoint_fracs.append(checkpoint)
        checkpoint_r2_vals.append(r2_val)
        checkpoint_mse_vals.append(mse_val)

    # Oracle: use the result at budget fraction 1.0 (last checkpoint)
    oracle_r2_val = checkpoint_r2_vals[-1]
    oracle_mse_val = checkpoint_mse_vals[-1]

    fracs = np.array(checkpoint_fracs)
    r2_vals = np.array(checkpoint_r2_vals)
    mse_vals = np.array(checkpoint_mse_vals)
    auc_val = auc_r2(fracs, r2_vals)

    btr = {}
    for alpha in ALPHA_LEVELS:
        btr[alpha] = budget_to_reach(fracs, mse_vals, oracle_mse_val, alpha)

    return RunResult(
        task_id=task.task_id,
        seed=seed,
        r2_at_checkpoints=r2_at_checkpoints,
        mse_at_checkpoints=mse_at_checkpoints,
        auc=auc_val,
        oracle_r2=oracle_r2_val,
        oracle_mse=oracle_mse_val,
        budget_to_reach_values=btr,
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
