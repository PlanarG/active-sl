"""
Baseline: fit on the entire training set with multistart L-BFGS-B.
Reports test R² (global across groups) per scaling law, written to output JSON.
"""
import sys, os
import numpy as np
from multiprocessing import cpu_count, get_context
from scipy.optimize import minimize
import time

from benchmark.task import load_tasks_for_dataset

# ── Config ──────────────────────────────────────────────────────────────
DATASET = sys.argv[1] if len(sys.argv) > 1 else "data_constrained_scaling_law"
N_REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_STARTS = 64
RNG_SEED = 42
N_WORKERS = min(16, cpu_count() - 1)
DEFAULT_CHECKPOINTS = [0.01, 0.05, 0.1]

# ── Global state (set by setup_task) ────────────────────────────────────
task = None
X_train = y_train = X_test = y_test = None
n_params = 0
bounds = lo = hi = None


def setup_task(task_obj, group_idx=0):
    global task, X_train, y_train, X_test, y_test
    global n_params, bounds, lo, hi

    task = task_obj
    gd = task.groups[group_idx]
    X_train, y_train = gd.X_train, gd.y_train
    X_test, y_test = gd.X_test, gd.y_test
    n_params = task.n_params
    bounds = task.param_bounds
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])


def model_fn(theta, X):
    return task.model_fn(theta, X)


# ── Fitting ─────────────────────────────────────────────────────────────
def _fit_one_worker(args):
    t0, X_sel, y_sel, model_fn_worker, bounds_worker = args
    n = len(y_sel)

    multi_output = y_sel.ndim == 2

    def obj(theta):
        p, J = model_fn_worker(theta, X_sel)
        r = p - y_sel
        if multi_output:
            r_flat = r.ravel()
            J_flat = J.reshape(-1, J.shape[-1])
            return float(np.sum(r_flat ** 2)) / n, (2.0 / n) * J_flat.T @ r_flat
        return r @ r / n, (2.0 / n) * J.T @ r

    res = minimize(obj, t0, method='L-BFGS-B', bounds=bounds_worker,
                   jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
    return res.x, res.fun


def multistart_fit(X_sel, y_sel, n_starts, rng):
    inits = []
    for _ in range(n_starts):
        t = np.empty(n_params)
        for j in range(n_params):
            if lo[j] > 0:
                t[j] = np.exp(rng.uniform(np.log(lo[j]), np.log(hi[j])))
            else:
                t[j] = rng.uniform(lo[j], hi[j])
        inits.append(t)

    args_list = [(t0, X_sel, y_sel, task.model_fn, bounds) for t0 in inits]
    if N_WORKERS > 1 and len(args_list) >= 2:
        ctx = get_context('spawn')
        with ctx.Pool(N_WORKERS) as pool:
            results = pool.map(_fit_one_worker, args_list)
    else:
        results = [_fit_one_worker(a) for a in args_list]

    out_t = np.array([r[0] for r in results])
    out_m = np.array([r[1] for r in results])
    return out_t[np.argmin(out_m)]


# ── Evaluation ──────────────────────────────────────────────────────────
def evaluate_global(task_obj, group_thetas):
    """Compute global R² by concatenating predictions across all groups."""
    all_pred, all_true = [], []
    for gi, gd in enumerate(task_obj.groups):
        theta = group_thetas.get(gi)
        if theta is None:
            continue
        setup_task(task_obj, gi)
        try:
            pred, _ = model_fn(theta, gd.X_test)
            pred = np.asarray(pred, dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                continue
            all_pred.append(pred.ravel())
            all_true.append(np.asarray(gd.y_test, dtype=np.float64).ravel())
        except Exception:
            continue
    if len(all_pred) == 0:
        return -1.0
    pred_cat = np.concatenate(all_pred)
    true_cat = np.concatenate(all_true)
    ss_res = np.sum((true_cat - pred_cat) ** 2)
    ss_tot = np.sum((true_cat - true_cat.mean()) ** 2)
    if ss_tot < 1e-300:
        return -1.0
    return float(np.clip(1 - ss_res / ss_tot, -1, 1))


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import json
    from datetime import datetime

    all_tasks = load_tasks_for_dataset(DATASET)
    # all_tasks = [t for t in all_tasks if t.sl_id in ["sl_6"]]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('output', DATASET, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'result.json')

    print(f"=== baseline_full_train === [{DATASET}]")
    print(f"SLs: {[t.sl_id for t in all_tasks]}, repeat={N_REPEAT}")
    print(f"N_STARTS={N_STARTS}, WORKERS={N_WORKERS}")
    print(f"Output: {result_path}\n")

    results = {}

    for task_obj in all_tasks:
        sl_id = task_obj.sl_id
        checkpoints = task_obj.budget_checkpoints or DEFAULT_CHECKPOINTS
        n_groups = len(task_obj.groups)
        setup_task(task_obj, 0)

        print(f"{'='*60}")
        print(f"[{sl_id}] groups={n_groups}, train={len(X_train)}, test={len(X_test)}, params={n_params}")

        all_r2 = []

        for rep in range(N_REPEAT):
            seed = RNG_SEED + rep
            t0 = time.time()

            group_thetas = {}
            for gi in range(n_groups):
                setup_task(task_obj, gi)
                rng = np.random.RandomState(seed)
                best_theta = multistart_fit(X_train, y_train, N_STARTS, rng)
                group_thetas[gi] = best_theta

            r2 = evaluate_global(task_obj, group_thetas)
            all_r2.append(r2)
            dt = time.time() - t0
            print(f"  repeat {rep} (seed={seed}): R²={r2:.4f} ({dt:.1f}s)")

        # All checkpoints get the same R² (full training set)
        results[sl_id] = {
            str(cp): {
                "mean": float(np.mean(all_r2)),
                "std": float(np.std(all_r2)),
                "values": [float(v) for v in all_r2],
            }
            for cp in checkpoints
        }

        print(f"\n  [{sl_id}] summary: R²={np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")

        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  -> updated {result_path}\n")

    print(f"\n{'='*60}")
    print(f"All done. Results in {result_path}")
