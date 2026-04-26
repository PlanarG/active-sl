"""
Pure D-optimal baseline (single best-fit mode, no MoG).

Selects the next point by maximising the local Fisher information gain
of the current best-fit theta, cost-penalised.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
from multiprocessing import cpu_count, get_context
from scipy.optimize import minimize
import time

from benchmark.task import load_tasks_for_dataset

# ── Config ──────────────────────────────────────────────────────────────
DATASET   = sys.argv[1] if len(sys.argv) > 1 else "data_constrained_scaling_law"
N_REPEAT  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_STARTS  = 64
RNG_SEED  = 42
COST_EXP  = 0.4
DEFAULT_CHECKPOINTS = [0.01, 0.05, 0.1]
CHECKPOINTS = DEFAULT_CHECKPOINTS
N_WORKERS = min(16, cpu_count() - 1)
SIGMA2_A0 = 2.0

# ── Global state ────────────────────────────────────────────────────────
task = None
X_train = y_train = X_test = y_test = cost_train = None
n_params = 0
n_out = 1
bounds = lo = hi = None
total_cost = 0.0
SIGMA2_B0 = 0.0
prior_std = prior_precision = None


def setup_task(task_obj, group_idx=0):
    global task, X_train, y_train, X_test, y_test, cost_train
    global n_params, n_out, bounds, lo, hi, total_cost
    global SIGMA2_B0, prior_std, prior_precision

    task       = task_obj
    gd         = task.groups[group_idx]
    X_train, y_train   = gd.X_train, gd.y_train
    X_test,  y_test    = gd.X_test,  gd.y_test
    cost_train = gd.cost_train
    n_params   = task.n_params
    n_out      = y_train.shape[1] if y_train.ndim == 2 else 1
    bounds     = task.param_bounds
    lo         = np.array([b[0] for b in bounds])
    hi         = np.array([b[1] for b in bounds])
    total_cost = cost_train.sum()

    y_var    = np.var(y_train)
    SIGMA2_B0 = SIGMA2_A0 * y_var

    prior_std = np.zeros(n_params)
    for j in range(n_params):
        if lo[j] > 0:
            prior_std[j] = (np.log(hi[j]) - np.log(lo[j])) / 4.0
        else:
            prior_std[j] = (hi[j] - lo[j]) / 4.0
    prior_precision = 1.0 / prior_std ** 2


def model_fn(theta, X):
    return task.model_fn(theta, X)


rng = np.random.RandomState(RNG_SEED)


def random_theta0():
    t = np.empty(n_params)
    for j in range(n_params):
        if lo[j] > 0:
            t[j] = np.exp(rng.uniform(np.log(lo[j]), np.log(hi[j])))
        else:
            t[j] = rng.uniform(lo[j], hi[j])
    return t


def _fit_one_worker(args):
    t0, X_sel, y_sel = args
    n      = len(y_sel)
    _multi = y_sel.ndim == 2

    def obj(theta):
        p, J = model_fn(theta, X_sel)
        r = p - y_sel
        if _multi:
            r_flat = r.ravel()
            J_flat = J.reshape(-1, J.shape[-1])
            return float(np.sum(r_flat ** 2)) / n, (2.0 / n) * J_flat.T @ r_flat
        return r @ r / n, (2.0 / n) * J.T @ r

    res = minimize(obj, t0, method='L-BFGS-B', bounds=bounds,
                   jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
    return res.x, res.fun


perturb_rng = np.random.RandomState(12345)


def perturb_theta(theta, scale=0.03):
    t = theta.copy()
    for j in range(n_params):
        eps = perturb_rng.randn() * scale
        if lo[j] > 0:
            t[j] = theta[j] * np.exp(eps)
        else:
            t[j] = theta[j] + eps * prior_std[j]
    return np.clip(t, [b[0] for b in bounds], [b[1] for b in bounds])


def multistart_fit(X_sel, y_sel, n_starts, warm_starts=None, perturb_scale=0.0):
    inits = []
    if warm_starts is not None:
        max_warm = max(n_starts - 16, n_starts // 2)
        for t in warm_starts[:max_warm]:
            inits.append(np.array(t).copy())
    n_warm_actual = len(inits)
    while len(inits) < n_starts:
        inits.append(random_theta0())

    if perturb_scale > 0 and warm_starts is not None and len(warm_starts) > 0:
        n_replace = (len(inits) - n_warm_actual) // 2
        for i in range(n_replace):
            base = warm_starts[i % len(warm_starts)]
            inits[n_warm_actual + i] = perturb_theta(base, scale=perturb_scale)

    args_list = [(t0, X_sel, y_sel) for t0 in inits]
    if N_WORKERS > 1 and len(args_list) >= 2:
        ctx = get_context('fork')
        with ctx.Pool(N_WORKERS) as pool:
            results = pool.map(_fit_one_worker, args_list)
    else:
        results = [_fit_one_worker(a) for a in args_list]

    out_t = np.array([r[0] for r in results])
    out_m = np.array([r[1] for r in results])
    return out_t, out_m


def bayesian_sigma2(mse, n_obs):
    ssr = n_obs * mse
    return (SIGMA2_B0 + ssr / 2.0) / (SIGMA2_A0 + n_obs / 2.0 - 1.0)


def get_jac_scale(theta):
    s = np.ones(n_params)
    for j in range(n_params):
        if lo[j] > 0:
            s[j] = theta[j]
    return s


def _scale_and_flatten_jac(J, jac_scale):
    J_trans = J * jac_scale[None, :]
    if J_trans.ndim == 3:
        return J_trans, J_trans.reshape(-1, J_trans.shape[-1])
    return J_trans, J_trans


def safe_symmetric_inv(H):
    p = H.shape[0]
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
        lam_max   = eigvals[-1]
        floor     = max(1e-10 * lam_max, 1e-30)
        eigvals_s = np.maximum(eigvals, floor)
        return (eigvecs / eigvals_s[None, :]) @ eigvecs.T
    except np.linalg.LinAlgError:
        pass
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    s_s = np.maximum(s, max(1e-6 * s[0], 1e-15))
    return (Vt.T / s_s[None, :]) @ U.T


# ── Fisher covariance for best-fit theta ─────────────────────────────
def compute_cov(theta, X_sel, y_sel, sigma2):
    _, J_sel    = model_fn(theta, X_sel)
    jac_scale   = get_jac_scale(theta)
    _, J_flat   = _scale_and_flatten_jac(J_sel, jac_scale)
    H           = J_flat.T @ J_flat / sigma2 + np.diag(prior_precision)
    H_inv       = safe_symmetric_inv(H)
    return H_inv, jac_scale


# ── D-optimal acquisition ─────────────────────────────────────────────
def dopt_scores(theta, H_inv, jac_scale, X_cand, sigma2):
    """
    ΔH = log det(I + J_c Σ J_c^T / σ²)

    For D=1 this reduces to log(1 + j_c^T Σ j_c / σ²).
    """
    D  = n_out
    Mc = X_cand.shape[0]

    # Candidate Jacobians
    _, J_c      = model_fn(theta, X_cand)
    J_c_s, _    = _scale_and_flatten_jac(J_c, jac_scale)   # (Mc,D,P) or (Mc,P)

    if D == 1:
        AJ_c  = J_c_s @ H_inv                               # (Mc, P)
        a_k   = np.sum(AJ_c * J_c_s, axis=1)               # (Mc,)
        return np.log1p(np.maximum(a_k, 0.0) / max(sigma2, 1e-300))
    else:
        FJ_c  = np.einsum('mdp,pq->mdq', J_c_s, H_inv)    # (Mc, D, P)
        M_c   = np.matmul(FJ_c, J_c_s.transpose(0, 2, 1)) # (Mc, D, D)
        A_c   = np.eye(D)[None] + M_c / max(sigma2, 1e-300)
        signs, logdets = np.linalg.slogdet(A_c)
        return np.where(signs > 0, logdets, -np.inf)


# ── R² helpers ────────────────────────────────────────────────────────
def compute_r2(theta, X_eval, y_eval):
    p, _   = model_fn(theta, X_eval)
    p_flat = np.asarray(p, dtype=np.float64).ravel()
    y_flat = np.asarray(y_eval, dtype=np.float64).ravel()
    ss_res = np.sum((y_flat - p_flat) ** 2)
    ss_tot = np.sum((y_flat - y_flat.mean()) ** 2)
    return float(np.clip(1 - ss_res / max(ss_tot, 1e-300), -1, 1))


def cheap_init(n_init, budget_limit):
    sorted_idx = np.argsort(cost_train)
    selected   = [sorted_idx[0]]
    cost_spent = cost_train[sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if len(selected) >= n_init:
            break
        if cost_spent + cost_train[idx] <= budget_limit:
            selected.append(idx)
            cost_spent += cost_train[idx]
    return selected


# ── Main AL loop ────────────────────────────────────────────────────────
def run_one(seed, verbose=True):
    global rng
    rng = np.random.RandomState(seed)

    budget = max(CHECKPOINTS) * total_cost

    if verbose:
        print(f"Budget={budget:.4e} ({max(CHECKPOINTS)*100:.0f}% of {total_cost:.4e}), D={n_out}")

    init_budget = 0.02 * total_cost
    n_init      = int(2.5 * n_params // n_out)
    selected    = cheap_init(n_init, init_budget)
    cost_spent  = cost_train[selected].sum()
    available   = set(range(len(X_train))) - set(selected)

    if verbose:
        print(f"Init: {len(selected)} pts, cost={cost_spent:.4e} "
              f"(frac={cost_spent/total_cost:.6f})")

    X_sel = X_train[selected]
    y_sel = y_train[selected]

    thetas, mses = multistart_fit(X_sel, y_sel, N_STARTS)
    best_theta   = thetas[np.argmin(mses)].copy()
    sigma2       = bayesian_sigma2(mses.min(), len(selected))

    if verbose:
        print(f"  Fit: best_MSE={mses.min():.4e}, sig2={sigma2:.4e}")
        print(f"Init R2: {compute_r2(best_theta, X_test, y_test):.4f}\n")

    warm_pool  = [best_theta.copy()]
    cp_thetas  = {}
    prev_theta = best_theta.copy()
    frac       = cost_spent / total_cost
    cp_remaining = sorted(CHECKPOINTS)
    while cp_remaining and frac >= cp_remaining[0]:
        cp_thetas[cp_remaining.pop(0)] = best_theta.copy()

    H_inv, jac_scale = compute_cov(best_theta, X_sel, y_sel, sigma2)

    iteration = 0
    while cost_spent < budget:
        iteration += 1
        t0_t = time.time()

        affordable = np.array([i for i in available
                               if cost_train[i] + cost_spent <= budget])
        if len(affordable) == 0:
            if verbose:
                print("No affordable candidates.")
            break

        X_cand = X_train[affordable]
        dgain  = dopt_scores(best_theta, H_inv, jac_scale, X_cand, sigma2)

        cost_factor = (cost_train[affordable] / total_cost) ** COST_EXP
        scores      = dgain / np.maximum(cost_factor, 1e-300)

        best_local = np.argmax(scores)
        best_idx   = affordable[best_local]

        selected.append(best_idx)
        available.discard(best_idx)
        cost_spent += cost_train[best_idx]

        X_sel = X_train[selected]
        y_sel = y_train[selected]

        thetas, mses = multistart_fit(X_sel, y_sel, N_STARTS,
                                      warm_starts=list(warm_pool))
        best_theta   = thetas[np.argmin(mses)].copy()

        warm_pool.append(best_theta)
        if len(warm_pool) > 30:
            warm_pool = warm_pool[-30:]

        sigma2           = bayesian_sigma2(mses.min(), len(selected))
        H_inv, jac_scale = compute_cov(best_theta, X_sel, y_sel, sigma2)

        dt   = time.time() - t0_t
        frac = cost_spent / total_cost

        if verbose:
            r2 = compute_r2(best_theta, X_test, y_test)
            print(f"  {iteration:3d}| idx={best_idx:3d} c={cost_train[best_idx]:.1e} "
                  f"dD={dgain[best_local]:.4f} sig2={sigma2:.4e} "
                  f"-> R2={r2:.4f} f={frac:.5f} ({dt:.1f}s)")

        while cp_remaining and frac >= cp_remaining[0]:
            cp_thetas[cp_remaining.pop(0)] = prev_theta.copy()
        prev_theta = best_theta.copy()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Final: {len(selected)} pts, frac={cost_spent/total_cost:.5f}")

    for cp in cp_remaining:
        cp_thetas[cp] = prev_theta.copy()

    return cp_thetas


if __name__ == '__main__':
    import json
    from datetime import datetime

    all_tasks = load_tasks_for_dataset(DATASET)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir   = os.path.join('output', DATASET, f"baseline_dopt_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'result.json')

    print(f"=== baseline_dopt === [{DATASET}]")
    print(f"SLs: {[t.sl_id for t in all_tasks]}, repeat={N_REPEAT}")
    print(f"N_STARTS={N_STARTS}, WORKERS={N_WORKERS}, COST_EXP={COST_EXP}")
    print(f"Output: {result_path}\n")

    def evaluate_global(task_obj, group_thetas):
        all_pred, all_true = [], []
        for gi, gd in enumerate(task_obj.groups):
            theta = group_thetas.get(gi)
            if theta is None:
                continue
            setup_task(task_obj, gi)
            try:
                pred, _ = model_fn(theta, gd.X_test)
                pred     = np.asarray(pred, dtype=np.float64)
                if not np.all(np.isfinite(pred)):
                    continue
                all_pred.append(pred.ravel())
                all_true.append(np.asarray(gd.y_test, dtype=np.float64).ravel())
            except Exception:
                continue
        if not all_pred:
            return -1.0
        p = np.concatenate(all_pred)
        y = np.concatenate(all_true)
        ss_res = np.sum((y - p) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(np.clip(1 - ss_res / max(ss_tot, 1e-300), -1, 1))

    results = {}

    for task_obj in all_tasks:
        sl_id       = task_obj.sl_id
        CHECKPOINTS = task_obj.budget_checkpoints or DEFAULT_CHECKPOINTS
        n_groups    = len(task_obj.groups)
        setup_task(task_obj, 0)

        print(f"{'='*60}")
        print(f"[{sl_id}] groups={n_groups}, train={len(X_train)}, test={len(X_test)}, "
              f"params={n_params}, D={n_out}, checkpoints={CHECKPOINTS}")

        all_cp_r2 = {cp: [] for cp in CHECKPOINTS}

        for rep in range(N_REPEAT):
            seed    = RNG_SEED + rep
            verbose = (rep == 0)
            group_cp_thetas = {cp: {} for cp in CHECKPOINTS}

            for gi in range(n_groups):
                setup_task(task_obj, gi)
                if verbose:
                    label = task_obj.groups[gi].group if n_groups > 1 else str(seed)
                    print(f"  --- repeat {rep}, group {label} ---")
                cp_t = run_one(seed, verbose=verbose)
                for cp in CHECKPOINTS:
                    group_cp_thetas[cp][gi] = cp_t[cp]

            for cp in CHECKPOINTS:
                all_cp_r2[cp].append(evaluate_global(task_obj, group_cp_thetas[cp]))

            vals = " | ".join(f"{cp}:{all_cp_r2[cp][-1]:.4f}" for cp in CHECKPOINTS)
            print(f"  repeat {rep} (seed={seed}): {vals}")

        results[sl_id] = {
            str(cp): {
                "mean":   float(np.mean(all_cp_r2[cp])),
                "std":    float(np.std(all_cp_r2[cp])),
                "values": [float(v) for v in all_cp_r2[cp]],
            }
            for cp in CHECKPOINTS
        }

        print(f"\n  [{sl_id}] summary:")
        for cp in CHECKPOINTS:
            m, s = np.mean(all_cp_r2[cp]), np.std(all_cp_r2[cp])
            print(f"    {cp*100:5.1f}%: R2={m:.4f} +/- {s:.4f}")

        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  -> updated {result_path}\n")

    print(f"\n{'='*60}")
    print(f"All done. Results in {result_path}")
