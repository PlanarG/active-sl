"""
MSPE-principled acquisition function.

Theory:
  MSPE = Σ_k w_k tr(J_t Σ_k J_t^T)  [intra]
       + Σ_k w_k ||f̂_k - f̄||²       [inter]

  After observing y at candidate x:
    ΔV_intra = Σ_k w_k · j_x^T Σ_k F_test Σ_k j_x / s_k²   (= V-optimal, unchanged)
    ΔV_inter = V_inter_before - E_y[V_inter_after]             (1D quadrature)

  where E_y[V_inter_after] uses the pairwise form with polynomial
  coefficients precomputed from the gain vectors g_k = J_t Σ_k j_x / s_k².
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
from multiprocessing import cpu_count, get_context
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import time

from benchmark.task import load_tasks_for_dataset

# ── Config ──────────────────────────────────────────────────────────────
DATASET = sys.argv[1] if len(sys.argv) > 1 else "data_constrained_scaling_law"
N_REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_STARTS = 64
RNG_SEED = 42
N_QUAD = 500
COST_EXP = 0.4
DEFAULT_CHECKPOINTS = [0.01, 0.05, 0.1]
CHECKPOINTS = DEFAULT_CHECKPOINTS
N_WORKERS = min(16, cpu_count() - 1)
SIGMA2_A0 = 2.0

# ── Global state (set by setup_task) ────────────────────────────────────
task = None
X_train = y_train = X_test = y_test = cost_train = None
n_params = 0
n_out = 1
bounds = lo = hi = None
total_cost = 0.0
Y_MIN = Y_MAX = 0.0
SIGMA2_B0 = 0.0
prior_std = prior_precision = None


def setup_task(task_obj, group_idx=0):
    global task, X_train, y_train, X_test, y_test, cost_train
    global n_params, n_out, bounds, lo, hi, total_cost
    global Y_MIN, Y_MAX, SIGMA2_B0, prior_std, prior_precision

    task = task_obj
    gd = task.groups[group_idx]
    X_train, y_train = gd.X_train, gd.y_train
    X_test, y_test = gd.X_test, gd.y_test
    cost_train = gd.cost_train
    n_params = task.n_params
    n_out = y_train.shape[1] if y_train.ndim == 2 else 1
    bounds = task.param_bounds
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    total_cost = cost_train.sum()

    Y_MIN = max(0.0, float(y_train.min()) - 1.0)
    Y_MAX = float(y_train.max()) + 1.0

    y_var = np.var(y_train)
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
    n = len(y_sel)
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
    t = np.clip(t, [b[0] for b in bounds], [b[1] for b in bounds])
    return t


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
        n_random = len(inits) - n_warm_actual
        n_replace = n_random // 2
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


# ── Helpers: flatten Jacobian ─────────────────────────────────────────
def _scale_and_flatten_jac(J, jac_scale):
    J_trans = J * jac_scale[None, :]
    if J_trans.ndim == 3:
        return J_trans, J_trans.reshape(-1, J_trans.shape[-1])
    return J_trans, J_trans


def safe_symmetric_inv(H, name="H"):
    p = H.shape[0]
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
        lam_max = eigvals[-1]
        floor = max(1e-10 * lam_max, 1e-30)
        eigvals_safe = np.maximum(eigvals, floor)
        H_inv = (eigvecs / eigvals_safe[None, :]) @ eigvecs.T
        log_det = np.sum(np.log(eigvals_safe))
        cond = lam_max / eigvals_safe[0]
        return H_inv, log_det, cond
    except np.linalg.LinAlgError:
        pass
    try:
        H_reg = 0.5 * (H + H.T)
        diag_mean = max(np.mean(np.abs(np.diag(H_reg))), 1e-10)
        H_reg += 1e-6 * diag_mean * np.eye(p)
        eigvals, eigvecs = np.linalg.eigh(H_reg)
        lam_max = eigvals[-1]
        floor = max(1e-8 * lam_max, 1e-20)
        eigvals_safe = np.maximum(eigvals, floor)
        H_inv = (eigvecs / eigvals_safe[None, :]) @ eigvecs.T
        log_det = np.sum(np.log(eigvals_safe))
        cond = lam_max / eigvals_safe[0]
        return H_inv, log_det, cond
    except np.linalg.LinAlgError:
        pass
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    s_max = s[0]
    floor = max(1e-6 * s_max, 1e-15)
    s_safe = np.maximum(s, floor)
    H_inv = (Vt.T / s_safe[None, :]) @ U.T
    log_det = np.sum(np.log(s_safe))
    cond = s_max / s_safe[-1]
    return H_inv, log_det, cond


# ── Fisher covariance ──────────────────────────────────────────────────
def compute_cov_info(theta, X_sel, y_sel, sigma2):
    pred, J_sel = model_fn(theta, X_sel)
    jac_scale = get_jac_scale(theta)
    _, J_flat = _scale_and_flatten_jac(J_sel, jac_scale)
    H = J_flat.T @ J_flat / sigma2 + np.diag(prior_precision)
    H_inv, log_det_H, cond = safe_symmetric_inv(H)
    mse = float(np.mean((np.asarray(pred) - np.asarray(y_sel)) ** 2))
    return H_inv, jac_scale, log_det_H, cond, mse


# ── Predictive stats (flattened for clustering) ──────────────────────
def pred_stats_one(H_inv, jac_scale, theta, X_pts, sigma2):
    pred, J = model_fn(theta, X_pts)
    _, J_flat = _scale_and_flatten_jac(J, jac_scale)
    AJ = J_flat @ H_inv
    s2 = np.sum(AJ * J_flat, axis=1)
    mu = np.asarray(pred).ravel()
    return mu, np.maximum(s2, 0.0)


# ── SKL and silhouette ─────────────────────────────────────────────────
def skl_matrix(mus, vars_):
    K = mus.shape[0]
    SKL = np.zeros((K, K))
    for m in range(K):
        for n in range(m + 1, K):
            vm, vn = vars_[m], vars_[n]
            dm = mus[m] - mus[n]
            skl = 0.25 * (vm / vn + vn / vm - 2 + dm ** 2 * (1 / vm + 1 / vn))
            SKL[m, n] = SKL[n, m] = np.mean(skl)
    return SKL


def silhouette_precomputed(dist_mat, labels):
    n = len(labels)
    unique = sorted(set(labels))
    if len(unique) < 2:
        return -1.0
    la = np.array(labels)
    sil = np.zeros(n)
    for i in range(n):
        ci = la[i]
        same = (la == ci)
        same[i] = False
        if same.sum() == 0:
            continue
        a_i = dist_mat[i, same].mean()
        b_i = np.inf
        for c in unique:
            if c == ci:
                continue
            other = la == c
            if other.sum() > 0:
                b_i = min(b_i, dist_mat[i, other].mean())
        sil[i] = (b_i - a_i) / max(a_i, b_i, 1e-300)
    return np.mean(sil)


# ── Clustering + mode construction ─────────────────────────────────────
def build_modes(thetas, mses, X_sel, y_sel, X_eval, sigma2, verbose=True):
    K = len(thetas)
    n_obs = len(X_sel)
    bic_temp = max(1.0, n_obs / (2.0 * n_params))

    all_covs = []
    for k in range(K):
        all_covs.append(compute_cov_info(thetas[k], X_sel, y_sel, sigma2))

    mus = None
    s2s = None
    for k in range(K):
        H_inv_k, js_k = all_covs[k][0], all_covs[k][1]
        mu_k, s2_k = pred_stats_one(H_inv_k, js_k, thetas[k], X_eval, sigma2)
        if mus is None:
            mus = np.zeros((K, len(mu_k)))
            s2s = np.zeros((K, len(mu_k)))
        mus[k] = mu_k
        s2s[k] = s2_k
    
    if K <= 2:
        bic = np.array([n_obs * np.log(max(mses[k], 1e-300)) + n_params * np.log(n_obs)
                        for k in range(K)])
        log_ws = -bic / (2.0 * bic_temp)
        log_ws -= log_ws.max()
        ws = np.exp(log_ws)
        ws /= ws.sum()
        cov_infos = [(all_covs[k][0], all_covs[k][1]) for k in range(K)]
        if verbose:
            print(f"  {K} modes (no clustering), T={bic_temp:.1f}")
        return thetas, ws, cov_infos

    total_vars = s2s + sigma2
    SKL = skl_matrix(mus, total_vars)
    SKL_clean = np.clip(SKL, 0, None)
    np.fill_diagonal(SKL_clean, 0)
    SKL_clean = (SKL_clean + SKL_clean.T) / 2

    dist_mat = np.log1p(SKL_clean)
    dist_condensed = squareform(dist_mat, checks=False)
    dist_condensed = np.nan_to_num(dist_condensed, nan=30, posinf=30, neginf=0)
    Z = linkage(dist_condensed, method='complete')

    heights = sorted(set(Z[:, 2]))
    best_score = -1.0
    best_labels = np.ones(K, dtype=int)
    for h in heights:
        labels = fcluster(Z, t=h + 1e-10, criterion='distance')
        nc = len(set(labels))
        if nc < 2 or nc >= K:
            continue
        score = silhouette_precomputed(dist_mat, labels)
        if score > best_score:
            best_score = score
            best_labels = labels.copy()

    if best_score < 0.01:
        best_labels = np.ones(K, dtype=int)

    best_nc = len(set(best_labels))
    cluster_ids = sorted(set(best_labels))
    reps = []
    for c in cluster_ids:
        members = np.where(best_labels == c)[0]
        reps.append(members[np.argmin(mses[members])])
    reps = np.array(reps)

    bic = np.array([n_obs * np.log(max(mses[reps[i]], 1e-300)) + n_params * np.log(n_obs)
                    for i in range(len(reps))])
    log_ws = -bic / (2.0 )
    log_ws -= log_ws.max()
    ws = np.exp(log_ws)
    ws /= ws.sum()

    keep = ws > 0.001
    if keep.sum() == 0:
        keep[np.argmax(ws)] = True

    keep_idx = np.where(keep)[0]
    rep_thetas = thetas[reps[keep_idx]]
    rep_ws = ws[keep_idx]
    rep_ws /= rep_ws.sum()
    rep_covs = [(all_covs[reps[i]][0], all_covs[reps[i]][1]) for i in keep_idx]

    if verbose:
        print(f"  {K}->{best_nc} clusters(sil={best_score:.3f})->{len(keep_idx)} modes, T={bic_temp:.1f}")
        for i in range(len(rep_thetas)):
            ri = reps[keep_idx[i]]
            print(f"    [{i}] w={rep_ws[i]:.4f} MSE={mses[ri]:.4e} cond={all_covs[ri][3]:.1e}")

    return rep_thetas, rep_ws, rep_covs


# ── MSPE-based acquisition (V-opt intra + MSPE inter) ─────────────────
def compute_selection_score(rep_thetas, weights, cov_infos, X_cand, X_target, sigma2):
    """
    Compute ΔE[MSPE] = ΔV_intra + ΔV_inter for each candidate.

    ΔV_intra: V-optimal (prediction variance reduction at test points).
    ΔV_inter: expected reduction in inter-mode prediction disagreement,
              computed via 1D quadrature over the scalar observation y.

    Both terms are in MSPE units (y²), so 1:1 combination is principled.
    """
    K = len(rep_thetas)
    Mc = X_cand.shape[0]
    D = n_out

    dv_intra = np.zeros(Mc)
    dv_inter = np.zeros(Mc)

    if D > 1:
        # Multi-output: use D=1 V-opt path per flattened dimension,
        # inter-mode term omitted (would need D-dim quadrature).
        for k in range(K):
            H_inv_k, jac_scale_k = cov_infos[k]
            w_k = weights[k]

            pred_c, J_c = model_fn(rep_thetas[k], X_cand)
            J_c_scaled, _ = _scale_and_flatten_jac(J_c, jac_scale_k)
            J_flat = J_c_scaled.reshape(-1, n_params)
            AJ = J_flat @ H_inv_k
            a_k_flat = np.sum(AJ * J_flat, axis=1)  # (Mc*D,)

            I_D = np.eye(D)
            FJ = np.einsum('mdp,pq->mdq', J_c_scaled, H_inv_k)
            M_all = np.matmul(FJ, J_c_scaled.transpose(0, 2, 1))

            _, J_t = model_fn(rep_thetas[k], X_target)
            J_t_scaled, _ = _scale_and_flatten_jac(J_t, jac_scale_k)
            J_t_flat = J_t_scaled.reshape(-1, n_params)
            A_t = J_t_flat @ H_inv_k

            cross_all = np.einsum('ip,jdp->jid', A_t, J_c_scaled)
            CTC_all = np.einsum('jid,jie->jde', cross_all, cross_all)

            S_all = sigma2 * I_D[None] + M_all
            try:
                S_all_inv = np.linalg.inv(S_all)
            except np.linalg.LinAlgError:
                S_all_inv = np.zeros_like(S_all)
                for j in range(Mc):
                    try:
                        S_all_inv[j] = np.linalg.inv(S_all[j])
                    except np.linalg.LinAlgError:
                        pass

            dv_k = np.einsum('jde,jed->j', S_all_inv, CTC_all)
            dv_intra += w_k * dv_k

        return dv_intra, dv_inter

    # ── D == 1: full MSPE decomposition ──────────────────────────────

    # Per-mode quantities
    mus_cand = np.zeros((K, Mc))        # m_k(c): prediction at candidate
    s2_cand = np.zeros((K, Mc))         # σ² + h_k(c): predictive variance
    mus_test = []                        # f̂_k: prediction at test points
    cross_mats = []                      # J_t Σ_k J_c^T: (Mt, Mc) per mode

    for k in range(K):
        H_inv_k, jac_scale_k = cov_infos[k]
        w_k = weights[k]

        # Candidate: predictions, leverage, Jacobian
        pred_c, J_c = model_fn(rep_thetas[k], X_cand)
        J_c_scaled = J_c * jac_scale_k[None, :]
        AJ_c = J_c_scaled @ H_inv_k                     # (Mc, P)
        a_k = np.sum(AJ_c * J_c_scaled, axis=1)         # (Mc,) = h_k

        mus_cand[k] = pred_c
        s2_cand[k] = sigma2 + a_k

        # Test: predictions, Jacobian, cross term
        pred_t, J_t = model_fn(rep_thetas[k], X_target)
        J_t_scaled = J_t * jac_scale_k[None, :]
        mus_test.append(np.asarray(pred_t))               # (Mt,)

        cross = J_t_scaled @ (H_inv_k @ J_c_scaled.T)    # (Mt, Mc)
        cross_mats.append(cross)

        # ΔV_intra: V-optimal
        dv_k = (cross ** 2).sum(axis=0) / s2_cand[k]     # (Mc,)
        dv_intra += w_k * dv_k

    mus_test = np.array(mus_test)  # (K, Mt)

    # ── ΔV_inter: inter-mode MSPE reduction via 1D quadrature ────────
    if K > 1:
        # V_inter_before = Σ_{k<l} w_k w_l ||f̂_k - f̂_l||²
        V_inter_before = 0.0
        for k in range(K):
            for l in range(k + 1, K):
                delta = mus_test[k] - mus_test[l]
                V_inter_before += weights[k] * weights[l] * np.dot(delta, delta)

        # Precompute gain vectors: g_k = cross_k / s_k²  shape (Mt, Mc)
        gains = [cross_mats[k] / s2_cand[k][None, :] for k in range(K)]

        # Precompute polynomial coefficients for each pair (k, l)
        # ||f̂_k'(y) - f̂_l'(y)||² = A_kl + B_kl·y + C_kl·y²
        pair_data = []  # list of (w_kl, A, B, C) each shape (Mc,)
        for k in range(K):
            for l in range(k + 1, K):
                w_kl = weights[k] * weights[l]
                delta_kl = (mus_test[k] - mus_test[l])[:, None]  # (Mt, 1)

                # a_kl = δ_kl - g_k·m_k + g_l·m_l   (Mt, Mc)
                a_kl = (delta_kl
                        - gains[k] * mus_cand[k][None, :]
                        + gains[l] * mus_cand[l][None, :])
                # b_kl = g_k - g_l   (Mt, Mc)
                b_kl = gains[k] - gains[l]

                A = np.sum(a_kl ** 2, axis=0)       # (Mc,)
                B = 2.0 * np.sum(a_kl * b_kl, axis=0)  # (Mc,)
                C = np.sum(b_kl ** 2, axis=0)        # (Mc,)
                pair_data.append((w_kl, A, B, C))

        # Quadrature grid (global, covers all candidates)
        sig_cand = np.sqrt(s2_cand)  # (K, Mc)
        y_lo = float((mus_cand - 4.0 * sig_cand).min())
        y_hi = float((mus_cand + 4.0 * sig_cand).max())
        y_lo = max(y_lo, Y_MIN)
        y_hi = min(y_hi, Y_MAX)
        y_grid = np.linspace(y_lo, y_hi, N_QUAD)  # (Nq,)

        # Precompute φ_k(y_q, c) for all k, q, c: shape (K, Nq, Mc)
        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
        phi = np.zeros((K, N_QUAD, Mc))
        for k in range(K):
            diff = y_grid[:, None] - mus_cand[k][None, :]     # (Nq, Mc)
            phi[k] = inv_sqrt_2pi / sig_cand[k][None, :] * np.exp(
                -0.5 * diff ** 2 / s2_cand[k][None, :])

        # p(y, c) = Σ_k w_k φ_k
        p_mix = np.zeros((N_QUAD, Mc))
        for k in range(K):
            p_mix += weights[k] * phi[k]
        p_mix = np.maximum(p_mix, 1e-300)

        # E[V_inter'] = Σ_{k<l} w_kl ∫ (φ_k φ_l / p) · (A + By + Cy²) dy
        E_inter_after = np.zeros(Mc)
        pair_idx = 0
        for k in range(K):
            for l in range(k + 1, K):
                w_kl, A, B, C = pair_data[pair_idx]
                pair_idx += 1

                ratio = phi[k] * phi[l] / p_mix           # (Nq, Mc)
                poly = (A[None, :]
                        + B[None, :] * y_grid[:, None]
                        + C[None, :] * y_grid[:, None] ** 2)  # (Nq, Mc)

                integrand = ratio * poly                    # (Nq, Mc)
                integral = np.trapezoid(integrand, y_grid, axis=0)  # (Mc,)
                E_inter_after += w_kl * integral

        dv_inter = np.maximum(V_inter_before - E_inter_after, 0.0)

    return dv_intra, dv_inter


# ── R² functions ─────────────────────────────────────────────────────
def compute_r2(theta, X_eval, y_eval):
    p, _ = model_fn(theta, X_eval)
    p_flat = np.asarray(p, dtype=np.float64).ravel()
    y_flat = np.asarray(y_eval, dtype=np.float64).ravel()
    ss_res = np.sum((y_flat - p_flat) ** 2)
    ss_tot = np.sum((y_flat - y_flat.mean()) ** 2)
    return float(np.clip(1 - ss_res / max(ss_tot, 1e-300), -1, 1))


def compute_r2_ensemble(rep_thetas, weights, X_eval, y_eval):
    preds = None
    for k, theta in enumerate(rep_thetas):
        p, _ = model_fn(theta, X_eval)
        p_flat = np.asarray(p, dtype=np.float64).ravel()
        if preds is None:
            preds = np.zeros_like(p_flat)
        preds += weights[k] * p_flat
    y_flat = np.asarray(y_eval, dtype=np.float64).ravel()
    ss_res = np.sum((y_flat - preds) ** 2)
    ss_tot = np.sum((y_flat - y_flat.mean()) ** 2)
    return float(np.clip(1 - ss_res / max(ss_tot, 1e-300), -1, 1))


def cheap_init(n_init, budget_limit):
    sorted_idx = np.argsort(cost_train)
    selected = [sorted_idx[0]]
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

    init_budget = 0.020 * total_cost
    n_init = 2.5 * n_params // n_out
    selected = cheap_init(n_init, init_budget)
    cost_spent = cost_train[selected].sum()
    available = set(range(len(X_train))) - set(selected)

    if verbose:
        costs_sel = cost_train[selected]
        print(f"Init: {len(selected)} pts, cost={cost_spent:.4e} "
              f"(frac={cost_spent/total_cost:.6f}), "
              f"cost range [{costs_sel.min():.2e}, {costs_sel.max():.2e}]")

    X_sel = X_train[selected]
    y_sel = y_train[selected]

    thetas, mses = multistart_fit(X_sel, y_sel, N_STARTS)
    good_mask = mses < mses.min() * 1.5
    if good_mask.sum() < 2:
        good_mask = np.ones(len(mses), dtype=bool)
    thetas_good = thetas[good_mask]
    mses_good = mses[good_mask]

    sigma2 = bayesian_sigma2(mses.min(), len(selected))
    if verbose:
        print(f"  Fit: {good_mask.sum()}/{len(mses)} good, "
              f"best_MSE={mses.min():.4e}, sig2={sigma2:.4e}")

    rep_thetas, weights, cov_infos = build_modes(
        thetas_good, mses_good, X_sel, y_sel, X_test, sigma2, verbose=verbose)


    warm_pool = [thetas[np.argmin(mses)].copy()]

    best_theta = thetas[np.argmin(mses)].copy()
    if verbose:
        r2_test = compute_r2(best_theta, X_test, y_test)
        print(f"Init R2: {r2_test:.4f}, K={len(rep_thetas)}\n")

    cp_remaining = sorted(CHECKPOINTS)
    cp_thetas = {}
    prev_theta = best_theta.copy()
    frac = cost_spent / total_cost
    while cp_remaining and frac >= cp_remaining[0]:
        cp_thetas[cp_remaining.pop(0)] = best_theta.copy()

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
        dv_intra, dv_inter = compute_selection_score(
            rep_thetas, weights, cov_infos, X_cand, X_test, sigma2)

        combined = dv_intra + dv_inter
        cost_factor = (cost_train[affordable] / total_cost) ** COST_EXP
        scores = combined / np.maximum(cost_factor, 1e-300)

        best_local = np.argmax(scores)
        best_idx = affordable[best_local]

        selected.append(best_idx)
        available.discard(best_idx)
        cost_spent += cost_train[best_idx]

        X_sel = X_train[selected]
        y_sel = y_train[selected]

        warm = list(warm_pool)
        thetas, mses = multistart_fit(X_sel, y_sel, N_STARTS, warm_starts=warm,
                                       perturb_scale=0.0)

        cur_best = thetas[np.argmin(mses)].copy()
        if not any(np.allclose(cur_best, w) for w in warm_pool):
            warm_pool.append(cur_best)
        if len(warm_pool) > 30:
            warm_pool = warm_pool[-30:]

        good_mask = mses < (mses.min() + 1e-30) * 1.5
        thetas_good = thetas[good_mask]
        mses_good = mses[good_mask]

        sigma2 = bayesian_sigma2(mses.min(), len(selected))

        do_verbose = verbose and (iteration <= 5 or iteration % 10 == 0)
        rep_thetas, rep_ws, cov_infos = build_modes(
            thetas_good, mses_good, X_sel, y_sel, X_test, sigma2, verbose=do_verbose)
        
        weights = rep_ws

        best_theta = cur_best

        dt = time.time() - t0_t
        frac = cost_spent / total_cost

        if verbose:
            r2_test = compute_r2(best_theta, X_test, y_test)
            print(f"  {iteration:3d}| idx={best_idx:3d} c={cost_train[best_idx]:.1e} "
                  f"dV={dv_intra[best_local]:.4f} dI={dv_inter[best_local]:.4f} "
                  f"sig2={sigma2:.4e} -> R2={r2_test:.4f} "
                  f"K={len(rep_thetas)} f={frac:.5f} ({dt:.1f}s)")

        while cp_remaining and frac >= cp_remaining[0]:
            cp_thetas[cp_remaining.pop(0)] = prev_theta.copy()
        prev_theta = best_theta.copy()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Final: {len(selected)} pts, frac={cost_spent / total_cost:.5f}")

    for cp in cp_remaining:
        cp_thetas[cp] = prev_theta.copy()

    return cp_thetas


if __name__ == '__main__':
    import json
    from datetime import datetime

    all_tasks = load_tasks_for_dataset(DATASET)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('output', DATASET, f"v28_fuck_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'result.json')

    print(f"=== v28_mspe === [{DATASET}]")
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

    results = {}

    for task_obj in all_tasks:
        sl_id = task_obj.sl_id
        CHECKPOINTS = task_obj.budget_checkpoints or DEFAULT_CHECKPOINTS
        n_groups = len(task_obj.groups)
        setup_task(task_obj, 0)

        print(f"{'='*60}")
        print(f"[{sl_id}] groups={n_groups}, train={len(X_train)}, test={len(X_test)}, "
              f"params={n_params}, D={n_out}, checkpoints={CHECKPOINTS}")

        all_cp_r2 = {cp: [] for cp in CHECKPOINTS}

        for rep in range(N_REPEAT):
            seed = RNG_SEED + rep
            verbose = (rep == 0)

            group_cp_thetas = {cp: {} for cp in CHECKPOINTS}

            for gi in range(n_groups):
                setup_task(task_obj, gi)
                if verbose and n_groups > 1:
                    gname = task_obj.groups[gi].group
                    print(f"  --- repeat {rep}, group {gname} (seed={seed}) ---")
                elif verbose:
                    print(f"  --- repeat {rep} (seed={seed}, verbose) ---")
                cp_thetas = run_one(seed, verbose=verbose)
                for cp in CHECKPOINTS:
                    group_cp_thetas[cp][gi] = cp_thetas[cp]

            for cp in CHECKPOINTS:
                r2 = evaluate_global(task_obj, group_cp_thetas[cp])
                all_cp_r2[cp].append(r2)

            vals = " | ".join(f"{cp}:{all_cp_r2[cp][-1]:.4f}" for cp in CHECKPOINTS)
            print(f"  repeat {rep} (seed={seed}): {vals}")

        results[sl_id] = {
            str(cp): {
                "mean": float(np.mean(all_cp_r2[cp])),
                "std": float(np.std(all_cp_r2[cp])),
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
