"""Scaling laws for domain mixture proportions (multi-output).

X columns: [proportion_domain_1..5]
Output: [loss_domain_1..5]
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12
_NUM_DOMAINS = 5


def _squeeze(pred, B):
    return pred[0] if B == 1 else pred


# sl_1 (30p): loss_i = a_i + b_i*log(p_i+eps) + sum_{j!=i} c_{ij}*p_j
# Per domain: a_i(1) + b_i(1) + c_{ij}(4) = 6 -> 30 total
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    if backend == "torch":
        out = xp.zeros((B, M, _NUM_DOMAINS), dtype=xp.float64)
    else:
        out = xp.zeros((B, M, _NUM_DOMAINS))
    offset = 0
    for i in range(_NUM_DOMAINS):
        a_i = theta[:, offset]
        b_i = theta[:, offset + 1]
        c_ij = theta[:, offset + 2: offset + 6]
        offset += 6
        p_i = ops.clamp_min(X[:, i], _EPS)
        val = a_i[:, None] + b_i[:, None] * xp.log(p_i)[None, :]
        j_indices = [j for j in range(_NUM_DOMAINS) if j != i]
        for k, j in enumerate(j_indices):
            val = val + c_ij[:, k:k+1] * X[None, :, j]
        if backend == "torch":
            out[:, :, i] = val
        elif backend == "jax":
            out = out.at[:, :, i].set(val)
        else:
            out[:, :, i] = val
    return _squeeze(out, B)


# sl_2 (35p): loss_i = A_i*(p_i+eps_i)^(-alpha_i)*exp(sum_j w_{ij}*p_j)
# Per domain: A(1)+eps(1)+alpha(1)+w(4 cross)=7 -> 35 total
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    if backend == "torch":
        out = xp.zeros((B, M, _NUM_DOMAINS), dtype=xp.float64)
    else:
        out = xp.zeros((B, M, _NUM_DOMAINS))
    offset = 0
    for i in range(_NUM_DOMAINS):
        A_i = theta[:, offset]
        eps_i = theta[:, offset + 1]
        alpha_i = theta[:, offset + 2]
        w_ij = theta[:, offset + 3: offset + 7]
        offset += 7
        p_i = ops.clamp_min(X[:, i] + eps_i[:, None], _EPS)
        power_term = A_i[:, None] * (p_i ** (-alpha_i[:, None]))
        j_indices = [j for j in range(_NUM_DOMAINS) if j != i]
        interaction = xp.zeros((B, M)) if backend != "torch" else xp.zeros((B, M), dtype=xp.float64)
        for k, j in enumerate(j_indices):
            interaction = interaction + w_ij[:, k:k+1] * X[None, :, j]
        interaction = ops.clamp(interaction, min=-20.0, max=20.0)
        val = power_term * ops.exp(interaction)
        if backend == "torch":
            out[:, :, i] = val
        elif backend == "jax":
            out = out.at[:, :, i].set(val)
        else:
            out[:, :, i] = val
    return _squeeze(out, B)


# sl_3 (35p): loss_i = base_i + coeff_i*p_i^exp_i + sum_{j!=i} W_{ij}*p_j
# Power law self + full linear cross (5 base + 5 coeff + 5 exp + 20 cross = 35)
# Repack: per domain i: base(1)+coeff(1)+exp(1)+W_{ij}(4)=7 -> 35
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    if backend == "torch":
        import torch; out = torch.zeros((B, M, _NUM_DOMAINS), dtype=torch.float64)
    else:
        import numpy as np; out = np.zeros((B, M, _NUM_DOMAINS))
    offset = 0
    for i in range(_NUM_DOMAINS):
        base_i = theta[:, offset]
        coeff_i = theta[:, offset + 1]
        exp_i = theta[:, offset + 2]
        W_ij = theta[:, offset + 3: offset + 7]
        offset += 7
        p_i = ops.clamp_min(X[:, i], _EPS)
        val = base_i[:, None] + coeff_i[:, None] * (p_i[None, :] ** exp_i[:, None])
        j_indices = [j for j in range(_NUM_DOMAINS) if j != i]
        for k, j in enumerate(j_indices):
            val = val + W_ij[:, k:k+1] * X[None, :, j]
        out[:, :, i] = val
    if backend == "jax":
        import jax.numpy as jnp
        out = jnp.array(out)
    return _squeeze(out, B)


# sl_4 (35p): loss_i = exp(sum_k C_{ik}*p_k^alpha_k + bias_i)
# Exponential of linear combo of power-transformed props
# 5 bias + 25 C + 5 alpha = 35
# Pack: 5 alpha first, then per domain: bias(1)+C(5)=6 -> 5+30=35
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    # First 5 params: shared alpha exponents
    alphas = theta[:, :5]  # (B, 5)
    if backend == "torch":
        out = xp.zeros((B, M, _NUM_DOMAINS), dtype=xp.float64)
    else:
        out = xp.zeros((B, M, _NUM_DOMAINS))
    # Power-transform: X_powered[None,:,k] = X[:,k]^alphas[:,k]
    offset = 5
    for i in range(_NUM_DOMAINS):
        bias_i = theta[:, offset]
        C_ik = theta[:, offset + 1: offset + 6]  # (B, 5)
        offset += 6
        # sum_k C_ik * p_k^alpha_k
        lin = bias_i[:, None]  # (B, 1) -> broadcast to (B, M)
        for k in range(_NUM_DOMAINS):
            p_k = ops.clamp_min(X[:, k], _EPS)
            lin = lin + C_ik[:, k:k+1] * (p_k[None, :] ** alphas[:, k:k+1])
        lin = ops.clamp(lin, min=-50.0, max=50.0)
        val = ops.exp(lin)
        if backend == "torch":
            out[:, :, i] = val
        elif backend == "jax":
            out = out.at[:, :, i].set(val)
        else:
            out[:, :, i] = val
    return _squeeze(out, B)


# sl_5 (35p): loss_i = b_i + sum_j W_{ij} * p_j^alpha_j
# Full weight matrix on power-transformed proportions (shared alpha)
# 5 alpha + 5 bias + 25 W = 35
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    alphas = theta[:, :5]  # (B, 5)
    if backend == "torch":
        import torch; out = torch.zeros((B, M, _NUM_DOMAINS), dtype=torch.float64)
    else:
        import numpy as np; out = np.zeros((B, M, _NUM_DOMAINS))
    offset = 5
    for i in range(_NUM_DOMAINS):
        b_i = theta[:, offset]
        W_ij = theta[:, offset + 1: offset + 6]  # (B, 5)
        offset += 6
        val = b_i[:, None]
        for j in range(_NUM_DOMAINS):
            p_j = ops.clamp_min(X[:, j], _EPS)
            val = val + W_ij[:, j:j+1] * (p_j[None, :] ** alphas[:, j:j+1])
        out[:, :, i] = val
    if backend == "jax":
        import jax.numpy as jnp; out = jnp.array(out)
    return _squeeze(out, B)


# sl_6 (35p): loss_i = C_i + A_i * (sum_j T_{ij}*p_j)^(-alpha_i)
# Effective-mixture power law
# Per domain: C(1)+A(1)+alpha(1)+T(5)=8... but 8*5=40. Use T(4 cross-domain only)=7 -> 35
# Actually: per domain: C(1)+A(1)+alpha(1)+T(4 cross)=7 -> 35
# Use all 5 T weights: C(1)+A(1)+alpha(1)+T(4)=7 -> 35
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    if backend == "torch":
        import torch; out = torch.zeros((B, M, _NUM_DOMAINS), dtype=torch.float64)
    else:
        import numpy as np; out = np.zeros((B, M, _NUM_DOMAINS))
    offset = 0
    for i in range(_NUM_DOMAINS):
        C_i = theta[:, offset]
        A_i = theta[:, offset + 1]
        alpha_i = theta[:, offset + 2]
        # T weights for all 5 domains (4 cross + self = use all proportions)
        T_ij = theta[:, offset + 3: offset + 7]  # 4 weights for j!=i
        offset += 7
        # Compute effective mixture: T_{i,self}*p_i + sum_{j!=i} T_{ij}*p_j
        # We use self proportion directly + 4 cross weights
        eff = X[None, :, i]  # self proportion, unweighted
        j_indices = [j for j in range(_NUM_DOMAINS) if j != i]
        for k, j in enumerate(j_indices):
            eff = eff + T_ij[:, k:k+1] * X[None, :, j]
        eff = ops.clamp_min(eff, _EPS)
        val = C_i[:, None] + A_i[:, None] * (eff ** (-alpha_i[:, None]))
        out[:, :, i] = val
    if backend == "jax":
        import jax.numpy as jnp; out = jnp.array(out)
    return _squeeze(out, B)


# sl_7 (40p): loss_i = intercept_i + sum_j (c_lin_{ij}*p_j + c_log_{ij}*log(p_j+eps))
# Full linear-and-log matrix
# Per domain: intercept(1) + c_lin(5) + c_log(5) - but c_log_{ii} merged, keep separate
# -> 1+5+5 = 11 per domain? Too many. Use 8 per domain: intercept(1)+c_lin(4 cross)+c_log_self(1)+c_log(2 cross)=8 -> 40
# Simpler: per domain 8p: intercept(1) + linear_self(1) + log_self(1) + cross_linear(4) + cross_log(1 shared) = 8 -> 40
# Actually let's do it properly: per domain i: intercept(1) + 4 cross_lin + 1 self_lin + 1 self_log + 1 self_quad_log = 8 -> 40
# Simplest: per domain 8p total -> 40. Use: a_i + b_i*p_i + c_i*log(p_i) + sum_{j!=i}(d_{ij}*p_j + e_i*log(p_j))
# Per domain: a(1)+b(1)+c(1)+d(4)+e(1)=8 -> 40
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    if backend == "torch":
        out = xp.zeros((B, M, _NUM_DOMAINS), dtype=xp.float64)
    else:
        out = xp.zeros((B, M, _NUM_DOMAINS))
    offset = 0
    for i in range(_NUM_DOMAINS):
        a_i = theta[:, offset]
        b_i = theta[:, offset + 1]
        c_i = theta[:, offset + 2]
        d_ij = theta[:, offset + 3: offset + 7]  # 4 cross-domain linear
        e_i = theta[:, offset + 7]  # shared cross-domain log coeff
        offset += 8
        p_i = ops.clamp_min(X[:, i], _EPS)
        val = a_i[:, None] + b_i[:, None] * X[None, :, i] + c_i[:, None] * xp.log(p_i)[None, :]
        j_indices = [j for j in range(_NUM_DOMAINS) if j != i]
        for k, j in enumerate(j_indices):
            p_j = ops.clamp_min(X[:, j], _EPS)
            val = val + d_ij[:, k:k+1] * X[None, :, j] + e_i[:, None] * xp.log(p_j)[None, :]
        if backend == "torch":
            out[:, :, i] = val
        elif backend == "jax":
            out = out.at[:, :, i].set(val)
        else:
            out[:, :, i] = val
    return _squeeze(out, B)


# sl_8 (15p): loss_i = c_i - a_i * p_i^b_i
# Single-domain power law (depends only on own proportion)
# Per domain: 3p -> 15
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    if backend == "torch":
        import torch; out = torch.zeros((B, M, _NUM_DOMAINS), dtype=torch.float64)
    else:
        import numpy as np; out = np.zeros((B, M, _NUM_DOMAINS))
    offset = 0
    for i in range(_NUM_DOMAINS):
        c_i = theta[:, offset]
        a_i = theta[:, offset + 1]
        b_i = theta[:, offset + 2]
        offset += 3
        p_i = ops.clamp_min(X[:, i], _EPS)
        val = c_i[:, None] - a_i[:, None] * (p_i[None, :] ** b_i[:, None])
        out[:, :, i] = val
    if backend == "jax":
        import jax.numpy as jnp; out = jnp.array(out)
    return _squeeze(out, B)


# sl_9 (15p): loss_i = a_i + b_i*log(p_i+eps) + c_i*[log(p_i+eps)]^2
# Quadratic-in-log
# Per domain: 3p -> 15
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    if backend == "torch":
        out = xp.zeros((B, M, _NUM_DOMAINS), dtype=xp.float64)
    else:
        out = xp.zeros((B, M, _NUM_DOMAINS))
    offset = 0
    for i in range(_NUM_DOMAINS):
        a_i = theta[:, offset]
        b_i = theta[:, offset + 1]
        c_i = theta[:, offset + 2]
        offset += 3
        p_i = ops.clamp_min(X[:, i], _EPS)
        lp = xp.log(p_i)[None, :]
        val = a_i[:, None] + b_i[:, None] * lp + c_i[:, None] * lp ** 2
        if backend == "torch":
            out[:, :, i] = val
        elif backend == "jax":
            out = out.at[:, :, i].set(val)
        else:
            out[:, :, i] = val
    return _squeeze(out, B)


# sl_10 (15p): loss_i = a_i + b_i / (p_i + eps_i)
# Reciprocal law
# Per domain: 3p -> 15
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    B, M = theta.shape[0], X.shape[0]
    if backend == "torch":
        import torch; out = torch.zeros((B, M, _NUM_DOMAINS), dtype=torch.float64)
    else:
        import numpy as np; out = np.zeros((B, M, _NUM_DOMAINS))
    offset = 0
    for i in range(_NUM_DOMAINS):
        a_i = theta[:, offset]
        b_i = theta[:, offset + 1]
        eps_i = theta[:, offset + 2]
        offset += 3
        denom = ops.clamp_min(X[:, i] + eps_i[:, None], _EPS)
        val = a_i[:, None] + b_i[:, None] / denom[None, :] if B > 1 else a_i[:, None] + b_i[:, None] / denom[None, :]
        # Fix: eps_i is (B,), X[:,i] is (M,) -> need proper broadcasting
        val = a_i[:, None] + b_i[:, None] / ops.clamp_min(X[None, :, i] + eps_i[:, None], _EPS)
        out[:, :, i] = val
    if backend == "jax":
        import jax.numpy as jnp; out = jnp.array(out)
    return _squeeze(out, B)


PARAM_BOUNDS = {
    # Dataset: p_i (proportions) in [0, 0.75], min nonzero ~0.03125,
    #          log(p_i) in [-3.47, -0.29], loss_i in [1.21, 3.97] (per domain).
    #
    # Bound derivation:
    #   - Loss-floor constants (a_i, base_i, C_i, c_i in sl_8): (-3, 6) —
    #     total loss is 1.21–3.97, so floor < 4.
    #   - Log coefficients (b_i, c_i in sl_1/7/9): loss / |log(p_min)| ~ 3/3.47 ~ 0.9,
    #     so |coeff| ≲ 3–5; use (-10, 5) or (-5, 5) with margin.
    #   - Linear cross-domain weights (c_ij, W_ij, d_ij): p_j ≤ 0.75, contribution
    #     ≲ 3 → |weight| ≲ 4; use (-10, 10) with generous margin.
    #   - Power-law exponents (exp_i, alpha_i, b_i in sl_8): typically 0–2 for physical
    #     decay; allow (-2, 4) for exploration.
    #   - Interaction / mixing weights (w_ij in sl_2, T_ij in sl_6): code already
    #     clamps exp() to [-20,20]/[-50,50], so overflow is impossible; use (-5, 5).
    #   - sl_2 A_i (scale): (0, 10) — (p+eps)^{-alpha} ~ 1–30, A * 30 ~ 3 → A ≲ 10.
    #   - sl_4/5 shared alphas: (-1, 3) — fitted values 0.97–1.98.
    #   - sl_8 c_i (maximum loss), a_i (decay scale), b_i (exponent): all positive,
    #     fitted a_i ~ 0.23–0.84, b_i ~ 0.23–0.34, c_i ~ 1.96–3.59.
    #   - sl_10 b_i: very small (~0.01–0.06 in fit); use (-1, 1).
    #   - sl_10 eps_i: small shift; use (-0.03, 0.3).
    #
    # No overflow: all expressions bounded by construction or by code-level clamps.

    # sl_1: 30p = 5 × [a_i, b_i, c_i1..c_i4]
    # loss_i = a_i + b_i*log(p_i) + sum_{j≠i} c_ij*p_j
    # Optimal: a~0.8–3.3, b~-0.02–0.003 (near zero), c~-0.5–1.6
    # b_i·log(p_i): log(p) in [-3.47, -0.29]; for |b|·3.47 ≲ 3 → |b| ≲ 1. Use (-10, 5).
    "sl_1": [(-3, 6), (-10, 5), (-10, 10), (-10, 10), (-10, 10), (-10, 10)] * 5,

    # sl_2: 35p = 5 × [A_i, eps_i, alpha_i, w_i1..w_i4]
    # loss_i = A_i*(p_i+eps_i)^{-alpha_i} * exp(sum_j w_ij*p_j)  [exp clamped ±20]
    # Optimal: A~2.2–6.2, eps~0.027–0.091, alpha~0.05–0.46, w~-1.5–0.1
    "sl_2": [(0, 10), (-0.03, 0.2), (0, 2), (-5, 5), (-5, 5), (-5, 5), (-5, 5)] * 5,

    # sl_3: 35p = 5 × [base_i, coeff_i, exp_i, W_i1..W_i4]
    # loss_i = base_i + coeff_i*p_i^exp_i + sum_{j≠i} W_ij*p_j
    # Optimal: base~0–4.4, coeff~-11–7.2, exp~1.2–2.6, W~-1.6–3.8
    # At exp_i=-2, p_min^{-2}=1024 → coeff·1024 ≲ 3 → coeff ≲ 0.003; tight with (-20,20).
    "sl_3": [(-3, 6), (-20, 20), (-2, 4), (-10, 10), (-10, 10), (-10, 10), (-10, 10)] * 5,

    # sl_4: 35p = 5 shared alphas + 5 × [bias_i, C_i1..C_i5]
    # loss_i = exp(bias_i + sum_k C_ik*p_k^alpha_k)  [lin clamped to ±50]
    # log(loss) in [0.19, 1.37]; so lin ~ 0.2–1.4; bias + sum ~ 0.2–1.4.
    # Optimal: alphas~0.97–1.93, bias~-0.85–1.45, C~-3.95–4.08
    "sl_4": [(-1, 3)] * 5 + [(-3, 3), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)] * 5,

    # sl_5: 35p = 5 shared alphas + 5 × [b_i, W_i1..W_i5]
    # loss_i = b_i + sum_j W_ij*p_j^alpha_j
    # Optimal: alphas~1.04–1.98, b~1.1–3.5, W~-15.5–7.5
    # At alpha=2, p_min^2=0.001 → W·0.001 ≲ 3 → W ≲ 3000 (but optimal max |W|~15.5).
    # Use (-25, 25) to contain observed -15.5 with margin.
    "sl_5": [(-1, 3)] * 5 + [(-3, 6), (-25, 25), (-25, 25), (-25, 25), (-25, 25), (-25, 25)] * 5,

    # sl_6: 35p = 5 × [C_i, A_i, alpha_i, T_i1..T_i4]
    # loss_i = C_i + A_i*(p_i + sum_{j≠i} T_ij*p_j)^{-alpha_i}  [eff clamped ≥ EPS]
    # Optimal: C~1.6–3.4, A~0–0.044 (near zero), alpha~0.22–1.83, T~-3.1–4.5
    # A_i very small in fit (sl_6 mostly reduces to constant C); allow (0, 10).
    "sl_6": [(-3, 6), (0, 10), (0, 3), (-5, 5), (-5, 5), (-5, 5), (-5, 5)] * 5,

    # sl_7: 40p = 5 × [a_i, b_i, c_i, d_i1..d_i4, e_i]
    # loss_i = a_i + b_i*p_i + c_i*log(p_i) + sum_{j≠i}(d_ij*p_j + e_i*log(p_j))
    # Optimal: a~-2.8–2.8, b~-0.9–6.7, c~-0.02–0.01, d~-3.6–6.6, e~-0.007–0.008
    # c_i and e_i are near-zero (log terms contribute little); use (-5, 5).
    "sl_7": [(-5, 8), (-10, 15), (-5, 5), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-5, 5)] * 5,

    # sl_8: 15p = 5 × [c_i, a_i, b_i]
    # loss_i = c_i - a_i*p_i^{b_i}  (physical: a_i>0, b_i>0, c_i = max loss at p_i→0)
    # Optimal: c~1.96–3.59, a~0.23–0.84, b~0.23–0.34
    # At p_i=0.03125, b_i=1: a·0.03125 ≲ 2 → a ≲ 64; use (0, 5) as tight bound.
    "sl_8": [(0, 6), (0, 5), (0, 3)] * 5,

    # sl_9: 15p = 5 × [a_i, b_i, c_i]
    # loss_i = a_i + b_i*log(p_i) + c_i*[log(p_i)]^2
    # Optimal: a~1.17–3.25, b~-0.15 to -0.04, c~-0.004 to -0.001 (near zero)
    # [log(p)]^2 in [0.08, 12]; c·12 ≲ 0.05 → negligible; use (-1, 1) for c_i.
    "sl_9": [(-3, 6), (-2, 1), (-1, 1)] * 5,

    # sl_10: 15p = 5 × [a_i, b_i, eps_i]
    # loss_i = a_i + b_i / (p_i + eps_i)  [denom clamped ≥ EPS]
    # Optimal: a~1.28–3.27, b~0.009–0.057, eps~0.014–0.100
    # b_i very small: b/(p+eps) ~ 0.05/(0.25+0.05) ~ 0.17; use (-1, 1).
    "sl_10": [(-3, 6), (-1, 1), (-0.03, 0.3)] * 5,
}

LAW_REGISTRY = {
    "sl_1": sl_1, "sl_2": sl_2, "sl_3": sl_3, "sl_4": sl_4, "sl_5": sl_5,
    "sl_6": sl_6, "sl_7": sl_7, "sl_8": sl_8, "sl_9": sl_9, "sl_10": sl_10,
}
PARAM_COUNTS = {
    "sl_1": 30, "sl_2": 35, "sl_3": 35, "sl_4": 35, "sl_5": 35,
    "sl_6": 35, "sl_7": 40, "sl_8": 15, "sl_9": 15, "sl_10": 15,
}
