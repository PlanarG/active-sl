"""Scaling laws for supervised fine-tuning data size.

X columns: [sft_data_size (N)]
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


# sl_1 (4p): L_inf + A * (N + N0)^(-alpha)
# theta: [L_inf, A, N0, alpha]
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = X[:, 0]
    L_inf, A, N0, alpha = [theta[:, i] for i in range(4)]
    base = ops.clamp_min(N[None, :] + N0[:, None], _EPS)  # (B, M)
    log_base = xp.log(ops.clamp_min(base, _EPS))  # (B, M)
    power = base ** (-alpha[:, None])  # (B, M)
    term = A[:, None] * power  # (B, M)
    pred = L_inf[:, None] + term  # (B, M)

    # Jacobian: (B, M, 4)
    ones = pred * 0.0 + 1.0
    d_L_inf = ones
    d_A = power                            # base^(-alpha)
    d_N0 = term * (-alpha[:, None]) / base  # A * (-alpha) * base^(-alpha-1) * 1
    d_alpha = -term * log_base             # A * base^(-alpha) * (-log(base))

    jac = ops.stack([d_L_inf, d_A, d_N0, d_alpha], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_2 (4p): L_inf + A / (1 + (N/N0)^alpha)  — Hill/Michaelis-Menten
# theta: [L_inf, A, N0, alpha]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, N0, alpha = [theta[:, i] for i in range(4)]
    N0_safe = ops.clamp_min(N0, _EPS)
    ratio = (N[None, :] / N0_safe[:, None]) ** alpha[:, None]  # (B, M)
    denom = 1.0 + ratio  # (B, M)
    frac = 1.0 / denom  # A's coefficient: 1/(1+ratio)
    pred = L_inf[:, None] + A[:, None] * frac  # (B, M)

    # Jacobian: (B, M, 4)
    # Let r = (N/N0)^alpha, f = 1/(1+r), pred = L_inf + A*f
    # d/d(L_inf) = 1
    # d/d(A)     = f
    # d(r)/d(N0) = alpha * (N/N0)^alpha * (-1/N0) = -alpha * r / N0
    # d(f)/d(r)  = -1/(1+r)^2 = -f^2
    # d/d(N0)    = A * (-f^2) * (-alpha * r / N0) = A * f^2 * alpha * r / N0
    # d(r)/d(alpha) = r * log(N/N0)
    # d/d(alpha) = A * (-f^2) * r * log(N/N0) = -A * f^2 * r * log(N/N0)
    log_ratio_base = xp.log(ops.clamp_min(N[None, :] / N0_safe[:, None], _EPS))  # (B, M)
    f2 = frac * frac  # f^2 = 1/(1+r)^2

    ones = pred * 0.0 + 1.0
    d_L_inf = ones
    d_A = frac
    d_N0 = A[:, None] * f2 * alpha[:, None] * ratio / N0_safe[:, None]
    d_alpha = -A[:, None] * f2 * ratio * log_ratio_base

    jac = ops.stack([d_L_inf, d_A, d_N0, d_alpha], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_3 (4p): L_inf + A / (N^alpha + B)  — denominator-offset power law
# theta: [L_inf, A, alpha, B]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, alpha, B = [theta[:, i] for i in range(4)]
    N_pow = N[None, :] ** alpha[:, None]  # (B, M)
    denom = N_pow + B[:, None]  # (B, M)
    denom = ops.clamp_min(denom, _EPS)
    frac = A[:, None] / denom  # (B, M)
    pred = L_inf[:, None] + frac  # (B, M)

    # Jacobian: (B, M, 4)
    # pred = L_inf + A / denom, where denom = N^alpha + B
    # d/d(L_inf) = 1
    # d/d(A) = 1/denom
    # d(denom)/d(alpha) = N^alpha * log(N)
    # d/d(alpha) = -A / denom^2 * N^alpha * log(N) = -frac / denom * N^alpha * log(N)
    # d/d(B) = -A / denom^2 * 1 = -frac / denom
    log_N = xp.log(ops.clamp_min(N, _EPS))  # (M,)

    ones = pred * 0.0 + 1.0
    d_L_inf = ones
    d_A = 1.0 / denom
    d_alpha = -frac / denom * N_pow * log_N[None, :]
    d_B = -frac / denom

    jac = ops.stack([d_L_inf, d_A, d_alpha, d_B], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_4 (4p): D + A * (N + B)^C  — generalized signed-exponent
# theta: [D, A, B, C]
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = X[:, 0]
    D, A, B, C = [theta[:, i] for i in range(4)]
    base = ops.clamp_min(N[None, :] + B[:, None], _EPS)  # (B_batch, M)
    log_base = xp.log(ops.clamp_min(base, _EPS))  # (B_batch, M)
    power = base ** C[:, None]  # (B_batch, M)
    term = A[:, None] * power  # (B_batch, M)
    pred = D[:, None] + term  # (B_batch, M)

    # Jacobian: (B_batch, M, 4)
    # pred = D + A * base^C
    # d/d(D) = 1
    # d/d(A) = base^C = power
    # d/d(B) = A * C * base^(C-1) * 1 = term * C / base
    # d/d(C) = A * base^C * log(base) = term * log_base
    ones = pred * 0.0 + 1.0
    d_D = ones
    d_A = power
    d_B = term * C[:, None] / base
    d_C = term * log_base

    jac = ops.stack([d_D, d_A, d_B, d_C], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_5 (4p): L_inf + A * exp(-B * N^d)  — stretched exponential (Weibull)
# theta: [L_inf, A, B, d]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, B, d = [theta[:, i] for i in range(4)]
    N_pow_d = N[None, :] ** d[:, None]  # (B_batch, M)
    exponent = -B[:, None] * N_pow_d  # (B_batch, M)
    exponent = ops.clamp(exponent, min=-50.0, max=50.0)
    exp_val = ops.exp(exponent)  # (B_batch, M)
    term = A[:, None] * exp_val  # (B_batch, M)
    pred = L_inf[:, None] + term  # (B_batch, M)

    # Jacobian: (B_batch, M, 4)
    # pred = L_inf + A * exp(-B * N^d)
    # d/d(L_inf) = 1
    # d/d(A) = exp(-B * N^d) = exp_val
    # d/d(B) = A * exp_val * (-N^d) = -term * N_pow_d
    # d/d(d) = A * exp_val * (-B * N^d * log(N)) = -term * B * N_pow_d * log(N)
    #        = term * exponent_raw * log(N)  ... but let's be explicit
    log_N = xp.log(ops.clamp_min(N, _EPS))  # (M,)

    ones = pred * 0.0 + 1.0
    d_L_inf = ones
    d_A = exp_val
    d_B = -term * N_pow_d
    d_d = -term * B[:, None] * N_pow_d * log_N[None, :]

    jac = ops.stack([d_L_inf, d_A, d_B, d_d], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_6 (4p): L_inf + A / N^alpha + (1-A) / N^beta  — dual power law
# theta: [L_inf, A, alpha, beta]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, alpha, beta = [theta[:, i] for i in range(4)]
    log_N = xp.log(ops.clamp_min(N, _EPS))  # (M,)
    N_neg_alpha = N[None, :] ** (-alpha[:, None])  # (B, M): 1/N^alpha
    N_neg_beta = N[None, :] ** (-beta[:, None])    # (B, M): 1/N^beta
    term1 = A[:, None] * N_neg_alpha                # (B, M)
    term2 = (1.0 - A[:, None]) * N_neg_beta         # (B, M)
    pred = L_inf[:, None] + term1 + term2            # (B, M)

    # Jacobian: (B, M, 4)
    # d/d(L_inf) = 1
    # d/d(A) = N^(-alpha) - N^(-beta) = N_neg_alpha - N_neg_beta
    # d/d(alpha) = A * N^(-alpha) * (-log(N)) = -term1 * log_N
    # d/d(beta) = (1-A) * N^(-beta) * (-log(N)) = -term2 * log_N
    ones = pred * 0.0 + 1.0
    d_L_inf = ones
    d_A = N_neg_alpha - N_neg_beta
    d_alpha = -term1 * log_N[None, :]
    d_beta = -term2 * log_N[None, :]

    jac = ops.stack([d_L_inf, d_A, d_alpha, d_beta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_7 (4p): L_inf + A / N^alpha + B * log(N)  — power-law + log hybrid
# theta: [L_inf, A, alpha, B]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, alpha, B = [theta[:, i] for i in range(4)]
    log_N = xp.log(ops.clamp_min(N, _EPS))  # (M,)
    N_neg_alpha = N[None, :] ** (-alpha[:, None])  # (B_batch, M)
    term1 = A[:, None] * N_neg_alpha  # (B_batch, M)
    pred = L_inf[:, None] + term1 + B[:, None] * log_N[None, :]  # (B_batch, M)

    # Jacobian: (B_batch, M, 4)
    # d/d(L_inf) = 1
    # d/d(A) = N^(-alpha)
    # d/d(alpha) = -A * N^(-alpha) * log(N) = -term1 * log_N
    # d/d(B) = log(N)
    ones = pred * 0.0 + 1.0
    d_L_inf = ones
    d_A = N_neg_alpha
    d_alpha = -term1 * log_N[None, :]
    d_B = ones * log_N[None, :]

    jac = ops.stack([d_L_inf, d_A, d_alpha, d_B], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_8 (3p): L_inf + A * N^(-alpha)  — simple unshifted power law
# theta: [L_inf, A, alpha]
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, alpha = [theta[:, i] for i in range(3)]
    log_N = xp.log(ops.clamp_min(N, _EPS))  # (M,)
    N_neg_alpha = N[None, :] ** (-alpha[:, None])  # (B, M)
    term = A[:, None] * N_neg_alpha  # (B, M)
    pred = L_inf[:, None] + term  # (B, M)

    # Jacobian: (B, M, 3)
    ones = pred * 0.0 + 1.0
    d_L_inf = ones
    d_A = N_neg_alpha
    d_alpha = -term * log_N[None, :]

    jac = ops.stack([d_L_inf, d_A, d_alpha], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_9 (2p): L_inf + A * log(N)  — log-linear
# theta: [L_inf, A]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A = [theta[:, i] for i in range(2)]
    log_N = xp.log(ops.clamp_min(N, _EPS))  # (M,)
    pred = L_inf[:, None] + A[:, None] * log_N[None, :]  # (B, M)

    # Jacobian: (B, M, 2)
    ones = pred * 0.0 + 1.0
    d_L_inf = ones
    d_A = ones * log_N[None, :]

    jac = ops.stack([d_L_inf, d_A], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_10 (4p): L_inf + A / (1 + exp(alpha * (log(N) - log(N0))))  — logistic on log-scale
# theta: [L_inf, A, alpha, N0]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, alpha, N0 = [theta[:, i] for i in range(4)]
    N0_safe = ops.clamp_min(N0, _EPS)
    log_N = xp.log(N)  # (M,)
    log_N0 = xp.log(ops.clamp_min(N0_safe, _EPS))  # (B,)
    z = log_N[None, :] - log_N0[:, None]  # (B, M)
    arg = alpha[:, None] * z  # (B, M)
    arg = ops.clamp(arg, min=-50.0, max=50.0)
    exp_arg = ops.exp(arg)  # (B, M)
    denom = 1.0 + exp_arg  # (B, M)
    frac = 1.0 / denom  # sigmoid(-arg) = 1/(1+exp(arg))
    pred = L_inf[:, None] + A[:, None] * frac  # (B, M)

    # Jacobian: (B, M, 4)
    # Let s = 1/(1+exp(arg)), arg = alpha * (log(N) - log(N0))
    # ds/d(arg) = -s * (1-s)  [since s = 1/(1+exp(arg)), ds/d(arg) = -exp(arg)/(1+exp(arg))^2 = -s*(1-s)]
    # d(arg)/d(alpha) = z = log(N) - log(N0)
    # d(arg)/d(N0) = alpha * (-1/N0)
    #
    # d/d(L_inf) = 1
    # d/d(A) = s = frac
    # d/d(alpha) = A * (-s*(1-s)) * z
    # d/d(N0) = A * (-s*(1-s)) * alpha * (-1/N0) = A * s*(1-s) * alpha / N0
    s = frac
    s_deriv = -s * (1.0 - s)  # ds/d(arg)

    ones = pred * 0.0 + 1.0
    d_L_inf = ones
    d_A = s
    d_alpha = A[:, None] * s_deriv * z
    d_N0 = A[:, None] * s_deriv * alpha[:, None] * (-1.0 / N0_safe[:, None])

    jac = ops.stack([d_L_inf, d_A, d_alpha, d_N0], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


PARAM_BOUNDS = {
    # Dataset: N ∈ [200, 819200] (doubling grid, 13 pts); sft_loss ∈ [0.59, 4.39]; 42 groups.
    # Fit per group (model × dataset). Loss decreases monotonically with more SFT data.
    # Many laws show "degeneracy": alpha→small + L_inf→-∞ when the curve hasn't converged
    # within the observed N range. Those cases have poor MSE and hit bounds deliberately.
    #
    # sl_1: [L_inf, A, N0, alpha] — L_inf + A*(N+N0)^(-alpha)
    # Fit: L_inf ∈ [-30,1.5] (hits -30 for ~7/42 slow-decay groups), A ∈ [1.3,109],
    #      N0 ∈ [-72, 1.2e4], alpha ∈ [0.005, 0.41] (hits 0.005 for ~11/42 groups)
    "sl_1": [(-30, 5), (0, 500), (-1e5, 1e5), (0.005, 0.5)],
    # sl_2: [L_inf, A, N0, alpha] — L_inf + A/(1+(N/N0)^alpha)
    # Fit: L_inf ∈ [-3.4,1.7], A ∈ [0.28,9.2], N0 ∈ [1,1e9] (some groups need N0>>N_max),
    #      alpha ∈ [0.09,0.78]. Well-behaved model (MSE median 1.4e-4).
    "sl_2": [(-5, 5), (0, 15), (1, 1e9), (0.01, 2)],
    # sl_3: [L_inf, A, alpha, B] — L_inf + A/(N^alpha+B); equivalent to sl_2 up to reparametrization
    # Fit: L_inf ∈ [-15,1.7], A ∈ [0.35,7912] (can reach ~2e7 for large N0 equivalent),
    #      alpha ∈ [0.01,0.78], B ∈ [-0.91,3608]
    "sl_3": [(-15, 5), (0, 1e8), (0.005, 2), (-1e4, 1e8)],
    # sl_4: [D, A, B, C] — D + A*(N+B)^C; equivalent to sl_1 with C=-alpha
    # Fit: D ∈ [-30,1.5] (hits -30 for ~18/42), A ∈ [1.3,109], B ∈ [-72,1.2e4], C ∈ [-0.41,-0.001]
    # C must be negative (loss decreases with N)
    "sl_4": [(-50, 5), (0, 200), (-1e5, 1e5), (-0.5, 0)],
    # sl_5: [L_inf, A, B, d] — L_inf + A*exp(-B*N^d); exp clamped to [-50,50]
    # Fit: L_inf ∈ [-15,1.8], A ∈ [0.26,500] (some degenerate groups), B ∈ [0,5.7], d ∈ [0.01,0.56]
    "sl_5": [(-15, 5), (0, 500), (0, 20), (0.01, 1)],
    # sl_6: [L_inf, A, alpha, beta] — L_inf + A/N^alpha + (1-A)/N^beta
    # Highly degenerate (A and exponents hit all bounds); allow full range for robustness
    # Fit: L_inf ∈ [-15,1.2], A ∈ [-100,100], alpha ∈ [0.01,5], beta ∈ [0.01,5]
    "sl_6": [(-15, 5), (-100, 100), (0.01, 5), (0.01, 5)],
    # sl_7: [L_inf, A, alpha, B] — L_inf + A/N^alpha + B*log(N)
    # Typically degenerates to sl_9 (A→0, 33/42 groups); physically constrained: L_inf>0, A>0, B<0
    # Fit: L_inf ∈ [0,6.2], A ∈ [0,13], alpha ∈ [0.01,1], B ∈ [-0.33,0]
    "sl_7": [(0, 10), (0, 200), (0.01, 1), (-2, 0)],
    # sl_8: [L_inf, A, alpha] — L_inf + A*N^(-alpha)
    # Fit: L_inf ∈ [-30,0.7] (hits -30 for ~15/42), A ∈ [1.8,36], alpha ∈ [0.005,0.16]
    "sl_8": [(-30, 5), (0, 50), (0.005, 0.5)],
    # sl_9: [L_inf, A] — L_inf + A*log(N); no degeneracy, cleanest model
    # Fit: L_inf ∈ [1.46,6.24] (always positive!), A ∈ [-0.33,-0.029] (always negative)
    "sl_9": [(0, 8), (-0.5, 0)],
    # sl_10: [L_inf, A, alpha, N0] — L_inf + A/(1+exp(alpha*(log(N)-log(N0))))
    # Fit: L_inf ∈ [0.36,5.7], A ∈ [-9.2,5], alpha ∈ [-0.78,0.17], N0 ∈ [1,1e9]
    # A<0 and alpha<0 is the typical physical case (loss decreases with N)
    "sl_10": [(0, 10), (-10, 10), (-1, 1), (1, 1e9)],
}

LAW_REGISTRY = {
    "sl_1": sl_1, "sl_2": sl_2, "sl_3": sl_3, "sl_4": sl_4, "sl_5": sl_5,
    "sl_6": sl_6, "sl_7": sl_7, "sl_8": sl_8, "sl_9": sl_9, "sl_10": sl_10,
}
PARAM_COUNTS = {
    "sl_1": 4, "sl_2": 4, "sl_3": 4, "sl_4": 4, "sl_5": 4,
    "sl_6": 4, "sl_7": 4, "sl_8": 3, "sl_9": 2, "sl_10": 4,
}
