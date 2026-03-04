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
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = X[:, 0]
    L_inf, A, N0, alpha = [theta[:, i] for i in range(4)]
    base = ops.clamp_min(N[None, :] + N0[:, None], _EPS)
    pred = L_inf[:, None] + A[:, None] * (base ** (-alpha[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


# sl_2 (4p): L_inf + A / (1 + (N/N0)^alpha)  — Hill/Michaelis-Menten
# theta: [L_inf, A, N0, alpha]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, N0, alpha = [theta[:, i] for i in range(4)]
    N0_safe = ops.clamp_min(N0, _EPS)
    ratio = (N[None, :] / N0_safe[:, None]) ** alpha[:, None]
    pred = L_inf[:, None] + A[:, None] / (1.0 + ratio)
    return pred[0] if pred.shape[0] == 1 else pred


# sl_3 (4p): L_inf + A / (N^alpha + B)  — denominator-offset power law
# theta: [L_inf, A, alpha, B]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, alpha, B = [theta[:, i] for i in range(4)]
    denom = N[None, :] ** alpha[:, None] + B[:, None]
    denom = ops.clamp_min(denom, _EPS)
    pred = L_inf[:, None] + A[:, None] / denom
    return pred[0] if pred.shape[0] == 1 else pred


# sl_4 (4p): D + A * (N + B)^C  — generalized signed-exponent
# theta: [D, A, B, C]
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = X[:, 0]
    D, A, B, C = [theta[:, i] for i in range(4)]
    base = ops.clamp_min(N[None, :] + B[:, None], _EPS)
    pred = D[:, None] + A[:, None] * (base ** C[:, None])
    return pred[0] if pred.shape[0] == 1 else pred


# sl_5 (4p): L_inf + A * exp(-B * N^d)  — stretched exponential (Weibull)
# theta: [L_inf, A, B, d]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, B, d = [theta[:, i] for i in range(4)]
    exponent = -B[:, None] * (N[None, :] ** d[:, None])
    exponent = ops.clamp(exponent, min=-50.0, max=50.0)
    pred = L_inf[:, None] + A[:, None] * ops.exp(exponent)
    return pred[0] if pred.shape[0] == 1 else pred


# sl_6 (4p): L_inf + A / N^alpha + B / N^beta  — dual power law
# theta: [L_inf, A, alpha, beta]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, alpha, beta = [theta[:, i] for i in range(4)]
    pred = L_inf[:, None] + A[:, None] / (N[None, :] ** alpha[:, None]) + (1.0 - A[:, None]) / (N[None, :] ** beta[:, None])
    return pred[0] if pred.shape[0] == 1 else pred


# sl_7 (4p): L_inf + A / N^alpha + B * log(N)  — power-law + log hybrid
# theta: [L_inf, A, alpha, B]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, alpha, B = [theta[:, i] for i in range(4)]
    pred = L_inf[:, None] + A[:, None] / (N[None, :] ** alpha[:, None]) + B[:, None] * xp.log(N[None, :])
    return pred[0] if pred.shape[0] == 1 else pred


# sl_8 (3p): L_inf + A * N^(-alpha)  — simple unshifted power law
# theta: [L_inf, A, alpha]
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A, alpha = [theta[:, i] for i in range(3)]
    pred = L_inf[:, None] + A[:, None] * (N[None, :] ** (-alpha[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


# sl_9 (2p): L_inf + A * log(N)  — log-linear
# theta: [L_inf, A]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    L_inf, A = [theta[:, i] for i in range(2)]
    pred = L_inf[:, None] + A[:, None] * xp.log(N[None, :])
    return pred[0] if pred.shape[0] == 1 else pred


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
    arg = alpha[:, None] * (xp.log(N[None, :]) - xp.log(N0_safe[:, None]))
    arg = ops.clamp(arg, min=-50.0, max=50.0)
    pred = L_inf[:, None] + A[:, None] / (1.0 + ops.exp(arg))
    return pred[0] if pred.shape[0] == 1 else pred


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
