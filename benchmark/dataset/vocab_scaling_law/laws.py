"""Scaling laws for vocabulary size scaling.

X columns: [non_vocab_parameters (P), vocab_size (V), num_characters (D)]
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


# sl_1 (5p): c0 + A * V^b * P^e * D^g
# theta: [c0, A, b, e, g]
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    c0, A, b, e, g = [theta[:, i] for i in range(5)]
    pred = c0[:, None] + A[:, None] * (V[None, :] ** b[:, None]) * (P[None, :] ** e[:, None]) * (D[None, :] ** g[:, None])
    return pred[0] if pred.shape[0] == 1 else pred


# sl_2 (7p): L + A * M_r(P^-alpha, D^-beta) * (1 + C * (log(V) - v0)^2)
# Generalized power mean with quadratic vocab gate
# M_r(x,y) = ((x^r + y^r)/2)^(1/r)
# theta: [L, A, alpha, beta, C, v0, r]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    L, A, alpha, beta, C, v0, r = [theta[:, i] for i in range(7)]
    term_p = P[None, :] ** (-alpha[:, None])
    term_d = D[None, :] ** (-beta[:, None])
    r_safe = ops.clamp_min(r, _EPS)
    mean_r = ((term_p ** r_safe[:, None] + term_d ** r_safe[:, None]) / 2.0) ** (1.0 / r_safe[:, None])
    vocab_gate = 1.0 + C[:, None] * (xp.log(V[None, :]) - v0[:, None]) ** 2
    pred = L[:, None] + A[:, None] * mean_r * vocab_gate
    return pred[0] if pred.shape[0] == 1 else pred


# sl_3 (7p): L0 + ((a * P^-alpha)^q + (b * (D * V^phi)^-beta)^q)^(1/q)
# Generalized q-mean of two Chinchilla-style terms
# theta: [L0, a, alpha, b, beta, phi, q]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    L0, a, alpha, b, beta, phi, q = [theta[:, i] for i in range(7)]
    t1 = a[:, None] * (P[None, :] ** (-alpha[:, None]))
    eff_D = D[None, :] * (V[None, :] ** phi[:, None])
    eff_D = ops.clamp_min(eff_D, _EPS)
    t2 = b[:, None] * (eff_D ** (-beta[:, None]))
    q_safe = ops.clamp_min(q, _EPS)
    combined = (t1 ** q_safe[:, None] + t2 ** q_safe[:, None]) ** (1.0 / q_safe[:, None])
    pred = L0[:, None] + combined
    return pred[0] if pred.shape[0] == 1 else pred


# sl_4 (7p): L_inf + A * max(P^a, lambda*D^b)^(-d) * V^(-g)
# Use softmax (logaddexp) instead of hard max for differentiability:
#   max(x,y) ~ log(exp(x) + exp(y)) but we operate in log-space
# theta: [L_inf, A, a, b, d, lam, g]
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    L_inf, A, a, b, d, lam, g = [theta[:, i] for i in range(7)]
    log_t1 = a[:, None] * xp.log(P[None, :])
    log_t2 = xp.log(ops.clamp_min(lam, _EPS))[:, None] + b[:, None] * xp.log(D[None, :])
    log_max = ops.maximum(log_t1, log_t2)
    pred = L_inf[:, None] + A[:, None] * ops.exp(-d[:, None] * log_max - g[:, None] * xp.log(V[None, :]))
    return pred[0] if pred.shape[0] == 1 else pred


# sl_5 (7p): p0 * P^p1 * V^p2 * D^p3 + p4 * P^p5 + p6
# Two additive power terms
# theta: [p0, p1, p2, p3, p4, p5, p6]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    p0, p1, p2, p3, p4, p5, p6 = [theta[:, i] for i in range(7)]
    t1 = p0[:, None] * (P[None, :] ** p1[:, None]) * (V[None, :] ** p2[:, None]) * (D[None, :] ** p3[:, None])
    t2 = p4[:, None] * (P[None, :] ** p5[:, None])
    pred = t1 + t2 + p6[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# sl_6 (7p): A * (P * V^k1)^(-alpha) + B * (D * V^k2)^(-beta) + c0
# Chinchilla dual-term, vocab modulates both eff-P and eff-D
# theta: [A, alpha, k1, B, beta, k2, c0]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    A, alpha, k1, B, beta, k2, c0 = [theta[:, i] for i in range(7)]
    eff_P = ops.clamp_min(P[None, :] * (V[None, :] ** k1[:, None]), _EPS)
    eff_D = ops.clamp_min(D[None, :] * (V[None, :] ** k2[:, None]), _EPS)
    pred = A[:, None] * (eff_P ** (-alpha[:, None])) + B[:, None] * (eff_D ** (-beta[:, None])) + c0[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# sl_7 (7p): A * P^(-alpha) * D^(-beta) + B * V^gamma * D^(-delta) + c0
# Additive P-D term + V-D interaction term
# theta: [A, alpha, beta, B, gamma, delta, c0]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    A, alpha, beta, B, gamma, delta, c0 = [theta[:, i] for i in range(7)]
    t1 = A[:, None] * (P[None, :] ** (-alpha[:, None])) * (D[None, :] ** (-beta[:, None]))
    t2 = B[:, None] * (V[None, :] ** gamma[:, None]) * (D[None, :] ** (-delta[:, None]))
    pred = t1 + t2 + c0[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# sl_8 (7p): c0 + c1*log(V) + V^beta * (c2 * P^(-alpha) + c3 * D^(-gamma))
# Log-V floor + V-power scaled reducible terms
# theta: [c0, c1, c2, alpha, c3, gamma, beta]
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    c0, c1, c2, alpha, c3, gamma, beta = [theta[:, i] for i in range(7)]
    floor = c0[:, None] + c1[:, None] * xp.log(V[None, :])
    reducible = c2[:, None] * (P[None, :] ** (-alpha[:, None])) + c3[:, None] * (D[None, :] ** (-gamma[:, None]))
    pred = floor + (V[None, :] ** beta[:, None]) * reducible
    return pred[0] if pred.shape[0] == 1 else pred


# sl_9 (7p): A * P^(-alpha) * D^(-beta) * (1 + gamma*log(V)) + delta * V^epsilon + L_inf
# Chinchilla with log-vocab modulation + vocab cross term
# theta: [A, alpha, beta, gamma, delta, epsilon, L_inf]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    A, alpha, beta, gamma, delta, epsilon, L_inf = [theta[:, i] for i in range(7)]
    main = A[:, None] * (P[None, :] ** (-alpha[:, None])) * (D[None, :] ** (-beta[:, None]))
    vocab_mod = 1.0 + gamma[:, None] * xp.log(V[None, :])
    cross = delta[:, None] * (V[None, :] ** epsilon[:, None])
    pred = main * vocab_mod + cross + L_inf[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# sl_10 (7p): L_min + exp(a + bP*log(P) + bV1*log(V) + bV2*log(V)^2 + bD*log(D) + bVD*log(V)*log(D))
# Log-space polynomial with quadratic vocab and V-D interaction
# theta: [L_min, a, bP, bV1, bV2, bD, bVD]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    L_min, a, bP, bV1, bV2, bD, bVD = [theta[:, i] for i in range(7)]
    lP = xp.log(P[None, :])
    lV = xp.log(V[None, :])
    lD = xp.log(D[None, :])
    exponent = a[:, None] + bP[:, None] * lP + bV1[:, None] * lV + bV2[:, None] * lV ** 2 + bD[:, None] * lD + bVD[:, None] * lV * lD
    exponent = ops.clamp(exponent, min=-50.0, max=50.0)
    pred = L_min[:, None] + ops.exp(exponent)
    return pred[0] if pred.shape[0] == 1 else pred


PARAM_BOUNDS = {
    # sl_1: [c0, A, b, e, g]
    "sl_1": [(-100, 100), (-1e4, 1e4), (-5, 5), (-5, 5), (-5, 5)],
    # sl_2: [L, A, alpha, beta, C, v0, r]
    "sl_2": [(-100, 100), (-1e4, 1e4), (-5, 5), (-5, 5), (-100, 100), (-10, 30), (0.1, 10)],
    # sl_3: [L0, a, alpha, b, beta, phi, q]
    "sl_3": [(-100, 100), (-1e4, 1e4), (-5, 5), (-1e4, 1e4), (-5, 5), (-5, 5), (0.1, 10)],
    # sl_4: [L_inf, A, a, b, d, lam, g]
    "sl_4": [(-100, 100), (-1e4, 1e4), (-5, 5), (-5, 5), (-5, 5), (1e-6, 1e4), (-5, 5)],
    # sl_5: [p0, p1, p2, p3, p4, p5, p6]
    "sl_5": [(-1e4, 1e4), (-5, 5), (-5, 5), (-5, 5), (-1e4, 1e4), (-5, 5), (-100, 100)],
    # sl_6: [A, alpha, k1, B, beta, k2, c0]
    "sl_6": [(-1e4, 1e4), (-5, 5), (-5, 5), (-1e4, 1e4), (-5, 5), (-5, 5), (-100, 100)],
    # sl_7: [A, alpha, beta, B, gamma, delta, c0]
    "sl_7": [(-1e4, 1e4), (-5, 5), (-5, 5), (-1e4, 1e4), (-5, 5), (-5, 5), (-100, 100)],
    # sl_8: [c0, c1, c2, alpha, c3, gamma, beta]
    "sl_8": [(-100, 100), (-100, 100), (-1e4, 1e4), (-5, 5), (-1e4, 1e4), (-5, 5), (-5, 5)],
    # sl_9: [A, alpha, beta, gamma, delta, epsilon, L_inf]
    "sl_9": [(-1e4, 1e4), (-5, 5), (-5, 5), (-100, 100), (-1e4, 1e4), (-5, 5), (-100, 100)],
    # sl_10: [L_min, a, bP, bV1, bV2, bD, bVD] — exp(poly) with clamp [-50,50]
    "sl_10": [(-100, 100), (-20, 20), (-2, 2), (-2, 2), (-0.1, 0.1), (-2, 2), (-0.1, 0.1)],
}

LAW_REGISTRY = {
    "sl_1": sl_1, "sl_2": sl_2, "sl_3": sl_3, "sl_4": sl_4, "sl_5": sl_5,
    "sl_6": sl_6, "sl_7": sl_7, "sl_8": sl_8, "sl_9": sl_9, "sl_10": sl_10,
}
PARAM_COUNTS = {
    "sl_1": 5, "sl_2": 7, "sl_3": 7, "sl_4": 7, "sl_5": 7,
    "sl_6": 7, "sl_7": 7, "sl_8": 7, "sl_9": 7, "sl_10": 7,
}
