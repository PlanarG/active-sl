"""Scaling laws for Mixture-of-Experts models.

X columns: [num_experts (E), dense_parameter_count (N)]
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


# Scaling law 1 (4 params):
#   L_inf + B / (N^alpha * E^beta)
# theta: [L_inf, B, alpha, beta]
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    L_inf = theta[:, 0]
    B = theta[:, 1]
    alpha = theta[:, 2]
    beta = theta[:, 3]

    denom = (N[None, :] ** alpha[:, None]) * (E[None, :] ** beta[:, None])
    denom = ops.clamp_min(denom, _EPS)

    pred = L_inf[:, None] + B[:, None] / denom
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 2 (5 params):
#   L + K * (N^alpha * E^beta)^(-gamma)
# theta: [L, K, alpha, beta, gamma]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    L = theta[:, 0]
    K = theta[:, 1]
    alpha = theta[:, 2]
    beta = theta[:, 3]
    gamma = theta[:, 4]

    base = (N[None, :] ** alpha[:, None]) * (E[None, :] ** beta[:, None])
    base = ops.clamp_min(base, _EPS)

    pred = L[:, None] + K[:, None] * (base ** (-gamma[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 3 (6 params):
#   A * P^alpha / (1 + B * E^beta) + C * P^(alpha*0.6) + D
#   (gamma = alpha * 0.6 is hard-coded)
# theta: [A, alpha, B, beta, C, D]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    A = theta[:, 0]
    alpha = theta[:, 1]
    B = theta[:, 2]
    beta = theta[:, 3]
    C = theta[:, 4]
    D = theta[:, 5]

    efficiency = A[:, None] * (N[None, :] ** alpha[:, None]) / (
        1.0 + B[:, None] * (E[None, :] ** beta[:, None])
    )
    param_scale = C[:, None] * (N[None, :] ** (alpha[:, None] * 0.6))

    pred = efficiency + param_scale + D[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 4 (6 params):
#   a / (N^alpha * (1 + b*E)^gamma) + c + d*(log(N) - 0.4*log(1+E))
# theta: [a, alpha, b, gamma, c, d]
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], 1.0)
    N = ops.clamp_min(X[:, 1], _EPS)

    a = theta[:, 0]
    alpha = theta[:, 1]
    b = theta[:, 2]
    gamma = theta[:, 3]
    c = theta[:, 4]
    d = theta[:, 5]

    expert_sat = (1.0 + b[:, None] * E[None, :]) ** gamma[:, None]
    expert_sat = ops.clamp_min(expert_sat, _EPS)
    main = a[:, None] / ((N[None, :] ** alpha[:, None]) * expert_sat)

    log_correction = d[:, None] * (xp.log(N[None, :]) - 0.4 * xp.log(1.0 + E[None, :]))

    pred = main + c[:, None] + log_correction
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 5 (6 params):
#   p0 + exp(p1 + p2*log(E) + p3*log(P) + p4*log(E)*log(P)) + p5*log(E)
# theta: [p0, p1, p2, p3, p4, p5]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], 1.0)
    N = ops.clamp_min(X[:, 1], _EPS)

    p0 = theta[:, 0]
    p1 = theta[:, 1]
    p2 = theta[:, 2]
    p3 = theta[:, 3]
    p4 = theta[:, 4]
    p5 = theta[:, 5]

    log_E = xp.log(E)[None, :]   # (1, M)
    log_N = xp.log(N)[None, :]   # (1, M)

    exponent = (
        p1[:, None]
        + p2[:, None] * log_E
        + p3[:, None] * log_N
        + p4[:, None] * log_E * log_N
    )
    # Clip exponent for numerical safety
    exponent = ops.clamp(exponent, min=-50.0, max=50.0)

    pred = p0[:, None] + ops.exp(exponent) + p5[:, None] * log_E
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 6 (6 params):
#   a * N^(-b) * (1 + c*E^(-d)) + e + f/(E * N^0.05)
# theta: [a, b, c, d, e, f]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], 1.0)
    N = ops.clamp_min(X[:, 1], _EPS)

    a = theta[:, 0]
    b = theta[:, 1]
    c = theta[:, 2]
    d = theta[:, 3]
    e = theta[:, 4]
    f = theta[:, 5]

    base = a[:, None] * (N[None, :] ** (-b[:, None]))
    expert_mod = 1.0 + c[:, None] * (E[None, :] ** (-d[:, None]))
    interaction = f[:, None] / (E[None, :] * (N[None, :] ** 0.05))

    pred = base * expert_mod + e[:, None] + interaction
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 7 (6 params):
#   p0 * E^p1 * P^p2 + p3 * P^p4 + p5
#   (multiplicative + additive power law)
# theta: [p0, p1, p2, p3, p4, p5]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    p0 = theta[:, 0]
    p1 = theta[:, 1]
    p2 = theta[:, 2]
    p3 = theta[:, 3]
    p4 = theta[:, 4]
    p5 = theta[:, 5]

    term1 = p0[:, None] * (E[None, :] ** p1[:, None]) * (N[None, :] ** p2[:, None])
    term2 = p3[:, None] * (N[None, :] ** p4[:, None])

    pred = term1 + term2 + p5[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 8 (4 params):
#   a * N^b * E^c + d
# theta: [a, b, c, d]
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    a = theta[:, 0]
    b = theta[:, 1]
    c = theta[:, 2]
    d = theta[:, 3]

    pred = a[:, None] * (N[None, :] ** b[:, None]) * (E[None, :] ** c[:, None]) + d[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 9 (4 params):
#   c0 + A * (N * E^g)^(-a)
# theta: [c0, A, g, a]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    c0 = theta[:, 0]
    A = theta[:, 1]
    g = theta[:, 2]
    a = theta[:, 3]

    N_eff = N[None, :] * (E[None, :] ** g[:, None])
    N_eff = ops.clamp_min(N_eff, _EPS)

    pred = c0[:, None] + A[:, None] * (N_eff ** (-a[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 10 (6 params):
#   bias + A * (N/1e9)^(-alpha) * ((1 + B*E^gamma) / (1 + B))^(-beta)
# theta: [bias, A, alpha, B, gamma, beta]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    bias = theta[:, 0]
    A = theta[:, 1]
    alpha = theta[:, 2]
    B = theta[:, 3]
    gamma = theta[:, 4]
    beta = theta[:, 5]

    N_scaled = N[None, :] / 1e9
    N_scaled = ops.clamp_min(N_scaled, _EPS)

    term_N = N_scaled ** (-alpha[:, None])

    expert_num = 1.0 + B[:, None] * (E[None, :] ** gamma[:, None])
    expert_den = ops.clamp_min(1.0 + B[:, None], _EPS)
    term_E = (expert_num / expert_den) ** (-beta[:, None])

    pred = bias[:, None] + A[:, None] * term_N * term_E
    return pred[0] if pred.shape[0] == 1 else pred


PARAM_BOUNDS = {
    # Dataset: E ∈ {1,2,...,512}, log(E) ∈ [0,6.24]; N ∈ [1.65e7,1.31e9]; loss ∈ [2.0,3.16]
    # sl_1: [L_inf, B, alpha, beta] — L_inf + B / (N^alpha * E^beta)
    # Fit: L_inf≈1.55, B≈38, alpha≈0.19, beta≈0.07
    "sl_1": [(0, 3), (0, 500), (0, 1.5), (-0.5, 1.5)],
    # sl_2: [L, K, alpha, beta, gamma] — L + K*(N^alpha * E^beta)^(-gamma)
    # Fit: L≈1.55, K≈38, alpha≈0.88, beta≈0.32, gamma≈0.22; alpha*gamma≈0.19, beta*gamma≈0.07
    "sl_2": [(0, 3), (0, 500), (0, 2), (0, 2), (0, 2)],
    # sl_3: [A, alpha, B, beta, C, D] — A*N^alpha/(1+B*E^beta) + C*N^(alpha*0.6) + D
    # Fit: A≈28, alpha≈-0.22, B≈0.12, beta≈0.64, C≈10.7, D≈1.32
    # alpha<0 required: N^alpha decreases loss as N grows
    "sl_3": [(0, 5e3), (-1.5, 0), (0, 20), (0, 2), (-1e4, 1e4), (-5, 5)],
    # sl_4: [a, alpha, b, gamma, c, d] — a/(N^alpha*(1+b*E)^gamma) + c + d*(logN-0.4*log(1+E))
    # Fit: a≈36, alpha≈0.19, b≈0.39, gamma≈0.12, c≈2.21, d≈-0.03
    "sl_4": [(0, 500), (0, 1.5), (0, 20), (0, 1.5), (-5, 5), (-1, 1)],
    # sl_5: [p0,p1,p2,p3,p4,p5] — p0 + exp(p1+p2*logE+p3*logN+p4*logE*logN) + p5*logE
    # Fit: p0≈1.59, p1≈3.80, p2≈-0.24, p3≈-0.20, p4≈0.013, p5≈-0.07
    # Exponent clamped to [-50,50] in model; p3*logN_max≈-0.2*21=-4.2 keeps p1<10 safe
    "sl_5": [(0, 3), (-5, 10), (-1, 1), (-1, 1), (-0.1, 0.1), (-1, 1)],
    # sl_6: [a, b, c, d, e, f] — a*N^(-b)*(1+c*E^(-d)) + e + f/(E*N^0.05)
    # Fit: a≈10.8, b≈0.17, c≈2.03, d≈0.14, e≈1.49, f≈-0.41
    "sl_6": [(0, 200), (0, 1.5), (-5, 20), (-0.5, 2), (-5, 5), (-20, 20)],
    # sl_7: [p0,p1,p2,p3,p4,p5] — p0*E^p1*N^p2 + p3*N^p4 + p5
    # Fit degenerate: p0≈6089, p1≈0, p2≈p4≈-0.215, p3≈-6060, p5≈1.51
    # True behavior: (p0+p3)*N^p2 + p5 ≈ 29*N^(-0.215) + 1.5 (p1≈0 kills E-dependence)
    "sl_7": [(-500, 500), (-1, 0.5), (-1.5, 0.5), (-500, 500), (-1.5, 0.5), (-5, 5)],
    # sl_8: [a, b, c, d] — a*N^b * E^c + d
    # Fit: a≈37.9, b≈-0.19, c≈-0.07, d≈1.55
    "sl_8": [(0, 500), (-1.5, 0.5), (-1.5, 0.5), (0, 3)],
    # sl_9: [c0, A, g, a] — c0 + A*(N*E^g)^(-a)
    # Fit: c0≈1.55, A≈37.9, g≈0.37, a≈0.19
    "sl_9": [(0, 3), (0, 500), (-1, 2), (0, 2)],
    # sl_10: [bias, A, alpha, B, gamma, beta] — bias + A*(N/1e9)^(-alpha)*((1+B*E^gamma)/(1+B))^(-beta)
    # Fit: bias≈1.59, A≈0.70, alpha≈0.20, B≈0.1 (near 0=degenerate: expert term→1), gamma≈2.62, beta≈0.03
    "sl_10": [(0, 3), (0, 15), (0, 2), (0, 100), (0, 6), (0, 2)],
}

LAW_REGISTRY = {
    "sl_1": sl_1, "sl_2": sl_2, "sl_3": sl_3, "sl_4": sl_4, "sl_5": sl_5,
    "sl_6": sl_6, "sl_7": sl_7, "sl_8": sl_8, "sl_9": sl_9, "sl_10": sl_10,
}
PARAM_COUNTS = {
    "sl_1": 4, "sl_2": 5, "sl_3": 6, "sl_4": 6, "sl_5": 6,
    "sl_6": 6, "sl_7": 6, "sl_8": 4, "sl_9": 4, "sl_10": 6,
}
