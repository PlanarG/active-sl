"""Scaling laws for parallel (tensor/pipeline) scaling.

X columns: [num_params (N), parallel_size (S)]
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


# sl_1 (6p): c0 + cN * N^(-α) + cS * S^(-β) + cNS * N^(-α) * S^(-β)
# theta: [c0, cN, alpha, cS, beta, cNS]
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    c0, cN, alpha, cS, beta, cNS = [theta[:, i] for i in range(6)]
    Na = N[None, :] ** (-alpha[:, None])
    Sb = S[None, :] ** (-beta[:, None])
    pred = c0[:, None] + cN[:, None] * Na + cS[:, None] * Sb + cNS[:, None] * Na * Sb
    return pred[0] if pred.shape[0] == 1 else pred


# sl_2 (5p): c0 + cN * N^(-α) + cS * S^(-β)
# theta: [c0, cN, alpha, cS, beta]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    c0, cN, alpha, cS, beta = [theta[:, i] for i in range(5)]
    pred = c0[:, None] + cN[:, None] * (N[None, :] ** (-alpha[:, None])) + cS[:, None] * (S[None, :] ** (-beta[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


# sl_3 (4p): a * N^b + c / (1 + S) + d
# theta: [a, b, c, d]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    a, b, c, d = [theta[:, i] for i in range(4)]
    pred = a[:, None] * (N[None, :] ** b[:, None]) + c[:, None] / (1.0 + S[None, :]) + d[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# sl_4 (4p): a * N^b + c * S^(-0.5) + d  (fixed beta=0.5)
# theta: [a, b, c, d]
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    a, b, c, d = [theta[:, i] for i in range(4)]
    pred = a[:, None] * (N[None, :] ** b[:, None]) + c[:, None] * (S[None, :] ** (-0.5)) + d[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# sl_5 (4p): (A / (N * (k * log(S) + 1)))^α + E
# theta: [A, k, alpha, E]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], 1.0)
    A, k, alpha, E = [theta[:, i] for i in range(4)]
    denom = N[None, :] * (k[:, None] * xp.log(S[None, :]) + 1.0)
    denom = ops.clamp_min(denom, _EPS)
    base = A[:, None] / denom
    base = ops.clamp_min(base, _EPS)
    pred = base ** alpha[:, None] + E[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# sl_6 (4p): c0 + c1 * (N^(-α) + S^(-β))
# theta: [c0, c1, alpha, beta]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    c0, c1, alpha, beta = [theta[:, i] for i in range(4)]
    pred = c0[:, None] + c1[:, None] * (N[None, :] ** (-alpha[:, None]) + S[None, :] ** (-beta[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


# sl_7 (4p): L0 + A * N^(-α) / (1 + k * ln(S))
# theta: [L0, A, alpha, k]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], 1.0)
    L0, A, alpha, k = [theta[:, i] for i in range(4)]
    numer = A[:, None] * (N[None, :] ** (-alpha[:, None]))
    denom = 1.0 + k[:, None] * xp.log(S[None, :])
    denom = ops.clamp_min(denom, _EPS)
    pred = L0[:, None] + numer / denom
    return pred[0] if pred.shape[0] == 1 else pred


# sl_8 (4p): (a * N^b + c) / (1 + d * log(S))
# theta: [a, b, c, d]
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], 1.0)
    a, b, c, d = [theta[:, i] for i in range(4)]
    numer = a[:, None] * (N[None, :] ** b[:, None]) + c[:, None]
    denom = 1.0 + d[:, None] * xp.log(S[None, :])
    denom = ops.clamp_min(denom, _EPS)
    pred = numer / denom
    return pred[0] if pred.shape[0] == 1 else pred


# sl_9 (3p): A * N^(-α) * S^(-β)
# theta: [A, alpha, beta]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    A, alpha, beta = [theta[:, i] for i in range(3)]
    pred = A[:, None] * (N[None, :] ** (-alpha[:, None])) * (S[None, :] ** (-beta[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


# sl_10 (4p): (A * N^(-α) + E) * S^(-β)
# theta: [A, alpha, E, beta]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    A, alpha, E, beta = [theta[:, i] for i in range(4)]
    pred = (A[:, None] * (N[None, :] ** (-alpha[:, None])) + E[:, None]) * (S[None, :] ** (-beta[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


PARAM_BOUNDS = {
    # Dataset: N ∈ [5.4e8, 4.4e9], log(N) ∈ [20.1, 22.2]; S ∈ {1,2,4,8}; loss ∈ [0.98, 2.11]
    # Groups (pile/stack) have different loss floors; fit per group.
    # sl_1: [c0, cN, alpha, cS, beta, cNS] — c0+cN*N^(-alpha)+cS*S^(-beta)+cNS*N^(-alpha)*S^(-beta)
    # Fit pile: [1.38, 112.7, 0.260, 0.097, 0.399, 5.69]; stack: [0.763, 66.0, 0.263, 0.049, 0.470, 5.39]
    "sl_1": [(-5, 5), (0, 500), (0, 2), (-50, 50), (0, 2), (-100, 100)],
    # sl_2: [c0, cN, alpha, cS, beta] — c0 + cN*N^(-alpha) + cS*S^(-beta)
    # Fit pile: [1.364, 117.8, 0.261, 0.121, 0.399]; stack: [0.750, 70.7, 0.264, 0.070, 0.472]
    "sl_2": [(-5, 5), (0, 500), (0, 2), (-50, 50), (0, 2)],
    # sl_3: [a, b, c, d] — a*N^b + c/(1+S) + d; b<0 required (more params → lower loss)
    # Fit pile: [117.9, -0.261, 0.174, 1.398]; stack: [70.7, -0.264, 0.112, 0.764]
    "sl_3": [(0, 500), (-2, 0), (-5, 5), (-5, 5)],
    # sl_4: [a, b, c, d] — a*N^b + c*S^(-0.5) + d; b<0 required
    # Fit pile: [117.9, -0.261, 0.105, 1.381]; stack: [70.7, -0.264, 0.068, 0.753]
    "sl_4": [(0, 500), (-2, 0), (-5, 5), (-5, 5)],
    # sl_5: [A, k, alpha, E] — (A/(N*(k*log(S)+1)))^alpha + E
    # A must be ~N*constant: A/N_min ∈ [0.02, 1.85] for physical predictions
    # Fit pile: [1.95e8, 0.334, 0.196, 1.291]; stack: [1.14e7, 0.396, 0.199, 0.709]
    "sl_5": [(0, 1e9), (-100, 100), (0, 3), (-5, 5)],
    # sl_6: [c0, c1, alpha, beta] — c0 + c1*(N^(-alpha) + S^(-beta))
    # Fit pile: [-115.8, 117.3, 0.261, ~0]; stack: [-69.4, 70.2, 0.264, ~0]
    # c0 hits -115 for pile (old -100 bound was too tight); beta≈0 (structural degeneracy)
    "sl_6": [(-200, 200), (-200, 200), (0, 2), (0, 2)],
    # sl_7: [L0, A, alpha, k] — L0 + A*N^(-alpha)/(1+k*ln(S))
    # Fit pile: [1.315, 48.1, 0.204, 0.055]; stack: [0.728, 30.6, 0.211, 0.065]
    "sl_7": [(-5, 5), (0, 500), (0, 2), (-50, 50)],
    # sl_8: [a, b, c, d] — (a*N^b + c)/(1+d*log(S)); b<0 required
    # Fit pile: [118.1, -0.260, 1.472, 0.017]; stack: [71.0, -0.263, 0.812, 0.020]
    "sl_8": [(0, 500), (-2, 0), (-5, 5), (-10, 10)],
    # sl_9: [A, alpha, beta] — A*N^(-alpha)*S^(-beta)
    # Fit pile: [7.73, 0.065, 0.017]; stack: [4.45, 0.067, 0.020] (small exponents)
    "sl_9": [(0, 5000), (0, 2), (0, 2)],
    # sl_10: [A, alpha, E, beta] — (A*N^(-alpha) + E)*S^(-beta)
    # Fit pile: [118.1, 0.260, 1.471, 0.017]; stack: [71.0, 0.263, 0.812, 0.020]
    "sl_10": [(0, 500), (0, 2), (-5, 5), (0, 2)],
}

LAW_REGISTRY = {
    "sl_1": sl_1, "sl_2": sl_2, "sl_3": sl_3, "sl_4": sl_4, "sl_5": sl_5,
    "sl_6": sl_6, "sl_7": sl_7, "sl_8": sl_8, "sl_9": sl_9, "sl_10": sl_10,
}
PARAM_COUNTS = {
    "sl_1": 6, "sl_2": 5, "sl_3": 4, "sl_4": 4, "sl_5": 4,
    "sl_6": 4, "sl_7": 4, "sl_8": 4, "sl_9": 3, "sl_10": 4,
}
