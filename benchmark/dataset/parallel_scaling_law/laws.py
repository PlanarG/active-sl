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
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    c0, cN, alpha, cS, beta, cNS = [theta[:, i] for i in range(6)]
    Na = N[None, :] ** (-alpha[:, None])   # (B, M)
    Sb = S[None, :] ** (-beta[:, None])    # (B, M)
    NaSb = Na * Sb
    pred = c0[:, None] + cN[:, None] * Na + cS[:, None] * Sb + cNS[:, None] * NaSb

    # Jacobian: d/d[c0, cN, alpha, cS, beta, cNS]
    ones = pred * 0.0 + 1.0
    logN = xp.log(ops.clamp_min(N[None, :], _EPS))  # (1, M) or (B, M) after broadcast
    logS = xp.log(ops.clamp_min(S[None, :], _EPS))

    d_c0 = ones
    d_cN = Na
    # d(Na)/d(alpha) = d(N^(-alpha))/d(alpha) = -N^(-alpha)*log(N) = -Na*logN
    d_alpha = (cN[:, None] * Na + cNS[:, None] * NaSb) * (-logN)
    d_cS = Sb
    # d(Sb)/d(beta) = -Sb*logS
    d_beta = (cS[:, None] * Sb + cNS[:, None] * NaSb) * (-logS)
    d_cNS = NaSb
    jac = ops.stack([d_c0, d_cN, d_alpha, d_cS, d_beta, d_cNS], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_2 (5p): c0 + cN * N^(-α) + cS * S^(-β)
# theta: [c0, cN, alpha, cS, beta]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    c0, cN, alpha, cS, beta = [theta[:, i] for i in range(5)]
    Na = N[None, :] ** (-alpha[:, None])
    Sb = S[None, :] ** (-beta[:, None])
    pred = c0[:, None] + cN[:, None] * Na + cS[:, None] * Sb

    # Jacobian: d/d[c0, cN, alpha, cS, beta]
    ones = pred * 0.0 + 1.0
    logN = xp.log(ops.clamp_min(N[None, :], _EPS))
    logS = xp.log(ops.clamp_min(S[None, :], _EPS))

    d_c0 = ones
    d_cN = Na
    d_alpha = cN[:, None] * Na * (-logN)
    d_cS = Sb
    d_beta = cS[:, None] * Sb * (-logS)
    jac = ops.stack([d_c0, d_cN, d_alpha, d_cS, d_beta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_3 (4p): a * N^b + c / (1 + S) + d
# theta: [a, b, c, d]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    a, b, c, d = [theta[:, i] for i in range(4)]
    Nb = N[None, :] ** b[:, None]
    inv_1pS = 1.0 / (1.0 + S[None, :])
    pred = a[:, None] * Nb + c[:, None] * inv_1pS + d[:, None]

    # Jacobian: d/d[a, b, c, d]
    ones = pred * 0.0 + 1.0
    logN = xp.log(ops.clamp_min(N[None, :], _EPS))

    d_a = Nb
    d_b = a[:, None] * Nb * logN
    d_c = inv_1pS + ones * 0.0  # broadcast
    d_d = ones
    jac = ops.stack([d_a, d_b, d_c, d_d], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_4 (4p): a * N^b + c * S^(-0.5) + d  (fixed beta=0.5)
# theta: [a, b, c, d]
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    a, b, c, d = [theta[:, i] for i in range(4)]
    Nb = N[None, :] ** b[:, None]
    S_inv_half = S[None, :] ** (-0.5)
    pred = a[:, None] * Nb + c[:, None] * S_inv_half + d[:, None]

    # Jacobian: d/d[a, b, c, d]
    ones = pred * 0.0 + 1.0
    logN = xp.log(ops.clamp_min(N[None, :], _EPS))

    d_a = Nb
    d_b = a[:, None] * Nb * logN
    d_c = S_inv_half + ones * 0.0  # broadcast
    d_d = ones
    jac = ops.stack([d_a, d_b, d_c, d_d], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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
    logS = xp.log(S[None, :])
    denom = N[None, :] * (k[:, None] * logS + 1.0)
    denom = ops.clamp_min(denom, _EPS)
    base = A[:, None] / denom
    base = ops.clamp_min(base, _EPS)
    power = base ** alpha[:, None]  # base^alpha
    pred = power + E[:, None]

    # Jacobian: d/d[A, k, alpha, E]
    # Let f = base^alpha, pred = f + E
    # base = A / denom, denom = N*(k*logS + 1)
    ones = pred * 0.0 + 1.0
    log_base = xp.log(ops.clamp_min(base, _EPS))

    # d(pred)/dA: d(base^alpha)/dA = alpha * base^(alpha-1) * (1/denom)
    #           = alpha * base^alpha * (1/base) * (1/denom) = alpha * power / A[:, None]
    # But safer: alpha * power * (1/base) * d(base)/dA = alpha * power / base * (1/denom)
    #          = alpha * power / (A/denom) * (1/denom) = alpha * power / A
    d_A = alpha[:, None] * power / ops.clamp_min(A[:, None], _EPS)

    # d(pred)/dk: d(base^alpha)/dk = alpha * base^(alpha-1) * d(base)/dk
    # d(base)/dk = A * d(1/denom)/dk = -A * logS * N / denom^2 = -base * logS * N / denom
    # Actually: d(base)/dk = -A * N * logS / denom^2 = -(A/denom) * (N*logS)/denom = -base * logS / (k*logS+1)
    # Simpler: d(base)/dk = -A * N * logS / denom^2
    # So d(f)/dk = alpha * power / base * (-A * N * logS / denom^2)
    #            = alpha * power * (-N * logS / denom) / base ... hmm
    # Let's use: d(log(base))/dk = d(log(A) - log(denom))/dk = -d(log(denom))/dk
    # d(log(denom))/dk = N*logS / denom ... but denom = N*(k*logS+1)
    # d(denom)/dk = N*logS, so d(log(denom))/dk = N*logS/denom = logS/(k*logS+1)
    # d(f)/dk = f * alpha * d(log(base))/dk = -power * alpha * logS / (k[:, None] * logS + 1.0)
    k_logS_1 = ops.clamp_min(k[:, None] * logS + 1.0, _EPS)
    d_k = -power * alpha[:, None] * logS / k_logS_1

    # d(pred)/dalpha: d(base^alpha)/dalpha = base^alpha * log(base) = power * log(base)
    d_alpha = power * log_base

    d_E = ones
    jac = ops.stack([d_A, d_k, d_alpha, d_E], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_6 (4p): c0 + c1 * (N^(-α) + S^(-β))
# theta: [c0, c1, alpha, beta]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    c0, c1, alpha, beta = [theta[:, i] for i in range(4)]
    Na = N[None, :] ** (-alpha[:, None])
    Sb = S[None, :] ** (-beta[:, None])
    sumNS = Na + Sb
    pred = c0[:, None] + c1[:, None] * sumNS

    # Jacobian: d/d[c0, c1, alpha, beta]
    ones = pred * 0.0 + 1.0
    logN = xp.log(ops.clamp_min(N[None, :], _EPS))
    logS = xp.log(ops.clamp_min(S[None, :], _EPS))

    d_c0 = ones
    d_c1 = sumNS
    d_alpha = c1[:, None] * Na * (-logN)
    d_beta = c1[:, None] * Sb * (-logS)
    jac = ops.stack([d_c0, d_c1, d_alpha, d_beta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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
    Na = N[None, :] ** (-alpha[:, None])
    logS = xp.log(S[None, :])
    denom = 1.0 + k[:, None] * logS
    denom = ops.clamp_min(denom, _EPS)
    numer = A[:, None] * Na
    frac = numer / denom
    pred = L0[:, None] + frac

    # Jacobian: d/d[L0, A, alpha, k]
    ones = pred * 0.0 + 1.0
    logN = xp.log(ops.clamp_min(N[None, :], _EPS))

    d_L0 = ones
    d_A = Na / denom
    # d(frac)/d(alpha) = A * d(Na)/d(alpha) / denom = A * (-Na*logN) / denom = -frac * logN
    d_alpha = -frac * logN
    # d(frac)/dk = -numer * logS / denom^2 = -frac * logS / denom
    d_k = -frac * logS / denom
    jac = ops.stack([d_L0, d_A, d_alpha, d_k], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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
    Nb = N[None, :] ** b[:, None]
    logS = xp.log(S[None, :])
    numer = a[:, None] * Nb + c[:, None]
    denom = 1.0 + d[:, None] * logS
    denom = ops.clamp_min(denom, _EPS)
    pred = numer / denom

    # Jacobian: d/d[a, b, c, d]
    logN = xp.log(ops.clamp_min(N[None, :], _EPS))
    inv_denom = 1.0 / denom

    d_a = Nb * inv_denom                               # d(numer)/da = N^b
    d_b = a[:, None] * Nb * logN * inv_denom            # d(numer)/db = a * N^b * logN
    d_c = inv_denom                                     # d(numer)/dc = 1
    d_d = -numer * logS * inv_denom ** 2                # d(denom)/dd = logS → quotient rule
    jac = ops.stack([d_a, d_b, d_c, d_d], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_9 (3p): A * N^(-α) * S^(-β)
# theta: [A, alpha, beta]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    A, alpha, beta = [theta[:, i] for i in range(3)]
    Na = N[None, :] ** (-alpha[:, None])
    Sb = S[None, :] ** (-beta[:, None])
    pred = A[:, None] * Na * Sb

    # Jacobian: d/d[A, alpha, beta]
    logN = xp.log(ops.clamp_min(N[None, :], _EPS))
    logS = xp.log(ops.clamp_min(S[None, :], _EPS))

    d_A = Na * Sb                   # pred / A
    d_alpha = pred * (-logN)        # d(N^(-alpha))/dalpha = -N^(-alpha)*logN
    d_beta = pred * (-logS)
    jac = ops.stack([d_A, d_alpha, d_beta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_10 (4p): (A * N^(-α) + E) * S^(-β)
# theta: [A, alpha, E, beta]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    N = ops.clamp_min(X[:, 0], _EPS)
    S = ops.clamp_min(X[:, 1], _EPS)
    A, alpha, E, beta = [theta[:, i] for i in range(4)]
    Na = N[None, :] ** (-alpha[:, None])
    Sb = S[None, :] ** (-beta[:, None])
    bracket = A[:, None] * Na + E[:, None]  # (A*N^(-alpha) + E)
    pred = bracket * Sb

    # Jacobian: d/d[A, alpha, E, beta]
    logN = xp.log(ops.clamp_min(N[None, :], _EPS))
    logS = xp.log(ops.clamp_min(S[None, :], _EPS))

    d_A = Na * Sb
    # d(bracket)/dalpha = A * (-Na * logN), then * Sb
    d_alpha = A[:, None] * Na * (-logN) * Sb
    d_E = Sb
    # d(pred)/dbeta = bracket * d(Sb)/dbeta = bracket * (-Sb * logS) = -pred * logS
    d_beta = -pred * logS
    jac = ops.stack([d_A, d_alpha, d_E, d_beta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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
