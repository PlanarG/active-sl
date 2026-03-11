"""Scaling laws for MoE sparsity models.

X columns: [P (= N_total / N_active), N_active]

sl_1 (5 params): L = exp(d1) * P^(-a) * N_active^(-b) * exp(c * log(P) * log(N_active)) + exp(d3)
sl_2 (4 params): L = exp(d1) * P^(-a) * N_active^(-b) + exp(d3)
sl_3 (6 params): L = exp(d1) * P^(-a) + exp(d2) * N_active^(-b) * exp(c * log(P) * log(N_active)) + exp(d3)
sl_4 (5 params): L = exp(d1) * P^(-a) + exp(d2) * N_active^(-b) + exp(d3)
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


# sl_1 (5 params): [d1, a, b, c, d3]
# L = exp(d1) * P^(-a) * N_active^(-b) * exp(c * log(P) * log(N_active)) + exp(d3)
#   = exp(d1 - a*log(P) - b*log(N) + c*log(P)*log(N)) + exp(d3)
# Let term = exp(d1 - a*log_P - b*log_N + c*log_P*log_N)
# Let base = exp(d3)
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    P = ops.clamp_min(X[:, 0], _EPS)
    N_act = ops.clamp_min(X[:, 1], _EPS)

    d1 = theta[:, 0]
    a = theta[:, 1]
    b = theta[:, 2]
    c = theta[:, 3]
    d3 = theta[:, 4]

    log_P = xp.log(P)[None, :]   # (1, M)
    log_N = xp.log(N_act)[None, :]  # (1, M)

    term = (
        ops.exp(d1[:, None])
        * (P[None, :] ** (-a[:, None]))
        * (N_act[None, :] ** (-b[:, None]))
        * ops.exp(c[:, None] * log_P * log_N)
    )  # (B, M)
    base = ops.exp(d3[:, None])  # (B, M)
    pred = term + base  # (B, M)

    # Jacobian: (B, M, 5)
    # term = exp(d1 - a*log_P - b*log_N + c*log_P*log_N)
    # d(term)/d(d1) = term
    # d(term)/d(a)  = -term * log_P
    # d(term)/d(b)  = -term * log_N
    # d(term)/d(c)  = term * log_P * log_N
    # d(base)/d(d3) = base
    zeros = pred * 0.0

    d_d1 = term
    d_a = -term * log_P
    d_b = -term * log_N
    d_c = term * log_P * log_N
    d_d3 = zeros + base

    jac = ops.stack([d_d1, d_a, d_b, d_c, d_d3], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_2 (4 params): [d1, a, b, d3]
# L = exp(d1) * P^(-a) * N_active^(-b) + exp(d3)
#   = exp(d1 - a*log_P - b*log_N) + exp(d3)
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    P = ops.clamp_min(X[:, 0], _EPS)
    N_act = ops.clamp_min(X[:, 1], _EPS)

    d1 = theta[:, 0]
    a = theta[:, 1]
    b = theta[:, 2]
    d3 = theta[:, 3]

    log_P = xp.log(P)[None, :]   # (1, M)
    log_N = xp.log(N_act)[None, :]  # (1, M)

    term = (
        ops.exp(d1[:, None])
        * (P[None, :] ** (-a[:, None]))
        * (N_act[None, :] ** (-b[:, None]))
    )  # (B, M)
    base = ops.exp(d3[:, None])  # (B, M)
    pred = term + base  # (B, M)

    # Jacobian: (B, M, 4)
    # term = exp(d1 - a*log_P - b*log_N)
    zeros = pred * 0.0

    d_d1 = term
    d_a = -term * log_P
    d_b = -term * log_N
    d_d3 = zeros + base

    jac = ops.stack([d_d1, d_a, d_b, d_d3], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_3 (6 params): [d1, a, d2, b, c, d3]
# L = exp(d1) * P^(-a) + exp(d2) * N_active^(-b) * exp(c * log(P) * log(N_active)) + exp(d3)
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    P = ops.clamp_min(X[:, 0], _EPS)
    N_act = ops.clamp_min(X[:, 1], _EPS)

    d1 = theta[:, 0]
    a = theta[:, 1]
    d2 = theta[:, 2]
    b = theta[:, 3]
    c = theta[:, 4]
    d3 = theta[:, 5]

    log_P = xp.log(P)[None, :]   # (1, M)
    log_N = xp.log(N_act)[None, :]  # (1, M)

    term_P = ops.exp(d1[:, None]) * (P[None, :] ** (-a[:, None]))  # (B, M)
    term_N = (
        ops.exp(d2[:, None])
        * (N_act[None, :] ** (-b[:, None]))
        * ops.exp(c[:, None] * log_P * log_N)
    )  # (B, M)
    base = ops.exp(d3[:, None])  # (B, M)
    pred = term_P + term_N + base  # (B, M)

    # Jacobian: (B, M, 6)
    # term_P = exp(d1 - a*log_P) => d/d(d1) = term_P, d/d(a) = -term_P*log_P
    # term_N = exp(d2 - b*log_N + c*log_P*log_N) => d/d(d2)=term_N, d/d(b)=-term_N*log_N, d/d(c)=term_N*log_P*log_N
    # base = exp(d3) => d/d(d3) = base
    zeros = pred * 0.0

    d_d1 = zeros + term_P
    d_a = -term_P * log_P
    d_d2 = zeros + term_N
    d_b = -term_N * log_N
    d_c = term_N * log_P * log_N
    d_d3 = zeros + base

    jac = ops.stack([d_d1, d_a, d_d2, d_b, d_c, d_d3], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_4 (5 params): [d1, a, d2, b, d3]
# L = exp(d1) * P^(-a) + exp(d2) * N_active^(-b) + exp(d3)
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    P = ops.clamp_min(X[:, 0], _EPS)
    N_act = ops.clamp_min(X[:, 1], _EPS)

    d1 = theta[:, 0]
    a = theta[:, 1]
    d2 = theta[:, 2]
    b = theta[:, 3]
    d3 = theta[:, 4]

    log_P = xp.log(P)[None, :]   # (1, M)
    log_N = xp.log(N_act)[None, :]  # (1, M)

    term_P = ops.exp(d1[:, None]) * (P[None, :] ** (-a[:, None]))  # (B, M)
    term_N = ops.exp(d2[:, None]) * (N_act[None, :] ** (-b[:, None]))  # (B, M)
    base = ops.exp(d3[:, None])  # (B, M)
    pred = term_P + term_N + base  # (B, M)

    # Jacobian: (B, M, 5)
    zeros = pred * 0.0

    d_d1 = zeros + term_P
    d_a = -term_P * log_P
    d_d2 = zeros + term_N
    d_b = -term_N * log_N
    d_d3 = zeros + base

    jac = ops.stack([d_d1, d_a, d_d2, d_b, d_d3], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


LAW_REGISTRY = {"sl_1": sl_1, "sl_2": sl_2, "sl_3": sl_3, "sl_4": sl_4}
PARAM_COUNTS = {"sl_1": 5, "sl_2": 4, "sl_3": 6, "sl_4": 5}

# Data ranges:
#   P in [1.86, 12.25], N_active in [0.015, 1.90], loss in [2.07, 3.40]
# d1, d2, d3: inside exp(), so reasonable range is (-5, 5)
# a, b: positive exponents, (0.01, 3.0)
# c: cross-term coefficient, (-1, 1)
PARAM_BOUNDS = {
    "sl_1": [(-5.0, 5.0), (0.01, 3.0), (0.01, 3.0), (-1.0, 1.0), (-5.0, 5.0)],
    "sl_2": [(-5.0, 5.0), (0.01, 3.0), (0.01, 3.0), (-5.0, 5.0)],
    "sl_3": [(-5.0, 5.0), (0.01, 3.0), (-5.0, 5.0), (0.01, 3.0), (-1.0, 1.0), (-5.0, 5.0)],
    "sl_4": [(-5.0, 5.0), (0.01, 3.0), (-5.0, 5.0), (0.01, 3.0), (-5.0, 5.0)],
}
