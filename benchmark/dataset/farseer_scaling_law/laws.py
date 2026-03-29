"""Scaling laws for the Farseer N-D benchmark split.

X columns: [N (Model Size), D (Training Tokens)]

Primary law:
    L(N, D) = exp(s * N^q + S) + exp(B * N^b + Q) * D^(-exp(A * N^a + E))
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12
_EXP_CLIP = 50.0


# Farseer law (9 params):
#   exp(s * N^q + S) + exp(B * N^b + Q) * D^(-exp(A * N^a + E))
# theta: [E, s, q, S, B, b, Q, A, a]
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    N = ops.clamp_min(X[:, 0], _EPS)
    D = ops.clamp_min(X[:, 1], _EPS)

    E = theta[:, 0]
    s = theta[:, 1]
    q = theta[:, 2]
    S = theta[:, 3]
    B = theta[:, 4]
    b = theta[:, 5]
    Q = theta[:, 6]
    A = theta[:, 7]
    a = theta[:, 8]

    log_N = xp.log(ops.clamp_min(N, _EPS))
    log_D = xp.log(ops.clamp_min(D, _EPS))

    N_pow_q = N[None, :] ** q[:, None]
    N_pow_b = N[None, :] ** b[:, None]
    N_pow_a = N[None, :] ** a[:, None]

    term1_arg = ops.clamp(s[:, None] * N_pow_q + S[:, None], min=-_EXP_CLIP, max=_EXP_CLIP)
    bn_arg = ops.clamp(B[:, None] * N_pow_b + Q[:, None], min=-_EXP_CLIP, max=_EXP_CLIP)
    exp_an_arg = ops.clamp(A[:, None] * N_pow_a + E[:, None], min=-_EXP_CLIP, max=_EXP_CLIP)

    term1 = ops.exp(term1_arg)
    exp_an = ops.exp(exp_an_arg)
    an = -exp_an
    log_term2 = ops.clamp(bn_arg + an * log_D[None, :], min=-_EXP_CLIP, max=_EXP_CLIP)
    term2 = ops.exp(log_term2)
    pred = term1 + term2

    ones = xp.ones_like(pred)

    d_E = term2 * log_D[None, :] * an
    d_s = term1 * N_pow_q
    d_q = term1 * s[:, None] * N_pow_q * log_N[None, :]
    d_S = term1
    d_B = term2 * N_pow_b
    d_b = term2 * B[:, None] * N_pow_b * log_N[None, :]
    d_Q = term2
    d_A = term2 * log_D[None, :] * an * N_pow_a
    d_a = term2 * log_D[None, :] * an * A[:, None] * N_pow_a * log_N[None, :]

    jac = ops.stack([d_E, d_s, d_q, d_S, d_B, d_b, d_Q, d_A, d_a], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


LAW_REGISTRY = {"sl_1": sl_1}
PARAM_COUNTS = {"sl_1": 9}

# Data ranges based on the integrated benchmark split:
#   N ∈ [9.96e7, 2.51e10], D ∈ [1.0e9, 5.12e11], loss ∈ [0.438, 0.675]
PARAM_BOUNDS = {
    # Bounds centered around the paper / notebook ground-truth parameters,
    # but widened substantially to reduce prior pressure while preserving
    # the sign of the exponent terms: q > 0, b < 0, a > 0.
    # E=3.133347198805445, s=-0.062465473, q=0.13, S=0.1284880679442551,
    # B=230.73437075885855, b=-0.1729, Q=-1.544209554,
    # A=-1.665630816, a=0.0458999999999619
    "sl_1": [
        (1.0, 6.0),        # E
        (-1.0, 0.3),       # s
        (0.01, 0.5),       # q > 0
        (-1.0, 1.0),       # S
        (10.0, 1000.0),    # B
        (-0.6, -0.01),     # b < 0
        (-5.0, 1.0),       # Q
        (-5.0, 1.0),       # A
        (0.001, 0.25),     # a > 0
    ],
}
