"""Scaling laws for Chinchilla-style compute-optimal models.

X columns: [N (Model Size), D (Training Tokens)]

Law: L(N, D) = E + A / N^alpha + B / D^beta
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


# Scaling law 1 (5 params):
#   E + A / N^alpha + B / D^beta
# theta: [E, A, alpha, B, beta]
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    N = ops.clamp_min(X[:, 0], _EPS)
    D = ops.clamp_min(X[:, 1], _EPS)

    E = theta[:, 0]
    A = theta[:, 1]
    alpha = theta[:, 2]
    B = theta[:, 3]
    beta = theta[:, 4]

    term_N = A[:, None] / ops.clamp_min(N[None, :] ** alpha[:, None], _EPS)
    term_D = B[:, None] / ops.clamp_min(D[None, :] ** beta[:, None], _EPS)

    pred = E[:, None] + term_N + term_D
    return pred[0] if pred.shape[0] == 1 else pred


LAW_REGISTRY = {"sl_1": sl_1}
PARAM_COUNTS = {"sl_1": 5}

# Data ranges:
#   N ∈ [5.7e7, 1.6e10], D ∈ [2.5e8, 3.2e11], loss ∈ [2.08, 5.01]
# Chinchilla paper fit: E≈1.69, A≈406.4, alpha≈0.34, B≈410.7, beta≈0.28
PARAM_BOUNDS = {
    # E (irreducible loss floor): [0, 5] — loss range is ~2-5
    # A (N coefficient): [0, 1e4] — large because N^alpha can be ~1e3
    # alpha (N exponent): [0.01, 2.0]
    # B (D coefficient): [0, 1e4] — large because D^beta can be ~1e3
    # beta (D exponent): [0.01, 2.0]
    "sl_1": [(0, 5), (0, 1e4), (0.01, 2.0), (0, 1e4), (0.01, 2.0)],
}
