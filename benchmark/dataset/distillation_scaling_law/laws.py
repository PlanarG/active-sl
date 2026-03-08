from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


# Distillation scaling law (9 parameters):
#   L_S(L_T, N_S, D_S) = L_T + (1 / L_T^c0) * sigmoid_term * power_law_term
#
# where:
#   chinchilla_term = 1.220 + (3355/N_S^0.408 + 18186/D_S^0.431)^0.452
#   ratio = L_T / (chinchilla_term * d1)
#   sigmoid_term = (1 + ratio^(1/f1))^(-c1 * f1)
#   power_law_term = (A'/N_S^alpha' + B'/D_S^beta')^gamma'
#
# Fixed constants: 1.220, 3355, 0.408, 18186, 0.431, 0.452
# Free parameters: [c0, d1, f1, c1, A_prime, alpha_prime, B_prime, beta_prime, gamma_prime]
#
# theta: (B, 9), X: (M, 3) with columns [NS, DS, LT]
# Output: (B, M) or (M,) for single theta
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    # X columns: [NS, DS, LT] (3 features)
    NS = ops.clamp_min(X[:, 0], _EPS)  # (M,)
    DS = ops.clamp_min(X[:, 1], _EPS)  # (M,)
    LT = ops.clamp_min(X[:, 2], _EPS)  # (M,)

    # theta: (B, 9)
    c0 = theta[:, 0]
    d1 = theta[:, 1]
    f1 = theta[:, 2]
    c1 = theta[:, 3]
    A_prime = theta[:, 4]
    alpha_prime = theta[:, 5]
    B_prime = theta[:, 6]
    beta_prime = theta[:, 7]
    gamma_prime = theta[:, 8]

    # Fixed constants from the paper
    # L_chinchilla(NS, DS) = 1.220 + (3355/NS^0.408 + 18186/DS^0.431)^0.452
    chinchilla_term = 1.220 + (
        3355.0 / (NS[None, :] ** 0.408) + 18186.0 / (DS[None, :] ** 0.431)
    ) ** 0.452

    # ratio = LT / (chinchilla_term * d1)
    ratio = LT[None, :] / ops.clamp_min(chinchilla_term * d1[:, None], _EPS)

    # sigmoid-like term: (1 + ratio^(1/f1))^(-c1*f1)
    f1_safe = ops.clamp_min(f1[:, None], _EPS)
    power_inner = ops.clamp_min(ratio, _EPS) ** (1.0 / f1_safe)
    sigmoid_term = (1.0 + power_inner) ** (-c1[:, None] * f1_safe)

    # Power law term: (A'/NS^alpha' + B'/DS^beta')^gamma'
    power_law = (
        A_prime[:, None] / ops.clamp_min(NS[None, :] ** alpha_prime[:, None], _EPS)
        + B_prime[:, None] / ops.clamp_min(DS[None, :] ** beta_prime[:, None], _EPS)
    )
    power_law = ops.clamp_min(power_law, _EPS) ** gamma_prime[:, None]

    # Full formula: LT + (1/LT^c0) * sigmoid_term * power_law
    pred = (
        LT[None, :]
        + (1.0 / ops.clamp_min(LT[None, :] ** c0[:, None], _EPS))
        * sigmoid_term
        * power_law
    )

    return pred[0] if pred.shape[0] == 1 else pred


# Parameter bounds for sl_1:
#   [c0, d1, f1, c1, A_prime, alpha_prime, B_prime, beta_prime, gamma_prime]
#
# Bound derivation:
#   - c0: exponent on LT (LT ~ 1.9-2.8), controls 1/LT^c0 prefactor.
#     Range (0, 5) covers from no effect to strong suppression.
#   - d1: scales chinchilla_term in denominator of ratio. chinchilla_term ~ 2-4.
#     d1 ~ 0.5-5 keeps ratio ~ O(1). Range (0.1, 10).
#   - f1: controls sharpness of sigmoid transition. (0.1, 10).
#   - c1: controls sigmoid slope/magnitude. (0, 5).
#   - A_prime: coefficient for N_S power-law term. NS ~ 1.4e8-7.75e9.
#     A'/NS^alpha' should be O(0.01-1), so A' up to ~1e6 with alpha'~0.3.
#     Range (1e-6, 1e10) for safety.
#   - alpha_prime: exponent on NS. (0.01, 2.0).
#   - B_prime: coefficient for D_S power-law term. DS ~ 3.7e9-5.1e11.
#     Similar reasoning. Range (1e-6, 1e10).
#   - beta_prime: exponent on DS. (0.01, 2.0).
#   - gamma_prime: outer exponent on power-law sum. (0.01, 3.0).
PARAM_BOUNDS = {
    "sl_1": [
        (0.0, 5.0),       # c0
        (0.1, 10.0),       # d1
        (0.1, 10.0),       # f1
        (0.0, 5.0),        # c1
        (1e-6, 1e10),      # A_prime
        (0.01, 2.0),       # alpha_prime
        (1e-6, 1e10),      # B_prime
        (0.01, 2.0),       # beta_prime
        (0.01, 3.0),       # gamma_prime
    ],
}

LAW_REGISTRY = {
    "sl_1": sl_1,
}
PARAM_COUNTS = {
    "sl_1": 9,
}
