from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


# Distillation scaling law (9 parameters):
#   L_S(L_T, N_S, D_S) = L_T + (1 / L_T^c0) * sigmoid_term * power_law_term
#
# where:
#   chinchilla_term = 1.220 + (3355/NS^0.408 + 18186/DS^0.431)^0.452
#   ratio = L_T / (chinchilla_term * d1)
#   sigmoid_term = (1 + ratio^(1/f1))^(-c1 * f1)
#   power_law_term = (A'/NS^alpha' + B'/DS^beta')^gamma'
#
# Fixed constants: 1.220, 3355, 0.408, 18186, 0.431, 0.452
# Free parameters: [c0, d1, f1, c1, A_prime, alpha_prime, B_prime, beta_prime, gamma_prime]
#
# theta: (B, 9), X: (M, 3) with columns [NS, DS, LT]
# Output: (B, M) or (M,) for single theta
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
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
    ) ** 0.452  # (1, M) broadcast to (B, M)

    # ratio = LT / (chinchilla_term * d1)
    ratio = LT[None, :] / ops.clamp_min(chinchilla_term * d1[:, None], _EPS)  # (B, M)

    # sigmoid-like term: S = (1 + ratio^(1/f1))^(-c1*f1)
    f1_safe = ops.clamp_min(f1[:, None], _EPS)  # (B, 1)
    ratio_safe = ops.clamp_min(ratio, _EPS)  # (B, M)
    u = ratio_safe ** (1.0 / f1_safe)  # ratio^(1/f1), (B, M)
    one_plus_u = 1.0 + u  # (B, M)
    S = one_plus_u ** (-c1[:, None] * f1_safe)  # sigmoid_term, (B, M)

    # Power law term: W = (A'/NS^alpha' + B'/DS^beta')^gamma'
    log_NS = xp.log(ops.clamp_min(NS, _EPS))  # (M,)
    log_DS = xp.log(ops.clamp_min(DS, _EPS))  # (M,)
    term_NS = A_prime[:, None] / ops.clamp_min(NS[None, :] ** alpha_prime[:, None], _EPS)  # (B, M)
    term_DS = B_prime[:, None] / ops.clamp_min(DS[None, :] ** beta_prime[:, None], _EPS)   # (B, M)
    inner = ops.clamp_min(term_NS + term_DS, _EPS)  # (B, M)
    W = inner ** gamma_prime[:, None]  # power_law, (B, M)

    # prefix = 1 / LT^c0
    log_LT = xp.log(ops.clamp_min(LT, _EPS))  # (M,)
    prefix = 1.0 / ops.clamp_min(LT[None, :] ** c0[:, None], _EPS)  # (B, M)

    # Full formula: pred = LT + prefix * S * W
    tail = prefix * S * W  # (B, M)
    pred = LT[None, :] + tail  # (B, M)

    # ---- Jacobian computation ----
    # tail = prefix * S * W
    # pred = LT + tail

    # Shared intermediates for sigmoid derivatives
    log_ratio = xp.log(ops.clamp_min(ratio_safe, _EPS))  # (B, M)
    log_one_plus_u = xp.log(ops.clamp_min(one_plus_u, _EPS))  # (B, M)
    log_inner = xp.log(ops.clamp_min(inner, _EPS))  # (B, M)

    # (1) d/d(c0): prefix = LT^(-c0), d(prefix)/d(c0) = -prefix * log(LT)
    d_c0 = -tail * log_LT[None, :]

    # (2) d/d(d1): affects ratio = LT/(chin*d1), d(ratio)/d(d1) = -ratio/d1
    # dS/d(ratio) = S * [-c1 * u / (ratio * (1+u))]   (via log-derivative)
    # d(pred)/d(d1) = prefix * W * dS/d(ratio) * (-ratio/d1)
    #              = tail * c1 * u / (d1 * (1+u))
    d_d1 = tail * c1[:, None] * u / (d1[:, None] * one_plus_u)

    # (3) d/d(f1): S = (1+u)^(-c1*f1), u = ratio^(1/f1)
    # d(log S)/d(f1) = -c1*log(1+u) + c1*u*log(ratio) / (f1*(1+u))
    # dS/d(f1) = S * [above]
    d_f1 = tail * (-c1[:, None] * log_one_plus_u
                    + c1[:, None] * u * log_ratio / (f1_safe * one_plus_u))

    # (4) d/d(c1): d(log S)/d(c1) = -f1*log(1+u)
    d_c1 = tail * (-f1_safe * log_one_plus_u)

    # (5) d/d(A_prime): W = inner^gamma', inner = term_NS + term_DS
    # dW/d(A') = gamma' * inner^(gamma'-1) * NS^(-alpha') = W * gamma' * NS^(-alpha') / inner
    NS_neg_alpha = term_NS / A_prime[:, None]  # NS^(-alpha'), safe since A_prime in numerator
    d_A_prime = prefix * S * W * gamma_prime[:, None] * NS_neg_alpha / inner

    # (6) d/d(alpha_prime): d(term_NS)/d(alpha') = -term_NS * log(NS)
    # dW/d(alpha') = W * gamma' * (-term_NS * log(NS)) / inner
    d_alpha_prime = prefix * S * W * gamma_prime[:, None] * (-term_NS * log_NS[None, :]) / inner

    # (7) d/d(B_prime): similar to A_prime
    DS_neg_beta = term_DS / B_prime[:, None]  # DS^(-beta')
    d_B_prime = prefix * S * W * gamma_prime[:, None] * DS_neg_beta / inner

    # (8) d/d(beta_prime): d(term_DS)/d(beta') = -term_DS * log(DS)
    d_beta_prime = prefix * S * W * gamma_prime[:, None] * (-term_DS * log_DS[None, :]) / inner

    # (9) d/d(gamma_prime): dW/d(gamma') = W * log(inner)
    d_gamma_prime = tail * log_inner

    jac = ops.stack([d_c0, d_d1, d_f1, d_c1, d_A_prime, d_alpha_prime,
                     d_B_prime, d_beta_prime, d_gamma_prime], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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
