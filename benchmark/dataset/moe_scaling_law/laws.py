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
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    L_inf = theta[:, 0]
    B = theta[:, 1]
    alpha = theta[:, 2]
    beta = theta[:, 3]

    logN = xp.log(ops.clamp_min(N, _EPS))
    logE = xp.log(ops.clamp_min(E, _EPS))

    denom = (N[None, :] ** alpha[:, None]) * (E[None, :] ** beta[:, None])
    denom = ops.clamp_min(denom, _EPS)

    frac = B[:, None] / denom  # B / (N^alpha * E^beta)
    pred = L_inf[:, None] + frac

    ones = pred * 0.0 + 1.0
    # d/d(L_inf) = 1
    d_L_inf = ones
    # d/d(B) = 1 / denom
    d_B = frac / B[:, None]  # = 1/denom, but reuse frac
    # d/d(alpha) = -B / denom * log(N) = -frac * log(N)
    d_alpha = -frac * logN[None, :]
    # d/d(beta) = -frac * log(E)
    d_beta = -frac * logE[None, :]

    jac = ops.stack([d_L_inf, d_B, d_alpha, d_beta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 2 (5 params):
#   L + K * (N^alpha * E^beta)^(-gamma)
# theta: [L, K, alpha, beta, gamma]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    L = theta[:, 0]
    K = theta[:, 1]
    alpha = theta[:, 2]
    beta = theta[:, 3]
    gamma = theta[:, 4]

    logN = xp.log(ops.clamp_min(N, _EPS))
    logE = xp.log(ops.clamp_min(E, _EPS))

    base = (N[None, :] ** alpha[:, None]) * (E[None, :] ** beta[:, None])
    base = ops.clamp_min(base, _EPS)

    power_term = base ** (-gamma[:, None])  # (N^a * E^b)^(-g)
    term = K[:, None] * power_term
    pred = L[:, None] + term

    ones = pred * 0.0 + 1.0
    # log(base) = alpha*log(N) + beta*log(E)
    log_base = alpha[:, None] * logN[None, :] + beta[:, None] * logE[None, :]

    # d/dL = 1
    d_L = ones
    # d/dK = power_term
    d_K = power_term
    # d/d(alpha) = K * power_term * (-gamma) * log(N) = term * (-gamma) * log(N)
    d_alpha = term * (-gamma[:, None]) * logN[None, :]
    # d/d(beta) = term * (-gamma) * log(E)
    d_beta = term * (-gamma[:, None]) * logE[None, :]
    # d/d(gamma) = K * power_term * (-log(base)) = term * (-log_base)
    d_gamma = term * (-log_base)

    jac = ops.stack([d_L, d_K, d_alpha, d_beta, d_gamma], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 3 (6 params):
#   A * P^alpha / (1 + B * E^beta) + C * P^(alpha*0.6) + D
#   (gamma = alpha * 0.6 is hard-coded)
# theta: [A, alpha, B, beta, C, D]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
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

    logN = xp.log(ops.clamp_min(N, _EPS))
    logE = xp.log(ops.clamp_min(E, _EPS))

    N_alpha = N[None, :] ** alpha[:, None]          # (B_t, M)
    E_beta = E[None, :] ** beta[:, None]             # (B_t, M)
    denom = 1.0 + B[:, None] * E_beta               # (B_t, M)
    efficiency = A[:, None] * N_alpha / denom        # term1

    gamma_val = alpha[:, None] * 0.6
    N_gamma = N[None, :] ** gamma_val                # (B_t, M)
    param_scale = C[:, None] * N_gamma               # term2

    pred = efficiency + param_scale + D[:, None]

    ones = pred * 0.0 + 1.0

    # d/dA = N^alpha / denom
    d_A = N_alpha / denom
    # d/d(alpha):
    #   d(efficiency)/d(alpha) = A * N^alpha * log(N) / denom = efficiency * log(N)
    #   d(param_scale)/d(alpha) = C * N^(alpha*0.6) * 0.6 * log(N) = param_scale * 0.6 * log(N)
    d_alpha = efficiency * logN[None, :] + param_scale * 0.6 * logN[None, :]
    # d/dB = -A * N^alpha * E^beta / denom^2 = -efficiency * E^beta / denom
    d_B = -efficiency * E_beta / denom
    # d/d(beta) = -A * N^alpha * B * E^beta * log(E) / denom^2
    #           = -efficiency * B[:, None] * E_beta * log(E) / denom
    d_beta = -efficiency * B[:, None] * E_beta * logE[None, :] / denom
    # d/dC = N^(alpha*0.6)
    d_C = N_gamma
    # d/dD = 1
    d_D = ones

    jac = ops.stack([d_A, d_alpha, d_B, d_beta, d_C, d_D], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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

    logN = xp.log(N)
    log1E = xp.log(1.0 + E)

    N_alpha = N[None, :] ** alpha[:, None]
    bE_term = 1.0 + b[:, None] * E[None, :]         # (B_t, M)
    bE_term_safe = ops.clamp_min(bE_term, _EPS)
    expert_sat = bE_term_safe ** gamma[:, None]
    expert_sat = ops.clamp_min(expert_sat, _EPS)

    main = a[:, None] / (N_alpha * expert_sat)       # a / (N^alpha * (1+bE)^gamma)

    log_correction = d[:, None] * (logN[None, :] - 0.4 * log1E[None, :])

    pred = main + c[:, None] + log_correction

    ones = pred * 0.0 + 1.0

    # d/da = 1 / (N^alpha * expert_sat) = main / a[:, None]
    d_a = main / a[:, None]
    # d/d(alpha) = -main * log(N)
    d_alpha = -main * logN[None, :]
    # d/db = -a * gamma * E * (1+bE)^(gamma-1) / (N^alpha * (1+bE)^(2*gamma))
    #       = -main * gamma * E / (1+bE)
    #   Since main = a / (N^a * (1+bE)^g), and d/db of (1+bE)^g = g*E*(1+bE)^(g-1)
    #   d(main)/db = -a * g * E * (1+bE)^(g-1) / (N^a * ((1+bE)^g)^2)
    #   but (1+bE)^(g-1) / ((1+bE)^g)^2 = 1/((1+bE)^(g+1))
    #   Simpler: main = a * (N^alpha * (1+bE)^gamma)^(-1)
    #   d/db = -main * gamma * E / (1+bE)
    d_b = -main * gamma[:, None] * E[None, :] / bE_term_safe
    # d/d(gamma) = -main * log(1+bE)
    log_bE_term = xp.log(ops.clamp_min(bE_term_safe, _EPS))
    d_gamma = -main * log_bE_term
    # d/dc = 1
    d_c = ones
    # d/dd = log(N) - 0.4*log(1+E)
    d_d = logN[None, :] - 0.4 * log1E[None, :]
    # broadcast d_d to (B_t, M)
    d_d = d_d + ones * 0.0

    jac = ops.stack([d_a, d_alpha, d_b, d_gamma, d_c, d_d], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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

    exp_val = ops.exp(exponent)   # (B_t, M)

    pred = p0[:, None] + exp_val + p5[:, None] * log_E

    ones = pred * 0.0 + 1.0

    # d/d(p0) = 1
    d_p0 = ones
    # d/d(p1) = exp_val * 1 = exp_val
    d_p1 = exp_val
    # d/d(p2) = exp_val * log_E
    d_p2 = exp_val * log_E
    # d/d(p3) = exp_val * log_N
    d_p3 = exp_val * log_N
    # d/d(p4) = exp_val * log_E * log_N
    d_p4 = exp_val * log_E * log_N
    # d/d(p5) = log_E
    d_p5 = log_E + ones * 0.0  # broadcast to (B_t, M)

    jac = ops.stack([d_p0, d_p1, d_p2, d_p3, d_p4, d_p5], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 6 (6 params):
#   a * N^(-b) * (1 + c*E^(-d)) + e + f/(E * N^0.05)
# theta: [a, b, c, d, e, f]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
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

    logN = xp.log(ops.clamp_min(N, _EPS))
    logE = xp.log(ops.clamp_min(E, _EPS))

    N_neg_b = N[None, :] ** (-b[:, None])            # (B_t, M)
    E_neg_d = E[None, :] ** (-d[:, None])             # (B_t, M)
    expert_mod = 1.0 + c[:, None] * E_neg_d           # (B_t, M)
    base = a[:, None] * N_neg_b                       # a * N^(-b)
    term1 = base * expert_mod                         # a * N^(-b) * (1 + c*E^(-d))
    interaction = f[:, None] / (E[None, :] * (N[None, :] ** 0.05))  # f/(E*N^0.05)

    pred = term1 + e[:, None] + interaction

    ones = pred * 0.0 + 1.0

    # d/da = N^(-b) * expert_mod
    d_a = N_neg_b * expert_mod
    # d/db = a * N^(-b) * (-log(N)) * expert_mod = -term1 * log(N)
    d_b = -term1 * logN[None, :]
    # d/dc = a * N^(-b) * E^(-d) = base * E_neg_d
    d_c = base * E_neg_d
    # d/dd = a * N^(-b) * c * E^(-d) * (-log(E)) = -base * c[:, None] * E_neg_d * log(E)
    d_d = -base * c[:, None] * E_neg_d * logE[None, :]
    # d/de = 1
    d_e = ones
    # d/df = 1 / (E * N^0.05)
    d_f = 1.0 / (E[None, :] * (N[None, :] ** 0.05))
    d_f = d_f + ones * 0.0  # ensure broadcast

    jac = ops.stack([d_a, d_b, d_c, d_d, d_e, d_f], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 7 (6 params):
#   p0 * E^p1 * P^p2 + p3 * P^p4 + p5
#   (multiplicative + additive power law)
# theta: [p0, p1, p2, p3, p4, p5]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
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

    logE = xp.log(ops.clamp_min(E, _EPS))
    logN = xp.log(ops.clamp_min(N, _EPS))

    E_p1 = E[None, :] ** p1[:, None]
    N_p2 = N[None, :] ** p2[:, None]
    N_p4 = N[None, :] ** p4[:, None]

    term1 = p0[:, None] * E_p1 * N_p2          # p0 * E^p1 * N^p2
    term2 = p3[:, None] * N_p4                  # p3 * N^p4

    pred = term1 + term2 + p5[:, None]

    ones = pred * 0.0 + 1.0

    # d/d(p0) = E^p1 * N^p2
    d_p0 = E_p1 * N_p2
    # d/d(p1) = term1 * log(E)
    d_p1 = term1 * logE[None, :]
    # d/d(p2) = term1 * log(N)
    d_p2 = term1 * logN[None, :]
    # d/d(p3) = N^p4
    d_p3 = N_p4
    # d/d(p4) = term2 * log(N)
    d_p4 = term2 * logN[None, :]
    # d/d(p5) = 1
    d_p5 = ones

    jac = ops.stack([d_p0, d_p1, d_p2, d_p3, d_p4, d_p5], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 8 (4 params):
#   a * N^b * E^c + d
# theta: [a, b, c, d]
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    a = theta[:, 0]
    b = theta[:, 1]
    c = theta[:, 2]
    d = theta[:, 3]

    logN = xp.log(ops.clamp_min(N, _EPS))
    logE = xp.log(ops.clamp_min(E, _EPS))

    N_b = N[None, :] ** b[:, None]
    E_c = E[None, :] ** c[:, None]
    term = a[:, None] * N_b * E_c               # a * N^b * E^c

    pred = term + d[:, None]

    ones = pred * 0.0 + 1.0

    # d/da = N^b * E^c
    d_a = N_b * E_c
    # d/db = term * log(N)
    d_b = term * logN[None, :]
    # d/dc = term * log(E)
    d_c = term * logE[None, :]
    # d/dd = 1
    d_d = ones

    jac = ops.stack([d_a, d_b, d_c, d_d], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 9 (4 params):
#   c0 + A * (N * E^g)^(-a)
# theta: [c0, A, g, a]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    E = ops.clamp_min(X[:, 0], _EPS)
    N = ops.clamp_min(X[:, 1], _EPS)

    c0 = theta[:, 0]
    A = theta[:, 1]
    g = theta[:, 2]
    a = theta[:, 3]

    logN = xp.log(ops.clamp_min(N, _EPS))
    logE = xp.log(ops.clamp_min(E, _EPS))

    N_eff = N[None, :] * (E[None, :] ** g[:, None])
    N_eff = ops.clamp_min(N_eff, _EPS)

    power_term = N_eff ** (-a[:, None])          # (N*E^g)^(-a)
    term = A[:, None] * power_term

    pred = c0[:, None] + term

    ones = pred * 0.0 + 1.0

    # log(N_eff) = log(N) + g*log(E)
    log_N_eff = logN[None, :] + g[:, None] * logE[None, :]

    # d/d(c0) = 1
    d_c0 = ones
    # d/d(A) = power_term
    d_A = power_term
    # d/d(g) = A * power_term * (-a) * log(E) = term * (-a) * log(E)
    #   since d/dg of N_eff^(-a) = (-a) * N_eff^(-a) * d(log N_eff)/dg
    #   and d(log N_eff)/dg = log(E)
    d_g = term * (-a[:, None]) * logE[None, :]
    # d/d(a) = A * power_term * (-log(N_eff)) = term * (-log_N_eff)
    d_a = term * (-log_N_eff)

    jac = ops.stack([d_c0, d_A, d_g, d_a], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 10 (6 params):
#   bias + A * (N/1e9)^(-alpha) * ((1 + B*E^gamma) / (1 + B))^(-beta)
# theta: [bias, A, alpha, B, gamma, beta]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
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

    logE = xp.log(ops.clamp_min(E, _EPS))

    N_scaled = N[None, :] / 1e9
    N_scaled = ops.clamp_min(N_scaled, _EPS)
    logN_scaled = xp.log(ops.clamp_min(N_scaled, _EPS))

    term_N = N_scaled ** (-alpha[:, None])           # (N/1e9)^(-alpha)

    E_gamma = E[None, :] ** gamma[:, None]
    expert_num = 1.0 + B[:, None] * E_gamma          # 1 + B*E^gamma
    expert_den = ops.clamp_min(1.0 + B[:, None], _EPS)  # 1 + B
    ratio = expert_num / expert_den                   # (1 + B*E^g) / (1 + B)
    ratio_safe = ops.clamp_min(ratio, _EPS)
    term_E = ratio_safe ** (-beta[:, None])           # ratio^(-beta)

    full_term = A[:, None] * term_N * term_E          # A * term_N * term_E
    pred = bias[:, None] + full_term

    ones = pred * 0.0 + 1.0

    # d/d(bias) = 1
    d_bias = ones
    # d/d(A) = term_N * term_E
    d_A = term_N * term_E
    # d/d(alpha) = full_term * (-log(N_scaled))
    d_alpha = full_term * (-logN_scaled)
    # d/d(B):
    #   ratio = (1 + B*E^g) / (1 + B)
    #   d(ratio)/dB = (E^g * (1+B) - (1+B*E^g)) / (1+B)^2
    #               = (E^g - 1) / (1+B)^2
    #   d(term_E)/dB = (-beta) * ratio^(-beta-1) * d(ratio)/dB
    #   d(full_term)/dB = A * term_N * d(term_E)/dB
    #                   = full_term * (-beta) / ratio * (E^g - 1) / (1+B)^2
    #   But ratio = expert_num / expert_den, so 1/ratio = expert_den / expert_num
    #   = full_term * (-beta) * (E^g - 1) / (expert_num * (1+B))
    #   Alternatively:  full_term * (-beta) * (E_gamma - 1.0) / (expert_num * expert_den)
    #   Wait let me redo: expert_den = 1+B
    #   d(ratio)/dB = (E^g - 1) / expert_den^2
    #   d(term_E)/dB = (-beta) * ratio^(-beta-1) * (E^g - 1) / expert_den^2
    #   = (-beta) * ratio^(-beta) * (1/ratio) * (E^g - 1) / expert_den^2
    #   = (-beta) * term_E * (expert_den / expert_num) * (E^g - 1) / expert_den^2
    #   = (-beta) * term_E * (E^g - 1) / (expert_num * expert_den)
    #   So: d_B = A * term_N * (-beta) * term_E * (E^g - 1) / (expert_num * expert_den)
    #          = full_term * (-beta) * (E_gamma - 1.0) / (expert_num * expert_den)
    d_B = full_term * (-beta[:, None]) * (E_gamma - 1.0) / (ops.clamp_min(expert_num, _EPS) * expert_den)

    # d/d(gamma):
    #   d(ratio)/d(gamma) = B * E^g * log(E) / (1+B)
    #   d(term_E)/d(gamma) = (-beta) * ratio^(-beta-1) * B * E^g * log(E) / expert_den
    #   = (-beta) * term_E * (1/ratio) * B * E^g * log(E) / expert_den
    #   = (-beta) * term_E * expert_den / expert_num * B * E^g * log(E) / expert_den
    #   = (-beta) * term_E * B * E^g * log(E) / expert_num
    #   d_gamma = full_term * (-beta) * B * E^g * log(E) / expert_num
    d_gamma = full_term * (-beta[:, None]) * B[:, None] * E_gamma * logE[None, :] / ops.clamp_min(expert_num, _EPS)

    # d/d(beta) = A * term_N * d(term_E)/d(beta)
    #   term_E = ratio^(-beta)
    #   d/d(beta) = ratio^(-beta) * (-log(ratio)) = term_E * (-log(ratio))
    #   d_beta = full_term * (-log(ratio))
    log_ratio = xp.log(ops.clamp_min(ratio_safe, _EPS))
    d_beta = full_term * (-log_ratio)

    jac = ops.stack([d_bias, d_A, d_alpha, d_B, d_gamma, d_beta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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
