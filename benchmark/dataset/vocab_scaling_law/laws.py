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
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    c0, A, b, e, g = [theta[:, i] for i in range(5)]

    logP = xp.log(ops.clamp_min(P, _EPS))
    logV = xp.log(ops.clamp_min(V, _EPS))
    logD = xp.log(ops.clamp_min(D, _EPS))

    V_b = V[None, :] ** b[:, None]
    P_e = P[None, :] ** e[:, None]
    D_g = D[None, :] ** g[:, None]
    term = A[:, None] * V_b * P_e * D_g  # A * V^b * P^e * D^g

    pred = c0[:, None] + term

    ones = pred * 0.0 + 1.0

    # d/d(c0) = 1
    d_c0 = ones
    # d/d(A) = V^b * P^e * D^g
    d_A = V_b * P_e * D_g
    # d/d(b) = term * log(V)
    d_b = term * logV[None, :]
    # d/d(e) = term * log(P)
    d_e = term * logP[None, :]
    # d/d(g) = term * log(D)
    d_g = term * logD[None, :]

    jac = ops.stack([d_c0, d_A, d_b, d_e, d_g], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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

    logP = xp.log(ops.clamp_min(P, _EPS))
    logV = xp.log(ops.clamp_min(V, _EPS))
    logD = xp.log(ops.clamp_min(D, _EPS))

    term_p = P[None, :] ** (-alpha[:, None])           # P^(-alpha)
    term_d = D[None, :] ** (-beta[:, None])             # D^(-beta)
    r_safe = ops.clamp_min(r, _EPS)

    # S = (x^r + y^r) / 2 where x=term_p, y=term_d
    tp_r = term_p ** r_safe[:, None]                    # P^(-alpha*r)
    td_r = term_d ** r_safe[:, None]                    # D^(-beta*r)
    S = (tp_r + td_r) / 2.0
    S_safe = ops.clamp_min(S, _EPS)
    inv_r = 1.0 / r_safe[:, None]
    mean_r = S_safe ** inv_r                            # ((x^r+y^r)/2)^(1/r)

    lV_diff = xp.log(V[None, :]) - v0[:, None]         # log(V) - v0
    vocab_gate = 1.0 + C[:, None] * lV_diff ** 2       # 1 + C*(logV - v0)^2

    product = mean_r * vocab_gate                       # mean_r * vocab_gate
    pred = L[:, None] + A[:, None] * product

    ones = pred * 0.0 + 1.0

    # Shared intermediates:
    # d(mean_r)/dS = (1/r) * S^(1/r - 1) = mean_r / (r * S)
    dmr_dS = mean_r / (r_safe[:, None] * S_safe)

    # --- d/dL = 1
    d_L = ones
    # --- d/dA = product
    d_A = product
    # --- d/d(alpha):
    #   d(tp_r)/d(alpha) = tp_r * r * (-log(P)) = -r * tp_r * logP   (since tp_r = P^(-alpha*r))
    #   d(S)/d(alpha) = (-r * tp_r * logP) / 2
    #   d(mean_r)/d(alpha) = dmr_dS * d(S)/d(alpha) = dmr_dS * (-r * tp_r * logP) / 2
    #                      = mean_r / (r * S) * (-r * tp_r * logP / 2)
    #                      = -mean_r * tp_r * logP / (2 * S)
    d_alpha = A[:, None] * vocab_gate * (-mean_r * tp_r * logP[None, :] / (2.0 * S_safe))

    # --- d/d(beta):
    #   d(td_r)/d(beta) = -r * td_r * logD
    #   d(S)/d(beta) = -r * td_r * logD / 2
    #   d(mean_r)/d(beta) = -mean_r * td_r * logD / (2 * S)
    d_beta = A[:, None] * vocab_gate * (-mean_r * td_r * logD[None, :] / (2.0 * S_safe))

    # --- d/d(C) = A * mean_r * (logV - v0)^2
    d_C = A[:, None] * mean_r * lV_diff ** 2

    # --- d/d(v0):
    #   d(vocab_gate)/d(v0) = C * 2 * (logV - v0) * (-1) = -2*C*(logV - v0)
    d_v0 = A[:, None] * mean_r * (-2.0 * C[:, None] * lV_diff)

    # --- d/d(r):
    #   mean_r = S^(1/r)
    #   log(mean_r) = (1/r)*log(S)
    #   d(mean_r)/d(r) = mean_r * d/dr[(1/r)*log(S)]
    #   = mean_r * [(-1/r^2)*log(S) + (1/r)*(1/S)*dS/dr]
    #
    #   dS/dr = d/dr [(tp_r + td_r)/2]
    #   d(tp_r)/dr = tp_r * log(term_p)   (since tp_r = term_p^r)
    #   d(td_r)/dr = td_r * log(term_d)
    #   dS/dr = (tp_r * log(term_p) + td_r * log(term_d)) / 2
    #
    #   But log(term_p) = -alpha * log(P), log(term_d) = -beta * log(D)
    log_term_p = -alpha[:, None] * logP[None, :]
    log_term_d = -beta[:, None] * logD[None, :]
    log_tp_r = r_safe[:, None] * log_term_p             # = log(tp_r)
    log_td_r = r_safe[:, None] * log_term_d             # = log(td_r)
    # But safer to use tp_r * log_term_p directly
    dS_dr = (tp_r * log_term_p + td_r * log_term_d) / 2.0
    log_S = xp.log(ops.clamp_min(S_safe, _EPS))
    dmr_dr = mean_r * ((-1.0 / (r_safe[:, None] ** 2)) * log_S + (1.0 / (r_safe[:, None] * S_safe)) * dS_dr)

    d_r = A[:, None] * vocab_gate * dmr_dr

    jac = ops.stack([d_L, d_A, d_alpha, d_beta, d_C, d_v0, d_r], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_3 (7p): L0 + ((a * P^-alpha)^q + (b * (D * V^phi)^-beta)^q)^(1/q)
# Generalized q-mean of two Chinchilla-style terms
# theta: [L0, a, alpha, b, beta, phi, q]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    L0, a, alpha, b, beta, phi, q = [theta[:, i] for i in range(7)]

    logP = xp.log(ops.clamp_min(P, _EPS))
    logV = xp.log(ops.clamp_min(V, _EPS))
    logD = xp.log(ops.clamp_min(D, _EPS))

    t1 = a[:, None] * (P[None, :] ** (-alpha[:, None]))     # a * P^(-alpha)
    eff_D = D[None, :] * (V[None, :] ** phi[:, None])
    eff_D = ops.clamp_min(eff_D, _EPS)
    log_eff_D = logD[None, :] + phi[:, None] * logV[None, :]
    t2 = b[:, None] * (eff_D ** (-beta[:, None]))            # b * (D*V^phi)^(-beta)

    q_safe = ops.clamp_min(q, _EPS)
    t1_safe = ops.clamp_min(t1, _EPS)
    t2_safe = ops.clamp_min(t2, _EPS)
    t1_q = t1_safe ** q_safe[:, None]
    t2_q = t2_safe ** q_safe[:, None]
    S = t1_q + t2_q
    S_safe = ops.clamp_min(S, _EPS)
    inv_q = 1.0 / q_safe[:, None]
    combined = S_safe ** inv_q                                # (t1^q + t2^q)^(1/q)

    pred = L0[:, None] + combined

    ones = pred * 0.0 + 1.0

    # Shared: d(combined)/d(S) = (1/q) * S^(1/q - 1) = combined / (q * S)
    dc_dS = combined / (q_safe[:, None] * S_safe)

    # For chain rule: d(t1_q)/d(theta) = q * t1^(q-1) * d(t1)/d(theta) = t1_q * q / t1 * d(t1)/d(theta)
    # But simpler: d(t1_q)/d(theta) = t1_q * q * d(log t1)/d(theta)
    # Similarly for t2_q

    # --- d/d(L0) = 1
    d_L0 = ones

    # --- d/d(a):
    #   d(t1)/d(a) = P^(-alpha) = t1 / a
    #   d(t1_q)/d(a) = q * t1^(q-1) * t1/a = t1_q * q / a
    #   d(combined)/d(a) = dc_dS * t1_q * q / a = combined / (q * S) * t1_q * q / a
    #                    = combined * t1_q / (S * a)
    d_a = dc_dS * t1_q * q_safe[:, None] / a[:, None]

    # --- d/d(alpha):
    #   d(t1)/d(alpha) = a * P^(-alpha) * (-log(P)) = -t1 * log(P)
    #   d(t1_q)/d(alpha) = q * t1^(q-1) * (-t1 * logP) = -t1_q * q * logP
    #   d(combined)/d(alpha) = dc_dS * (-t1_q * q * logP) = -combined * t1_q * logP / S
    d_alpha = dc_dS * (-t1_q * q_safe[:, None] * logP[None, :])

    # --- d/d(b):
    #   d(t2)/d(b) = eff_D^(-beta) = t2/b
    #   d(t2_q)/d(b) = t2_q * q / b
    d_b = dc_dS * t2_q * q_safe[:, None] / b[:, None]

    # --- d/d(beta):
    #   d(t2)/d(beta) = b * (-log(eff_D)) * eff_D^(-beta) = -t2 * log(eff_D)
    #   d(t2_q)/d(beta) = -t2_q * q * log(eff_D)
    d_beta = dc_dS * (-t2_q * q_safe[:, None] * log_eff_D)

    # --- d/d(phi):
    #   d(eff_D)/d(phi) = D * V^phi * log(V) = eff_D * log(V)
    #   d(t2)/d(phi) = b * (-beta) * eff_D^(-beta-1) * eff_D * log(V)
    #                = -beta * t2 * log(V)
    #   d(t2_q)/d(phi) = -t2_q * q * beta * log(V)
    d_phi = dc_dS * (-t2_q * q_safe[:, None] * beta[:, None] * logV[None, :])

    # --- d/d(q):
    #   combined = S^(1/q)
    #   log(combined) = (1/q) * log(S)
    #   d(combined)/d(q) = combined * d/dq[(1/q)*log(S)]
    #   = combined * [(-1/q^2)*log(S) + (1/q)*(1/S)*dS/dq]
    #
    #   dS/dq = d(t1_q)/dq + d(t2_q)/dq
    #   d(t1_q)/dq = t1_q * log(t1)   (since t1_q = t1^q)
    #   d(t2_q)/dq = t2_q * log(t2)
    log_t1 = xp.log(ops.clamp_min(t1_safe, _EPS))
    log_t2 = xp.log(ops.clamp_min(t2_safe, _EPS))
    dS_dq = t1_q * log_t1 + t2_q * log_t2
    log_S = xp.log(ops.clamp_min(S_safe, _EPS))
    d_q = combined * ((-1.0 / (q_safe[:, None] ** 2)) * log_S + (1.0 / (q_safe[:, None] * S_safe)) * dS_dq)

    jac = ops.stack([d_L0, d_a, d_alpha, d_b, d_beta, d_phi, d_q], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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
    L_inf, A_p, a, b, d_p, lam, g = [theta[:, i] for i in range(7)]

    logP = xp.log(P)
    logD = xp.log(D)
    logV = xp.log(V)

    log_t1 = a[:, None] * logP[None, :]                           # a*log(P)
    log_lam = xp.log(ops.clamp_min(lam, _EPS))
    log_t2 = log_lam[:, None] + b[:, None] * logD[None, :]        # log(lam) + b*log(D)
    log_max = ops.maximum(log_t1, log_t2)                          # max(log_t1, log_t2)

    # exponent_val = -d * log_max - g * log(V)
    exponent_val = -d_p[:, None] * log_max - g[:, None] * logV[None, :]
    exp_val = ops.exp(exponent_val)
    term = A_p[:, None] * exp_val
    pred = L_inf[:, None] + term

    ones = pred * 0.0 + 1.0

    # For the hard max, the gradient passes through the active branch.
    # indicator: 1 where log_t1 >= log_t2, 0 otherwise
    # Use a soft indicator for numerical stability: sigmoid with large scale would
    # be ideal, but since ops.maximum is used (hard max), use exact indicators.
    ind1 = (log_t1 >= log_t2) * 1.0  # indicator for t1 branch
    # For ties, treat as t1 branch (arbitrary but consistent)
    ind2 = 1.0 - ind1

    # d(log_max)/d(a) = ind1 * log(P)
    # d(log_max)/d(b) = ind2 * log(D)
    # d(log_max)/d(lam) = ind2 / lam   (d(log_lam)/d(lam) = 1/lam)
    # d(log_max)/d(d) = 0, d(log_max)/d(g) = 0

    # d(pred)/d(L_inf) = 1
    d_L_inf = ones
    # d(pred)/d(A) = exp_val
    d_A = exp_val
    # d(pred)/d(a) = term * (-d) * d(log_max)/d(a) = term * (-d) * ind1 * logP
    d_a = term * (-d_p[:, None]) * ind1 * logP[None, :]
    # d(pred)/d(b) = term * (-d) * ind2 * logD
    d_b = term * (-d_p[:, None]) * ind2 * logD[None, :]
    # d(pred)/d(d) = term * (-log_max) = A * exp_val * (-log_max)
    d_d = term * (-log_max)
    # d(pred)/d(lam) = term * (-d) * ind2 * (1/lam)
    lam_safe = ops.clamp_min(lam, _EPS)
    d_lam = term * (-d_p[:, None]) * ind2 / lam_safe[:, None]
    # d(pred)/d(g) = term * (-logV)
    d_g = term * (-logV[None, :])

    jac = ops.stack([d_L_inf, d_A, d_a, d_b, d_d, d_lam, d_g], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_5 (7p): p0 * P^p1 * V^p2 * D^p3 + p4 * P^p5 + p6
# Two additive power terms
# theta: [p0, p1, p2, p3, p4, p5, p6]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    p0, p1, p2, p3, p4, p5, p6 = [theta[:, i] for i in range(7)]

    logP = xp.log(ops.clamp_min(P, _EPS))
    logV = xp.log(ops.clamp_min(V, _EPS))
    logD = xp.log(ops.clamp_min(D, _EPS))

    P_p1 = P[None, :] ** p1[:, None]
    V_p2 = V[None, :] ** p2[:, None]
    D_p3 = D[None, :] ** p3[:, None]
    P_p5 = P[None, :] ** p5[:, None]

    t1 = p0[:, None] * P_p1 * V_p2 * D_p3          # p0 * P^p1 * V^p2 * D^p3
    t2 = p4[:, None] * P_p5                          # p4 * P^p5
    pred = t1 + t2 + p6[:, None]

    ones = pred * 0.0 + 1.0

    # d/d(p0) = P^p1 * V^p2 * D^p3
    d_p0 = P_p1 * V_p2 * D_p3
    # d/d(p1) = t1 * log(P)
    d_p1 = t1 * logP[None, :]
    # d/d(p2) = t1 * log(V)
    d_p2 = t1 * logV[None, :]
    # d/d(p3) = t1 * log(D)
    d_p3 = t1 * logD[None, :]
    # d/d(p4) = P^p5
    d_p4 = P_p5
    # d/d(p5) = t2 * log(P)
    d_p5 = t2 * logP[None, :]
    # d/d(p6) = 1
    d_p6 = ones

    jac = ops.stack([d_p0, d_p1, d_p2, d_p3, d_p4, d_p5, d_p6], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_6 (7p): A * (P * V^k1)^(-alpha) + B * (D * V^k2)^(-beta) + c0
# Chinchilla dual-term, vocab modulates both eff-P and eff-D
# theta: [A, alpha, k1, B, beta, k2, c0]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    A, alpha, k1, B, beta, k2, c0 = [theta[:, i] for i in range(7)]

    logP = xp.log(ops.clamp_min(P, _EPS))
    logV = xp.log(ops.clamp_min(V, _EPS))
    logD = xp.log(ops.clamp_min(D, _EPS))

    eff_P = ops.clamp_min(P[None, :] * (V[None, :] ** k1[:, None]), _EPS)
    eff_D = ops.clamp_min(D[None, :] * (V[None, :] ** k2[:, None]), _EPS)
    log_eff_P = logP[None, :] + k1[:, None] * logV[None, :]
    log_eff_D = logD[None, :] + k2[:, None] * logV[None, :]

    eff_P_neg_alpha = eff_P ** (-alpha[:, None])
    eff_D_neg_beta = eff_D ** (-beta[:, None])

    term1 = A[:, None] * eff_P_neg_alpha              # A * (P*V^k1)^(-alpha)
    term2 = B[:, None] * eff_D_neg_beta               # B * (D*V^k2)^(-beta)
    pred = term1 + term2 + c0[:, None]

    ones = pred * 0.0 + 1.0

    # d/d(A) = eff_P^(-alpha)
    d_A = eff_P_neg_alpha
    # d/d(alpha) = A * eff_P^(-alpha) * (-log(eff_P)) = -term1 * log_eff_P
    d_alpha = -term1 * log_eff_P
    # d/d(k1):
    #   d(eff_P)/d(k1) = P * V^k1 * log(V) = eff_P * log(V)
    #   d(eff_P^(-alpha))/d(k1) = (-alpha) * eff_P^(-alpha-1) * eff_P * log(V)
    #                            = (-alpha) * eff_P^(-alpha) * log(V)
    #   d(term1)/d(k1) = -alpha * term1 * log(V) / 1  ... wait
    #   = A * (-alpha) * eff_P^(-alpha) * log(V) = -term1 * alpha * log(V)
    d_k1 = -term1 * alpha[:, None] * logV[None, :]
    # d/d(B) = eff_D^(-beta)
    d_B = eff_D_neg_beta
    # d/d(beta) = -term2 * log_eff_D
    d_beta = -term2 * log_eff_D
    # d/d(k2) = -term2 * beta * log(V)
    d_k2 = -term2 * beta[:, None] * logV[None, :]
    # d/d(c0) = 1
    d_c0 = ones

    jac = ops.stack([d_A, d_alpha, d_k1, d_B, d_beta, d_k2, d_c0], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_7 (7p): A * P^(-alpha) * D^(-beta) + B * V^gamma * D^(-delta) + c0
# Additive P-D term + V-D interaction term
# theta: [A, alpha, beta, B, gamma, delta, c0]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    P = ops.clamp_min(X[:, 0], _EPS)
    V = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    A, alpha, beta, B, gamma, delta, c0 = [theta[:, i] for i in range(7)]

    logP = xp.log(ops.clamp_min(P, _EPS))
    logV = xp.log(ops.clamp_min(V, _EPS))
    logD = xp.log(ops.clamp_min(D, _EPS))

    P_neg_alpha = P[None, :] ** (-alpha[:, None])
    D_neg_beta = D[None, :] ** (-beta[:, None])
    V_gamma = V[None, :] ** gamma[:, None]
    D_neg_delta = D[None, :] ** (-delta[:, None])

    t1 = A[:, None] * P_neg_alpha * D_neg_beta        # A * P^(-alpha) * D^(-beta)
    t2 = B[:, None] * V_gamma * D_neg_delta            # B * V^gamma * D^(-delta)
    pred = t1 + t2 + c0[:, None]

    ones = pred * 0.0 + 1.0

    # d/d(A) = P^(-alpha) * D^(-beta)
    d_A = P_neg_alpha * D_neg_beta
    # d/d(alpha) = -t1 * log(P)
    d_alpha = -t1 * logP[None, :]
    # d/d(beta) = -t1 * log(D)
    d_beta = -t1 * logD[None, :]
    # d/d(B) = V^gamma * D^(-delta)
    d_B = V_gamma * D_neg_delta
    # d/d(gamma) = t2 * log(V)
    d_gamma = t2 * logV[None, :]
    # d/d(delta) = -t2 * log(D)
    d_delta = -t2 * logD[None, :]
    # d/d(c0) = 1
    d_c0 = ones

    jac = ops.stack([d_A, d_alpha, d_beta, d_B, d_gamma, d_delta, d_c0], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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

    logP = xp.log(ops.clamp_min(P, _EPS))
    logV = xp.log(ops.clamp_min(V, _EPS))
    logD = xp.log(ops.clamp_min(D, _EPS))

    P_neg_alpha = P[None, :] ** (-alpha[:, None])
    D_neg_gamma = D[None, :] ** (-gamma[:, None])
    V_beta = V[None, :] ** beta[:, None]

    floor = c0[:, None] + c1[:, None] * logV[None, :]
    reducible = c2[:, None] * P_neg_alpha + c3[:, None] * D_neg_gamma
    scaled = V_beta * reducible

    pred = floor + scaled

    ones = pred * 0.0 + 1.0

    # d/d(c0) = 1
    d_c0 = ones
    # d/d(c1) = log(V)
    d_c1 = logV[None, :] + ones * 0.0
    # d/d(c2) = V^beta * P^(-alpha)
    d_c2 = V_beta * P_neg_alpha
    # d/d(alpha) = V^beta * c2 * P^(-alpha) * (-log(P))
    d_alpha = V_beta * c2[:, None] * P_neg_alpha * (-logP[None, :])
    # d/d(c3) = V^beta * D^(-gamma)
    d_c3 = V_beta * D_neg_gamma
    # d/d(gamma) = V^beta * c3 * D^(-gamma) * (-log(D))
    d_gamma = V_beta * c3[:, None] * D_neg_gamma * (-logD[None, :])
    # d/d(beta) = V^beta * log(V) * reducible = scaled * log(V)
    d_beta = scaled * logV[None, :]

    jac = ops.stack([d_c0, d_c1, d_c2, d_alpha, d_c3, d_gamma, d_beta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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

    logP = xp.log(ops.clamp_min(P, _EPS))
    logV = xp.log(ops.clamp_min(V, _EPS))
    logD = xp.log(ops.clamp_min(D, _EPS))

    P_neg_alpha = P[None, :] ** (-alpha[:, None])
    D_neg_beta = D[None, :] ** (-beta[:, None])
    V_epsilon = V[None, :] ** epsilon[:, None]

    main = A[:, None] * P_neg_alpha * D_neg_beta       # A * P^(-alpha) * D^(-beta)
    vocab_mod = 1.0 + gamma[:, None] * logV[None, :]   # 1 + gamma*log(V)
    cross = delta[:, None] * V_epsilon                  # delta * V^epsilon

    pred = main * vocab_mod + cross + L_inf[:, None]

    ones = pred * 0.0 + 1.0

    # d/d(A) = P^(-alpha) * D^(-beta) * vocab_mod
    d_A = P_neg_alpha * D_neg_beta * vocab_mod
    # d/d(alpha) = main * (-log(P)) * vocab_mod
    d_alpha = main * (-logP[None, :]) * vocab_mod
    # d/d(beta) = main * (-log(D)) * vocab_mod
    d_beta = main * (-logD[None, :]) * vocab_mod
    # d/d(gamma) = main * log(V)
    d_gamma = main * logV[None, :]
    # d/d(delta) = V^epsilon
    d_delta = V_epsilon
    # d/d(epsilon) = delta * V^epsilon * log(V) = cross * log(V)
    d_epsilon = cross * logV[None, :]
    # d/d(L_inf) = 1
    d_L_inf = ones

    jac = ops.stack([d_A, d_alpha, d_beta, d_gamma, d_delta, d_epsilon, d_L_inf], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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

    exp_val = ops.exp(exponent)
    pred = L_min[:, None] + exp_val

    ones = pred * 0.0 + 1.0

    # d/d(L_min) = 1
    d_L_min = ones
    # d/d(a) = exp_val * 1 = exp_val
    d_a = exp_val
    # d/d(bP) = exp_val * log(P)
    d_bP = exp_val * lP
    # d/d(bV1) = exp_val * log(V)
    d_bV1 = exp_val * lV
    # d/d(bV2) = exp_val * log(V)^2
    d_bV2 = exp_val * lV ** 2
    # d/d(bD) = exp_val * log(D)
    d_bD = exp_val * lD
    # d/d(bVD) = exp_val * log(V)*log(D)
    d_bVD = exp_val * lV * lD

    jac = ops.stack([d_L_min, d_a, d_bP, d_bV1, d_bV2, d_bD, d_bVD], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


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
