"""Scaling laws for vocabulary size scaling.

X columns: [non_vocab_parameters (P), vocab_size (V), num_characters (D)]
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-6


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

    logP = xp.log(P)
    logD = xp.log(D)
    logV = xp.log(V)

    r_safe = ops.clamp_min(r, _EPS)
    inv_r = 1.0 / r_safe

    # ── All heavy lifting in log-space ──────────────────────────
    # log(tp_r) = -alpha * r * log(P),  log(td_r) = -beta * r * log(D)
    log_tp_r = -alpha[:, None] * r_safe[:, None] * logP[None, :]
    log_td_r = -beta[:, None] * r_safe[:, None] * logD[None, :]

    # log(S) = log((tp_r + td_r)/2)  via logsumexp
    log_max = xp.maximum(log_tp_r, log_td_r)
    log_sum_raw = log_max + xp.log(
        xp.exp(log_tp_r - log_max) + xp.exp(log_td_r - log_max)
    )
    log_S = log_sum_raw - xp.log(2.0)   # = log((tp_r+td_r)/2)

    # mean_r = S^(1/r) = exp(log_S / r), clipped to prevent overflow
    _LOG_CLIP = 50.0
    log_mean_r = xp.clip(log_S * inv_r[:, None], -_LOG_CLIP, _LOG_CLIP)
    mean_r = xp.exp(log_mean_r)

    # ── Stable ratio via sigmoid ────────────────────────────────
    # tp_r / (tp_r + td_r) = sigmoid(log_tp_r - log_td_r)
    # td_r / (tp_r + td_r) = sigmoid(log_td_r - log_tp_r)
    # Note: tp_r / (2*S) = tp_r / (tp_r + td_r) = sigmoid(diff)
    diff = log_tp_r - log_td_r
    diff_clip = xp.clip(diff, -_LOG_CLIP, _LOG_CLIP)
    sig_p = 1.0 / (1.0 + xp.exp(-diff_clip))   # = tp_r/(tp_r+td_r)
    sig_d = 1.0 - sig_p                          # = td_r/(tp_r+td_r)

    # ── Vocab gate ──────────────────────────────────────────────
    lV_diff = logV[None, :] - v0[:, None]
    lV_diff2 = lV_diff ** 2
    vocab_gate = 1.0 + C[:, None] * lV_diff2

    # ── Prediction ──────────────────────────────────────────────
    product = mean_r * vocab_gate
    pred = L[:, None] + A[:, None] * product

    # ── Jacobian ────────────────────────────────────────────────
    d_L = xp.ones_like(pred)
    d_A = product

    # d/d(alpha): mean_r * tp_r/(2S) * logP = mean_r * sig_p * logP
    d_alpha = A[:, None] * vocab_gate * (-mean_r * sig_p * logP[None, :])

    # d/d(beta): mean_r * td_r/(2S) * logD = mean_r * sig_d * logD
    d_beta = A[:, None] * vocab_gate * (-mean_r * sig_d * logD[None, :])

    d_C = A[:, None] * mean_r * lV_diff2
    d_v0 = A[:, None] * mean_r * (-2.0 * C[:, None] * lV_diff)

    # d/d(r): dmr_dr = mean_r * [(-1/r²)*log_S + (1/r)*(sig_p*log_term_p + sig_d*log_term_d)]
    # where log_term_p = -alpha*logP, log_term_d = -beta*logD
    log_term_p = -alpha[:, None] * logP[None, :]
    log_term_d = -beta[:, None] * logD[None, :]
    dlogS_dr_over_r = sig_p * log_term_p + sig_d * log_term_d  # = (1/S)*dS/dr
    dmr_dr = mean_r * (
        -inv_r[:, None] ** 2 * xp.clip(log_S, -_LOG_CLIP, _LOG_CLIP)
        + inv_r[:, None] * dlogS_dr_over_r
    )
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

    logP = xp.log(P)
    logV = xp.log(V)
    logD = xp.log(D)

    q_safe = ops.clamp_min(q, _EPS)
    inv_q = 1.0 / q_safe

    _LOG_CLIP = 50.0

    # ── Log-space for t1, t2 ───────────────────────────────────
    # t1 = a * P^(-alpha)  =>  log_t1 = log(a) + (-alpha)*log(P)
    log_a = xp.log(ops.clamp_min(a, _EPS))
    log_b = xp.log(ops.clamp_min(b, _EPS))

    log_t1 = log_a[:, None] + (-alpha[:, None]) * logP[None, :]

    # t2 = b * (D * V^phi)^(-beta)
    # log_t2 = log(b) + (-beta)*(log(D) + phi*log(V))
    log_eff_D = logD[None, :] + phi[:, None] * logV[None, :]
    log_t2 = log_b[:, None] + (-beta[:, None]) * log_eff_D

    # ── Log-space for S = t1^q + t2^q ─────────────────────────
    # log_t1_q = q * log_t1,  log_t2_q = q * log_t2
    log_t1_q = q_safe[:, None] * log_t1
    log_t2_q = q_safe[:, None] * log_t2

    # log_S = logsumexp(log_t1_q, log_t2_q)
    log_max = xp.maximum(log_t1_q, log_t2_q)
    log_S = log_max + xp.log(
        xp.exp(log_t1_q - log_max) + xp.exp(log_t2_q - log_max)
    )

    # combined = S^(1/q) = exp(log_S / q)
    log_combined = xp.clip(log_S * inv_q[:, None], -_LOG_CLIP, _LOG_CLIP)
    combined = xp.exp(log_combined)

    # ── Stable ratios via sigmoid ──────────────────────────────
    # t1_q / S = sigmoid(log_t1_q - log_t2_q)
    # t2_q / S = sigmoid(log_t2_q - log_t1_q)
    diff = xp.clip(log_t1_q - log_t2_q, -_LOG_CLIP, _LOG_CLIP)
    sig_1 = 1.0 / (1.0 + xp.exp(-diff))    # = t1^q / S
    sig_2 = 1.0 - sig_1                      # = t2^q / S

    # ── Prediction ─────────────────────────────────────────────
    pred = L0[:, None] + combined

    # ── Jacobian ───────────────────────────────────────────────
    # Shared factor: d(combined)/d(S) * d(S)/d(t_k^q) * d(t_k^q)/d(...)
    # d(combined)/d(S) = combined / (q * S)
    # d(S)/d(t1_q) = 1
    # d(t1_q)/d(theta) = t1_q * q * d(log_t1)/d(theta)
    #
    # Chain: d(combined)/d(log_t1) = combined / (q*S) * t1_q * q
    #                               = combined * (t1_q / S)
    #                               = combined * sig_1
    # Similarly for t2.

    d_L0 = xp.ones_like(pred)

    # d/d(a): d(log_t1)/d(a) = 1/a
    #   => d(combined)/d(a) = combined * sig_1 * q * (1/a)
    d_a = combined * sig_1 * q_safe[:, None] / a[:, None]

    # d/d(alpha): d(log_t1)/d(alpha) = -logP
    #   => d(t1_q)/d(alpha) = t1_q * q * (-logP)
    #   => d(combined)/d(alpha) = combined * sig_1 * q * (-logP)
    d_alpha = combined * sig_1 * q_safe[:, None] * (-logP[None, :])

    # d/d(b): d(log_t2)/d(b) = 1/b
    d_b = combined * sig_2 * q_safe[:, None] / b[:, None]

    # d/d(beta): d(log_t2)/d(beta) = -log_eff_D
    d_beta = combined * sig_2 * q_safe[:, None] * (-log_eff_D)

    # d/d(phi): d(log_t2)/d(phi) = (-beta) * logV
    d_phi = combined * sig_2 * q_safe[:, None] * (-beta[:, None]) * logV[None, :]

    # d/d(q):
    #   combined = S^(1/q),  log(combined) = log_S / q
    #   d(combined)/d(q) = combined * [ -log_S/q² + (1/q)(1/S)*dS/dq ]
    #
    #   dS/dq = t1_q * log_t1 + t2_q * log_t2
    #   (1/S)*dS/dq = sig_1 * log_t1 + sig_2 * log_t2
    #
    #   Use clipped log values throughout
    log_t1_clip = xp.clip(log_t1, -_LOG_CLIP, _LOG_CLIP)
    log_t2_clip = xp.clip(log_t2, -_LOG_CLIP, _LOG_CLIP)
    log_S_clip = xp.clip(log_S, -_LOG_CLIP, _LOG_CLIP)

    inv_S_dS_dq = sig_1 * log_t1_clip + sig_2 * log_t2_clip

    d_q = combined * (
        -inv_q[:, None] ** 2 * log_S_clip
        + inv_q[:, None] * inv_S_dS_dq
    )

    jac = ops.stack([d_L0, d_a, d_alpha, d_b, d_beta, d_phi, d_q], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


_LOG_CLIP = 50.0


# sl_4 (7p): L_inf + A * max(P^a, lambda*D^b)^(-d) * V^(-g)
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

    # log(P^a) = a*logP,  log(lam*D^b) = log(lam) + b*logD
    log_t1 = a[:, None] * logP[None, :]
    log_lam = xp.log(ops.clamp_min(lam, _EPS))
    log_t2 = log_lam[:, None] + b[:, None] * logD[None, :]
    log_max = xp.maximum(log_t1, log_t2)

    # term = A * max(...)^(-d) * V^(-g) = A * exp(-d*log_max - g*logV)
    log_term = xp.log(ops.clamp_min(A_p, _EPS))[:, None] \
               - d_p[:, None] * log_max - g[:, None] * logV[None, :]
    log_term = xp.clip(log_term, -_LOG_CLIP, _LOG_CLIP)
    term = xp.exp(log_term)

    pred = L_inf[:, None] + term

    # Indicator for which branch is active
    ind1 = (log_t1 >= log_t2) * 1.0
    ind2 = 1.0 - ind1

    d_L_inf = xp.ones_like(pred)
    # d/dA = term / A
    d_A = term / ops.clamp_min(A_p, _EPS)[:, None]
    # d/da = term * (-d) * ind1 * logP
    d_a = term * (-d_p[:, None]) * ind1 * logP[None, :]
    # d/db = term * (-d) * ind2 * logD
    d_b = term * (-d_p[:, None]) * ind2 * logD[None, :]
    # d/dd = term * (-log_max)
    d_d = term * (-log_max)
    # d/dlam = term * (-d) * ind2 / lam
    d_lam = term * (-d_p[:, None]) * ind2 / ops.clamp_min(lam, _EPS)[:, None]
    # d/dg = term * (-logV)
    d_g = term * (-logV[None, :])

    jac = ops.stack([d_L_inf, d_A, d_a, d_b, d_d, d_lam, d_g], axis=-1)
    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_5 (7p): p0 * P^p1 * V^p2 * D^p3 + p4 * P^p5 + p6
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

    logP = xp.log(P)
    logV = xp.log(V)
    logD = xp.log(D)

    # t1 = p0 * P^p1 * V^p2 * D^p3
    # log|t1| = log|p0| + p1*logP + p2*logV + p3*logD
    log_t1_abs = xp.log(ops.clamp_min(xp.abs(p0), _EPS))[:, None] \
                 + p1[:, None] * logP[None, :] \
                 + p2[:, None] * logV[None, :] \
                 + p3[:, None] * logD[None, :]
    log_t1_abs = xp.clip(log_t1_abs, -_LOG_CLIP, _LOG_CLIP)
    sign_p0 = xp.sign(p0)
    t1 = sign_p0[:, None] * xp.exp(log_t1_abs)

    # t2 = p4 * P^p5
    log_t2_abs = xp.log(ops.clamp_min(xp.abs(p4), _EPS))[:, None] \
                 + p5[:, None] * logP[None, :]
    log_t2_abs = xp.clip(log_t2_abs, -_LOG_CLIP, _LOG_CLIP)
    sign_p4 = xp.sign(p4)
    t2 = sign_p4[:, None] * xp.exp(log_t2_abs)

    pred = t1 + t2 + p6[:, None]

    # Power-law parts (unsigned, for Jacobian): P^p1*V^p2*D^p3, P^p5
    pvd = xp.exp(xp.clip(
        p1[:, None] * logP[None, :] + p2[:, None] * logV[None, :] + p3[:, None] * logD[None, :],
        -_LOG_CLIP, _LOG_CLIP))
    pp5 = xp.exp(xp.clip(p5[:, None] * logP[None, :], -_LOG_CLIP, _LOG_CLIP))

    d_p0 = sign_p0[:, None] * pvd  # preserves sign correctly when p0 < 0
    d_p1 = t1 * logP[None, :]
    d_p2 = t1 * logV[None, :]
    d_p3 = t1 * logD[None, :]
    d_p4 = sign_p4[:, None] * pp5
    d_p5 = t2 * logP[None, :]
    d_p6 = xp.ones_like(pred)

    jac = ops.stack([d_p0, d_p1, d_p2, d_p3, d_p4, d_p5, d_p6], axis=-1)
    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_6 (7p): A * (P * V^k1)^(-alpha) + B * (D * V^k2)^(-beta) + c0
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

    logP = xp.log(P)
    logV = xp.log(V)
    logD = xp.log(D)

    # log_eff_P = logP + k1*logV,  log_eff_D = logD + k2*logV
    log_eff_P = logP[None, :] + k1[:, None] * logV[None, :]
    log_eff_D = logD[None, :] + k2[:, None] * logV[None, :]

    # term1 = A * exp(-alpha * log_eff_P)
    log_term1 = xp.log(ops.clamp_min(A, _EPS))[:, None] \
                - alpha[:, None] * log_eff_P
    log_term1 = xp.clip(log_term1, -_LOG_CLIP, _LOG_CLIP)
    term1 = xp.exp(log_term1)

    # term2 = B * exp(-beta * log_eff_D)
    log_term2 = xp.log(ops.clamp_min(B, _EPS))[:, None] \
                - beta[:, None] * log_eff_D
    log_term2 = xp.clip(log_term2, -_LOG_CLIP, _LOG_CLIP)
    term2 = xp.exp(log_term2)

    pred = term1 + term2 + c0[:, None]

    d_A = term1 / ops.clamp_min(A, _EPS)[:, None]
    d_alpha = -term1 * log_eff_P
    d_k1 = -term1 * alpha[:, None] * logV[None, :]
    d_B = term2 / ops.clamp_min(B, _EPS)[:, None]
    d_beta = -term2 * log_eff_D
    d_k2 = -term2 * beta[:, None] * logV[None, :]
    d_c0 = xp.ones_like(pred)

    jac = ops.stack([d_A, d_alpha, d_k1, d_B, d_beta, d_k2, d_c0], axis=-1)
    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_7 (7p): A * P^(-alpha) * D^(-beta) + B * V^gamma * D^(-delta) + c0
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

    logP = xp.log(P)
    logV = xp.log(V)
    logD = xp.log(D)

    # t1 = A * P^(-alpha) * D^(-beta) = A * exp(-alpha*logP - beta*logD)
    log_t1 = xp.log(ops.clamp_min(A, _EPS))[:, None] \
             - alpha[:, None] * logP[None, :] \
             - beta[:, None] * logD[None, :]
    log_t1 = xp.clip(log_t1, -_LOG_CLIP, _LOG_CLIP)
    t1 = xp.exp(log_t1)

    # t2 = B * V^gamma * D^(-delta) = B * exp(gamma*logV - delta*logD)
    log_t2 = xp.log(ops.clamp_min(B, _EPS))[:, None] \
             + gamma[:, None] * logV[None, :] \
             - delta[:, None] * logD[None, :]
    log_t2 = xp.clip(log_t2, -_LOG_CLIP, _LOG_CLIP)
    t2 = xp.exp(log_t2)

    pred = t1 + t2 + c0[:, None]

    d_A = t1 / ops.clamp_min(A, _EPS)[:, None]
    d_alpha = -t1 * logP[None, :]
    d_beta = -t1 * logD[None, :]
    d_B = t2 / ops.clamp_min(B, _EPS)[:, None]
    d_gamma = t2 * logV[None, :]
    d_delta = -t2 * logD[None, :]
    d_c0 = xp.ones_like(pred)

    jac = ops.stack([d_A, d_alpha, d_beta, d_B, d_gamma, d_delta, d_c0], axis=-1)
    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_8 (7p): c0 + c1*log(V) + V^beta * (c2 * P^(-alpha) + c3 * D^(-gamma))
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

    logP = xp.log(P)
    logV = xp.log(V)
    logD = xp.log(D)

    # V^beta * c2 * P^(-alpha) = c2 * exp(beta*logV - alpha*logP)
    log_vp = beta[:, None] * logV[None, :] - alpha[:, None] * logP[None, :]
    log_vp = xp.clip(log_vp, -_LOG_CLIP, _LOG_CLIP)
    vp = xp.exp(log_vp)   # V^beta * P^(-alpha)

    # V^beta * c3 * D^(-gamma) = c3 * exp(beta*logV - gamma*logD)
    log_vd = beta[:, None] * logV[None, :] - gamma[:, None] * logD[None, :]
    log_vd = xp.clip(log_vd, -_LOG_CLIP, _LOG_CLIP)
    vd = xp.exp(log_vd)   # V^beta * D^(-gamma)

    scaled = c2[:, None] * vp + c3[:, None] * vd
    pred = c0[:, None] + c1[:, None] * logV[None, :] + scaled

    d_c0 = xp.ones_like(pred)
    d_c1 = xp.broadcast_to(logV[None, :], pred.shape) + 0.0  # force copy
    d_c2 = vp
    d_alpha = c2[:, None] * vp * (-logP[None, :])
    d_c3 = vd
    d_gamma = c3[:, None] * vd * (-logD[None, :])
    d_beta = scaled * logV[None, :]

    jac = ops.stack([d_c0, d_c1, d_c2, d_alpha, d_c3, d_gamma, d_beta], axis=-1)
    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_9 (7p): A * P^(-alpha) * D^(-beta) * (1 + gamma*log(V)) + delta * V^epsilon + L_inf
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

    logP = xp.log(P)
    logV = xp.log(V)
    logD = xp.log(D)

    # main = A * P^(-alpha) * D^(-beta) = A * exp(-alpha*logP - beta*logD)
    log_main = xp.log(ops.clamp_min(A, _EPS))[:, None] \
               - alpha[:, None] * logP[None, :] \
               - beta[:, None] * logD[None, :]
    log_main = xp.clip(log_main, -_LOG_CLIP, _LOG_CLIP)
    main = xp.exp(log_main)

    vocab_mod = 1.0 + gamma[:, None] * logV[None, :]

    # cross = delta * V^epsilon = delta * exp(epsilon*logV)
    log_cross = xp.log(ops.clamp_min(xp.abs(delta), _EPS))[:, None] \
                + epsilon[:, None] * logV[None, :]
    log_cross = xp.clip(log_cross, -_LOG_CLIP, _LOG_CLIP)
    sign_delta = xp.sign(delta)
    cross = sign_delta[:, None] * xp.exp(log_cross)

    pred = main * vocab_mod + cross + L_inf[:, None]

    d_A = main / ops.clamp_min(A, _EPS)[:, None] * vocab_mod
    d_alpha = main * (-logP[None, :]) * vocab_mod
    d_beta = main * (-logD[None, :]) * vocab_mod
    d_gamma = main * logV[None, :]
    d_delta = cross / (sign_delta[:, None] * ops.clamp_min(xp.abs(delta), _EPS)[:, None] + 1e-300) \
              * sign_delta[:, None]  # = V^epsilon with correct sign handling
    # Simpler: d/d(delta) = V^epsilon
    v_eps = xp.exp(xp.clip(epsilon[:, None] * logV[None, :], -_LOG_CLIP, _LOG_CLIP))
    d_delta = v_eps
    d_epsilon = cross * logV[None, :]
    d_L_inf = xp.ones_like(pred)

    jac = ops.stack([d_A, d_alpha, d_beta, d_gamma, d_delta, d_epsilon, d_L_inf], axis=-1)
    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_10 (7p): L_min + exp(a + bP*log(P) + bV1*log(V) + bV2*log(V)^2 + bD*log(D) + bVD*log(V)*log(D))
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

    exponent = (a[:, None] + bP[:, None] * lP + bV1[:, None] * lV
                + bV2[:, None] * lV ** 2 + bD[:, None] * lD
                + bVD[:, None] * lV * lD)
    exponent = xp.clip(exponent, -_LOG_CLIP, _LOG_CLIP)
    exp_val = xp.exp(exponent)

    pred = L_min[:, None] + exp_val

    d_L_min = xp.ones_like(pred)
    d_a = exp_val
    d_bP = exp_val * lP
    d_bV1 = exp_val * lV
    d_bV2 = exp_val * lV ** 2
    d_bD = exp_val * lD
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
