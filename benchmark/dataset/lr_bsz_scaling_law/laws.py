"""Scaling laws for learning-rate / batch-size / data-size / model-size."""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-30


# Scaling law 1 (15 params):
#   Degree-2 polynomial in log-space of 4 features, predicting log(lm_loss).
#   Output = exp(poly).
#
#   Features after log: z = [log(lr), log(bsz), log(data_size), log(non_embedding_param_size)]
#   Polynomial terms (15 total):
#     bias, z0, z1, z2, z3, z0^2, z1^2, z2^2, z3^2, z0*z1, z0*z2, z0*z3, z1*z2, z1*z3, z2*z3
#
# theta: (B, 15)
# X: [lr, bsz, data_size, non_embedding_param_size]
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    xp = ops.xp

    x0 = ops.clamp_min(X[:, 0], _EPS)
    x1 = ops.clamp_min(X[:, 1], _EPS)
    x2 = ops.clamp_min(X[:, 2], _EPS)
    x3 = ops.clamp_min(X[:, 3], _EPS)

    z0 = xp.log(x0)
    z1 = xp.log(x1)
    z2 = xp.log(x2)
    z3 = xp.log(x3)

    # Build feature matrix: (M, 15)
    ones = z0 * 0.0 + 1.0
    if backend == "torch":
        features = xp.stack([
            ones, z0, z1, z2, z3,
            z0 * z0, z1 * z1, z2 * z2, z3 * z3,
            z0 * z1, z0 * z2, z0 * z3, z1 * z2, z1 * z3, z2 * z3,
        ], dim=-1)  # (M, 15)
    else:
        features = xp.stack([
            ones, z0, z1, z2, z3,
            z0 * z0, z1 * z1, z2 * z2, z3 * z3,
            z0 * z1, z0 * z2, z0 * z3, z1 * z2, z1 * z3, z2 * z3,
        ], axis=-1)  # (M, 15)

    # theta: (B, 15), features: (M, 15) -> log_pred: (B, M)
    if backend == "torch":
        log_pred = xp.matmul(theta, features.T)
    else:
        log_pred = theta @ features.T

    pred = ops.exp(log_pred)
    return pred[0] if pred.shape[0] == 1 else pred


# sl_2 (26p): Physics-inspired softplus-penalty model
# base = L_inf + Cp*exp(-ap*p) + Cd*exp(-ad*s) + Cdp*exp(-adp*(s-k*p)) + Cbb*exp(-abb*v)
# u_star = u0 + up*p + us*s + uv*v;  v_star = v0 + vp*p + vs*s
# cL = softplus(cL0+cLp*p+cLs*s+cLv*v);  cB = softplus(cB0+cBp*p+cBs*s+cBv*v)
# penalty = cL*du^2 + cB*dv^2 + 2*rho*sqrt(cL*cB)*du*dv
# loss = base + penalty
# theta: [L_inf, Cp, ap, Cd, ad, Cdp, adp, k, u0, up, us, uv, v0, vp, vs,
#          cL0, cLp, cLs, cB0, cBp, cBs, rho, cLv, cBv, Cbb, abb]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    lr = ops.clamp_min(X[:, 0], _EPS)
    bsz = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    P = ops.clamp_min(X[:, 3], _EPS)
    u = xp.log(lr); v = xp.log(bsz); s = xp.log(D); p = xp.log(P)
    t = [theta[:, i] for i in range(26)]
    L_inf, Cp, ap, Cd, ad, Cdp, adp, k = t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]
    u0, up_, us, uv, v0, vp_, vs = t[8], t[9], t[10], t[11], t[12], t[13], t[14]
    cL0, cLp, cLs, cB0, cBp, cBs, rho = t[15], t[16], t[17], t[18], t[19], t[20], t[21]
    cLv, cBv, Cbb, abb = t[22], t[23], t[24], t[25]

    def softplus(x):
        return xp.log(1.0 + ops.exp(ops.clamp(x, min=-20.0, max=20.0)))

    # Base loss (power-law terms in exp form)
    Cp_s = softplus(Cp); ap_s = softplus(ap)
    Cd_s = softplus(Cd); ad_s = softplus(ad)
    Cdp_s = softplus(Cdp); adp_s = softplus(adp)
    Cbb_s = softplus(Cbb); abb_s = softplus(abb)

    base = (L_inf[:, None]
            + Cp_s[:, None] * ops.exp(-ap_s[:, None] * p[None, :])
            + Cd_s[:, None] * ops.exp(-ad_s[:, None] * s[None, :])
            + Cdp_s[:, None] * ops.exp(-adp_s[:, None] * (s[None, :] - k[:, None] * p[None, :]))
            + Cbb_s[:, None] * ops.exp(-abb_s[:, None] * v[None, :]))

    # Optimal lr and bsz
    u_star = u0[:, None] + up_[:, None] * p[None, :] + us[:, None] * s[None, :] + uv[:, None] * v[None, :]
    v_star = v0[:, None] + vp_[:, None] * p[None, :] + vs[:, None] * s[None, :]
    du = u[None, :] - u_star
    dv = v[None, :] - v_star

    # State-dependent curvatures
    cL = softplus(cL0[:, None] + cLp[:, None] * p[None, :] + cLs[:, None] * s[None, :] + cLv[:, None] * v[None, :])
    cB = softplus(cB0[:, None] + cBp[:, None] * p[None, :] + cBs[:, None] * s[None, :] + cBv[:, None] * v[None, :])

    # Correlated penalty
    rho_t = xp.tanh(rho[:, None]) if hasattr(xp, 'tanh') else (ops.exp(2.0 * rho[:, None]) - 1.0) / (ops.exp(2.0 * rho[:, None]) + 1.0)
    g = (cL * cB) ** 0.5
    penalty = cL * du ** 2 + cB * dv ** 2 + 2.0 * rho_t * g * du * dv

    pred = base + penalty
    return pred[0] if pred.shape[0] == 1 else pred


# sl_3 (24p): Chinchilla power-law + decoupled LR/BSZ quadratic valleys
# L = E + A*N^(-alpha) + B*D^(-beta) + F/(N^wN * D^wD)
#   + C_eff*(log(lr)-opt_lr)^2 + G_eff*(log(bsz)-opt_bsz)^2
# theta: [E, A, alpha, B, beta, F, wN, wD, C0, CN, CD, CB, mu0, muN, muD, muB, muND,
#          G0, GN, GD, nu0, nuN, nuD, nuND]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    lr = ops.clamp_min(X[:, 0], _EPS)
    bsz = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    P = ops.clamp_min(X[:, 3], _EPS)
    lnlr = xp.log(lr); lnb = xp.log(bsz); lnD = xp.log(D); lnP = xp.log(P)
    t = [theta[:, i] for i in range(24)]
    E, A, alpha, B, beta, F, wN, wD = t[:8]
    C0, CN, CD, CB, mu0, muN, muD, muB, muND = t[8:17]
    G0, GN, GD, nu0, nuN, nuD, nuND = t[17:24]

    base = (E[:, None]
            + A[:, None] * (P[None, :] ** (-alpha[:, None]))
            + B[:, None] * (D[None, :] ** (-beta[:, None]))
            + F[:, None] / (ops.clamp_min(P[None, :] ** wN[:, None] * D[None, :] ** wD[:, None], _EPS)))

    opt_lr = mu0[:, None] + muN[:, None]*lnP[None, :] + muD[:, None]*lnD[None, :] + muB[:, None]*lnb[None, :] + muND[:, None]*lnP[None, :]*lnD[None, :]
    C_eff = C0[:, None] * ops.exp(CN[:, None]*lnP[None, :] + CD[:, None]*lnD[None, :] + CB[:, None]*lnb[None, :])
    lr_pen = C_eff * (lnlr[None, :] - opt_lr) ** 2

    opt_bsz = nu0[:, None] + nuN[:, None]*lnP[None, :] + nuD[:, None]*lnD[None, :] + nuND[:, None]*lnP[None, :]*lnD[None, :]
    G_eff = G0[:, None] * ops.exp(GN[:, None]*lnP[None, :] + GD[:, None]*lnD[None, :])
    bsz_pen = G_eff * (lnb[None, :] - opt_bsz) ** 2

    pred = base + lr_pen + bsz_pen
    return pred[0] if pred.shape[0] == 1 else pred


# sl_4 (20p): Log-polynomial-2 + inverse features
# log(loss) = w . phi(X), loss = exp(...)
# phi = [1, log(lr), log(bsz), log(D), log(P),
#        log(lr)^2, log(bsz)^2, log(D)^2, log(P)^2,
#        log(lr)*log(bsz), log(lr)*log(D), log(lr)*log(P),
#        log(bsz)*log(D), log(bsz)*log(P), log(D)*log(P),
#        log(D)-log(P), 1/bsz, 1/bsz^2, 1/D, 1/P]
# theta: 20 weights
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    lr = ops.clamp_min(X[:, 0], _EPS)
    bsz = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    P = ops.clamp_min(X[:, 3], _EPS)
    z0 = xp.log(lr); z1 = xp.log(bsz); z2 = xp.log(D); z3 = xp.log(P)
    ones = z0 * 0.0 + 1.0
    feat_list = [
        ones, z0, z1, z2, z3,
        z0*z0, z1*z1, z2*z2, z3*z3,
        z0*z1, z0*z2, z0*z3, z1*z2, z1*z3, z2*z3,
        z2 - z3,
        1.0 / bsz, 1.0 / (bsz * bsz), 1.0 / D, 1.0 / P,
    ]
    if backend == "torch":
        features = xp.stack(feat_list, dim=-1)
        log_pred = xp.matmul(theta, features.T)
    else:
        features = xp.stack(feat_list, axis=-1)
        log_pred = theta @ features.T
    pred = ops.exp(log_pred)
    return pred[0] if pred.shape[0] == 1 else pred


# sl_5 (19p): Chinchilla + exp-decay + LR quadratic penalty
# u=log(lr), v=log(bsz), s=log(D), n=log(P)
# u_star = u0+kb*v+kn*n+kd*s; lr_amp = clr0*exp(-wb*v-wn*n-ws*s)
# loss = L0 + AN*exp(-aN*n) + AD*exp(-aD*s) + AB*exp(-aB*v)
#      + AR*exp(-aR*(s-n)^2) + AX*exp(-aX*(s-v)) + lr_amp*(u-u_star)^2
# theta: [L0, AN, aN, AD, aD, AB, aB, clr0, u0, kb, kn, kd, wb, wn, ws, AR, aR, AX, aX]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    lr = ops.clamp_min(X[:, 0], _EPS)
    bsz = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    P = ops.clamp_min(X[:, 3], _EPS)
    u = xp.log(lr); v = xp.log(bsz); s = xp.log(D); n = xp.log(P)
    t = [theta[:, i] for i in range(19)]
    L0, AN, aN, AD, aD, AB, aB = t[:7]
    clr0, u0, kb, kn, kd, wb, wn, ws = t[7:15]
    AR, aR, AX, aX = t[15:19]

    base = (L0[:, None]
            + AN[:, None] * ops.exp(-aN[:, None] * n[None, :])
            + AD[:, None] * ops.exp(-aD[:, None] * s[None, :])
            + AB[:, None] * ops.exp(-aB[:, None] * v[None, :]))
    ratio_term = AR[:, None] * ops.exp(-aR[:, None] * (s[None, :] - n[None, :]) ** 2)
    cross_term = AX[:, None] * ops.exp(-aX[:, None] * (s[None, :] - v[None, :]))
    u_star = u0[:, None] + kb[:, None]*v[None, :] + kn[:, None]*n[None, :] + kd[:, None]*s[None, :]
    lr_amp = clr0[:, None] * ops.exp(-wb[:, None]*v[None, :] - wn[:, None]*n[None, :] - ws[:, None]*s[None, :])
    lr_pen = lr_amp * (u[None, :] - u_star) ** 2
    pred = base + ratio_term + cross_term + lr_pen
    return pred[0] if pred.shape[0] == 1 else pred


# sl_6 (14p): L_inf + exp(partial poly2)
# loss = L_inf + exp(w0 + w_d*log(D) + w_p*log(P) + w_dp*log(D)*log(P)
#                   + w_lr*log(lr) + w_lr2*log(lr)^2
#                   + w_bsz*log(bsz) + w_bsz2*log(bsz)^2
#                   + w_lrbsz*log(lr)*log(bsz)
#                   + w_lrD*log(lr)*log(D) + w_lrP*log(lr)*log(P)
#                   + w_bszD*log(bsz)*log(D) + w_bszP*log(bsz)*log(P))
# theta: [L_inf, w0, w_d, w_p, w_dp, w_lr, w_lr2, w_bsz, w_bsz2,
#          w_lrbsz, w_lrD, w_lrP, w_bszD, w_bszP]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    lr = ops.clamp_min(X[:, 0], _EPS)
    bsz = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    P = ops.clamp_min(X[:, 3], _EPS)
    lnlr = xp.log(lr); lnb = xp.log(bsz); lnD = xp.log(D); lnP = xp.log(P)
    t = [theta[:, i] for i in range(14)]
    L_inf = t[0]
    exponent = (t[1][:, None]
                + t[2][:, None]*lnD[None, :] + t[3][:, None]*lnP[None, :]
                + t[4][:, None]*lnD[None, :]*lnP[None, :]
                + t[5][:, None]*lnlr[None, :] + t[6][:, None]*lnlr[None, :]**2
                + t[7][:, None]*lnb[None, :] + t[8][:, None]*lnb[None, :]**2
                + t[9][:, None]*lnlr[None, :]*lnb[None, :]
                + t[10][:, None]*lnlr[None, :]*lnD[None, :] + t[11][:, None]*lnlr[None, :]*lnP[None, :]
                + t[12][:, None]*lnb[None, :]*lnD[None, :] + t[13][:, None]*lnb[None, :]*lnP[None, :])
    exponent = ops.clamp(exponent, min=-50.0, max=50.0)
    pred = L_inf[:, None] + ops.exp(exponent)
    return pred[0] if pred.shape[0] == 1 else pred


# sl_7 (31p): E + exp(poly2_A) + exp(poly2_B) dual-term
# features = [1, x1..x4, x1^2..x4^2, x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4]
# (15 features from log inputs)
# loss = E + exp(features . w1) + exp(features . w2)
# theta: [E, w1[0..14], w2[0..14]]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    lr = ops.clamp_min(X[:, 0], _EPS)
    bsz = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    P = ops.clamp_min(X[:, 3], _EPS)
    z0 = xp.log(lr); z1 = xp.log(bsz); z2 = xp.log(D); z3 = xp.log(P)
    ones = z0 * 0.0 + 1.0
    feat_list = [
        ones, z0, z1, z2, z3,
        z0*z0, z1*z1, z2*z2, z3*z3,
        z0*z1, z0*z2, z0*z3, z1*z2, z1*z3, z2*z3,
    ]
    if backend == "torch":
        features = xp.stack(feat_list, dim=-1)  # (M, 15)
    else:
        features = xp.stack(feat_list, axis=-1)  # (M, 15)
    E = theta[:, 0]
    w1 = theta[:, 1:16]   # (B, 15)
    w2 = theta[:, 16:31]  # (B, 15)
    if backend == "torch":
        log1 = xp.matmul(w1, features.T)
        log2 = xp.matmul(w2, features.T)
    else:
        log1 = w1 @ features.T
        log2 = w2 @ features.T
    log1 = ops.clamp(log1, min=-50.0, max=50.0)
    log2 = ops.clamp(log2, min=-50.0, max=50.0)
    pred = E[:, None] + ops.exp(log1) + ops.exp(log2)
    return pred[0] if pred.shape[0] == 1 else pred


# sl_8 (20p): Chinchilla + asymmetric tanh-skewed penalties
# u=log(lr), v=log(bsz), s=log(D), p=log(P)
# term_P = cP*exp(-aP*p); term_D = cD*exp(-aD*s); term_R = cR*exp(-aR*(s-p))
# lr_opt = phi0+phi_b*v+phi_p*p+phi_d*s; dev=u-lr_opt
# lr_pen = k_lr*dev^2*(1+a_lr*tanh(dev))
# ns = u-0.5*v; ns_opt = psi0+psi_p*p+psi_d*s; dev_ns=ns-ns_opt
# ns_pen = k_ns*dev_ns^2*(1+a_ns*tanh(dev_ns))
# dp_pen = k_dp*((s-p)-delta0)^2
# loss = L0 + term_P + term_D + term_R + lr_pen + ns_pen + dp_pen
# theta: [L0, cP, aP, cD, aD, cR, aR, phi0, phi_b, phi_p, phi_d,
#          k_lr, a_lr, psi0, psi_p, psi_d, k_ns, a_ns, delta0, k_dp]
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    lr = ops.clamp_min(X[:, 0], _EPS)
    bsz = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    P = ops.clamp_min(X[:, 3], _EPS)
    u = xp.log(lr); v = xp.log(bsz); s = xp.log(D); p = xp.log(P)
    t = [theta[:, i] for i in range(20)]

    def softplus(x):
        return xp.log(1.0 + ops.exp(ops.clamp(x, min=-20.0, max=20.0)))

    def tanh(x):
        e2x = ops.exp(ops.clamp(2.0 * x, min=-40.0, max=40.0))
        return (e2x - 1.0) / (e2x + 1.0)

    L0 = t[0]
    cP = softplus(t[1]); aP = softplus(t[2])
    cD = softplus(t[3]); aD = softplus(t[4])
    cR = softplus(t[5]); aR = softplus(t[6])

    term_P = cP[:, None] * ops.exp(-aP[:, None] * p[None, :])
    term_D = cD[:, None] * ops.exp(-aD[:, None] * s[None, :])
    term_R = cR[:, None] * ops.exp(-aR[:, None] * (s[None, :] - p[None, :]))

    lr_opt = t[7][:, None] + t[8][:, None]*v[None, :] + t[9][:, None]*p[None, :] + t[10][:, None]*s[None, :]
    dev_lr = u[None, :] - lr_opt
    k_lr = softplus(t[11])
    a_lr = tanh(t[12])
    lr_pen = k_lr[:, None] * dev_lr**2 * (1.0 + a_lr[:, None] * tanh(dev_lr))

    ns = u[None, :] - 0.5 * v[None, :]
    ns_opt = t[13][:, None] + t[14][:, None]*p[None, :] + t[15][:, None]*s[None, :]
    dev_ns = ns - ns_opt
    k_ns = softplus(t[16])
    a_ns = tanh(t[17])
    ns_pen = k_ns[:, None] * dev_ns**2 * (1.0 + a_ns[:, None] * tanh(dev_ns))

    dp_pen = softplus(t[19])[:, None] * ((s[None, :] - p[None, :]) - t[18][:, None])**2

    pred = L0[:, None] + term_P + term_D + term_R + lr_pen + ns_pen + dp_pen
    return pred[0] if pred.shape[0] == 1 else pred


# sl_9 (15p): Direct poly2(log10) without exp transform
# x1=log10(lr), x2=log10(bsz), x3=log10(D), x4=log10(P)
# loss = c0 + c1*x1 + ... + c14*x3*x4
# theta: 15 coefficients
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    lr = ops.clamp_min(X[:, 0], _EPS)
    bsz = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    P = ops.clamp_min(X[:, 3], _EPS)
    log10_inv = 1.0 / xp.log(lr * 0.0 + 10.0)
    z0 = xp.log(lr) * log10_inv
    z1 = xp.log(bsz) * log10_inv
    z2 = xp.log(D) * log10_inv
    z3 = xp.log(P) * log10_inv
    ones = z0 * 0.0 + 1.0
    feat_list = [
        ones, z0, z1, z2, z3,
        z0*z0, z1*z1, z2*z2, z3*z3,
        z0*z1, z0*z2, z0*z3, z1*z2, z1*z3, z2*z3,
    ]
    if backend == "torch":
        features = xp.stack(feat_list, dim=-1)
        pred = xp.matmul(theta, features.T)
    else:
        features = xp.stack(feat_list, axis=-1)
        pred = theta @ features.T
    return pred[0] if pred.shape[0] == 1 else pred


# sl_10 (18p): Direct poly2(log) + fixed-exponent power features
# loss = poly2(log(lr), log(bsz), log(D), log(P)) + w_D*D^(-0.5) + w_P*P^(-0.5) + w_bsz*bsz^(-1)
# theta: [c0..c14 (15 poly coeffs), w_D, w_P, w_bsz]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    lr = ops.clamp_min(X[:, 0], _EPS)
    bsz = ops.clamp_min(X[:, 1], _EPS)
    D = ops.clamp_min(X[:, 2], _EPS)
    P = ops.clamp_min(X[:, 3], _EPS)
    z0 = xp.log(lr); z1 = xp.log(bsz); z2 = xp.log(D); z3 = xp.log(P)
    ones = z0 * 0.0 + 1.0
    feat_list = [
        ones, z0, z1, z2, z3,
        z0*z0, z1*z1, z2*z2, z3*z3,
        z0*z1, z0*z2, z0*z3, z1*z2, z1*z3, z2*z3,
    ]
    if backend == "torch":
        features = xp.stack(feat_list, dim=-1)
        poly = xp.matmul(theta[:, :15], features.T)
    else:
        features = xp.stack(feat_list, axis=-1)
        poly = theta[:, :15] @ features.T
    w_D = theta[:, 15]
    w_P = theta[:, 16]
    w_bsz = theta[:, 17]
    power_terms = (w_D[:, None] * (D[None, :] ** (-0.5))
                   + w_P[:, None] * (P[None, :] ** (-0.5))
                   + w_bsz[:, None] / bsz[None, :])
    pred = poly + power_terms
    return pred[0] if pred.shape[0] == 1 else pred


PARAM_BOUNDS = {
    # Dataset: lr∈[1.2e-4,0.022], bsz∈[16,4096], D∈[2e9,1e11], P∈[6e7,1.07e9]
    # z0=log(lr)∈[-9,-4], z1=log(bsz)∈[3,8], z2=log(D)∈[21,25], z3=log(P)∈[18,21]
    # lm_loss∈[2.08,3.70], log(loss)∈[0.73,1.31], Δlog(loss)≈0.58

    # sl_1: 15p poly2 -> exp (NO clamp in model)
    # [bias, z0, z1, z2, z3, z0^2, z1^2, z2^2, z3^2, z0z1, z0z2, z0z3, z1z2, z1z3, z2z3]
    # poly output = log(loss) ∈ [0.73, 1.31]. Linear coeff bounds: |c|*Δz ≤ 0.58.
    # Δz0=5.2,Δz1=5.5,Δz2=3.9,Δz3=2.9 → max|c_linear|≤0.2. Quad/cross: |c|*Δ(z^2) ≤ 0.58.
    # Max Δ(z2^2)=183 → |c|≤0.003. Bias absorbs mean offsets: z2_mean*c2≈23*0.2=4.6 → bias∈(-15,15).
    # Fit: bias=6.4, linear max=0.25, quad/cross max=0.013.
    "sl_1": [(-15, 15)] + [(-0.35, 0.35)] * 4 + [(-0.025, 0.025)] * 10,

    # sl_2: 26p softplus-penalty model (all softplus/tanh-clamped internally)
    # [L_inf, Cp, ap, Cd, ad, Cdp, adp, k, u0, up, us, uv, v0, vp, vs,
    #  cL0, cLp, cLs, cB0, cBp, cBs, rho, cLv, cBv, Cbb, abb]
    # L_inf: irreducible loss ≈ 1.0. Cp/Cd/Cdp pre-softplus: fit found ~9-13 → allow up to 15.
    # ap/ad/adp/abb: rate params, pre-softplus.
    # k: mixing param for D/P interaction; k>3 risks overflow in exp(-adp*(s-k*p)) → restrict k≤3.
    # u0: optimal log(lr) offset; v0: optimal log(bsz) offset.
    # cL0,cB0: LR/BSZ curvature pre-softplus (fit found ~8-10); cLp,cBp,cLs,cBs,cLv,cBv: scaling.
    # rho: correlation pre-tanh; Cbb/abb: batch-size penalty.
    "sl_2": (
        [(0, 3)]           # L_inf
        + [(-5, 15), (-5, 5)] * 3  # Cp,ap, Cd,ad, Cdp,adp
        + [(-5, 3)]        # k (k≤3 prevents exp overflow in Cdp term)
        + [(-20, 2)]       # u0 (log(lr_opt)∈[-9,-4])
        + [(-3, 3)] * 3    # up, us, uv
        + [(0, 20)]        # v0 (log(bsz_opt)∈[3,8]; fit found 11.7)
        + [(-3, 3)] * 2    # vp, vs
        + [(-12, 12)] * 2  # cL0, cLp (fit found cL0=7.9, cLp=-9.2 near bounds)
        + [(-12, 12)] * 2  # cLs, cB0 (fit found cB0=8.0 near upper bound)
        + [(-12, 12)] * 2  # cBp, cBs
        + [(-8, 8)]        # rho (pre-tanh; fit found -4.9 near lower -5)
        + [(-12, 12)] * 2  # cLv, cBv
        + [(-5, 15), (-5, 5)]  # Cbb, abb
    ),

    # sl_3: 24p Chinchilla + LR/BSZ penalties
    # [E, A, alpha, B, beta, F, wN, wD,
    #  C0, CN, CD, CB, mu0, muN, muD, muB, muND,
    #  G0, GN, GD, nu0, nuN, nuD, nuND]
    # E: irreducible loss (fit: 1.74). A/B: Chinchilla amplitudes (fit: 113, 3285).
    # F: joint (N,D) amplitude (fit: 8.4e6, near upper 1e7 → expand to 2e7).
    # C0,G0: LR/BSZ curvature (fit: 0.14, 0.16).
    # mu0: log(lr_opt) intercept (fit: -7.4). nu0: log(bsz_opt) intercept (fit: 7.9).
    "sl_3": (
        [(0.5, 3)]         # E (fit: 1.74)
        + [(0, 5e5), (0.05, 2)] * 2  # A,alpha, B,beta
        + [(0, 2e7), (0.05, 2), (0.05, 2)]  # F, wN, wD
        + [(0, 3), (-3, 3), (-3, 3), (-3, 3)]  # C0, CN, CD, CB
        + [(-20, 5)]       # mu0 (log(lr_opt) intercept; fit: -7.4)
        + [(-10, 3)] * 4   # muN, muD, muB, muND
        + [(0, 3), (-3, 3), (-3, 3)]  # G0, GN, GD
        + [(0, 15)]        # nu0 (log(bsz_opt) intercept; fit: 7.9)
        + [(-3, 3)] * 3    # nuN, nuD, nuND
    ),

    # sl_4: 20p poly2+extras -> exp (NO clamp in model)
    # [15 poly coeffs, z2-z3, 1/bsz, 1/bsz^2, 1/D, 1/P]
    # Same poly as sl_1 for first 15 params. Extra features added inside the exponent.
    # 1/bsz∈[2.4e-4,0.0625]: |w|*0.062 ≤ 0.58 → |w|≤9.3; fit: -0.031.
    # 1/bsz^2∈[6e-8,3.9e-3]: |w|*3.9e-3 ≤ 0.58 → |w|≤149; fit: 3.08.
    # 1/D∈[1e-11,5e-10]: |w|*5e-10 ≤ 0.58 → |w|≤1.2e9; fit: ~0.
    # 1/P∈[9.3e-10,1.67e-8]: |w|*1.67e-8 ≤ 0.58 → |w|≤3.5e7; fit: ~0.
    "sl_4": (
        [(-15, 15)] + [(-0.35, 0.35)] * 4 + [(-0.025, 0.025)] * 10
        + [(-0.3, 0.3)]    # z2-z3 = log(D/P); fit: -0.093
        + [(-10, 10)]      # 1/bsz; fit: -0.031
        + [(-200, 200)]    # 1/bsz^2; fit: 3.08
        + [(-1.5e9, 1.5e9)]  # 1/D; fit: ~0
        + [(-5e7, 5e7)]    # 1/P; fit: ~0
    ),

    # sl_5: 19p Chinchilla + exp-decay + LR penalty
    # [L0, AN, aN, AD, aD, AB, aB, clr0, u0, kb, kn, kd, wb, wn, ws, AR, aR, AX, aX]
    # AN*exp(-aN*n): n=log(P)∈[18,21]; aN~0.26 → AN~110 for 0.5 contribution.
    # AD*exp(-aD*s): s=log(D)∈[21,25]; aD~0.52 → AD~7200 for 0.5 contribution.
    # AB*exp(-aB*v): v=log(bsz)∈[3,8]; contribution~AB*0.2; AB~1 at optimal.
    # AX*exp(-aX*(s-v)): s-v∈[14,22]; aX~0.49 → AX~535 for visible contribution.
    # clr0: LR curvature amplitude; u0: log(lr_opt) center.
    "sl_5": (
        [(0, 3)]           # L0 (fit: 1.5)
        + [(0, 2e4), (0.01, 2)]  # AN, aN (fit: 110, 0.26)
        + [(0, 2e4), (0.01, 2)]  # AD, aD (fit: 7202, 0.52)
        + [(0, 20), (0.01, 2)]   # AB, aB (fit: 1.03, 0.40)
        + [(0, 200)]       # clr0 (fit: 74.1)
        + [(-12, 3)]       # u0 (log(lr_opt) center; fit: -1.96)
        + [(-2, 2)] * 3    # kb, kn, kd
        + [(-2, 2)] * 3    # wb, wn, ws
        + [(0, 5), (0, 2)]     # AR, aR (fit: 0.27, 0.11)
        + [(0, 2000), (0, 2)]  # AX, aX (fit: 535, 0.49)
    ),

    # sl_6: 14p L_inf + exp(poly13), HAS clamp [-50,50] on exponent
    # [L_inf, w0, w_d, w_p, w_dp, w_lr, w_lr2, w_bsz, w_bsz2, w_lrbsz, w_lrD, w_lrP, w_bszD, w_bszP]
    # exp(poly) = lm_loss - L_inf ∈ (0, 1.7]; log of that ≤ 0.53. Same scale analysis as sl_1.
    # w_lr,w_bsz: up to 0.47 (z0/z1 range 5.2/5.5); w_dp,cross: small (quad range ~143→|c|≤0.004).
    # Fit (DE): L_inf=1.82, w0=-0.74, w_d=0.15, w_p=0.31, w_dp=-0.011, w_lr=0.47, w_lr2=0.052,
    #   w_bsz=0.36, w_bsz2=0.034, w_lrbsz=-0.029, w_lrD=-0.007, w_lrP=0.027, w_bszD=-0.033, w_bszP=-0.007
    "sl_6": (
        [(0, 3)]           # L_inf
        + [(-8, 8)]        # w0 (bias of inner poly)
        + [(-0.5, 0.5)] * 2   # w_d, w_p (z2,z3 linear)
        + [(-0.025, 0.025)]   # w_dp (z2*z3 cross; Δ=143)
        + [(-0.6, 0.6)] * 2   # w_lr, w_lr2 (z0 linear and quad)
        + [(-0.5, 0.5)] * 2   # w_bsz, w_bsz2 (z1 linear and quad)
        + [(-0.04, 0.04)]     # w_lrbsz (z0*z1 cross; Δ=62.5)
        + [(-0.02, 0.02)]     # w_lrD (z0*z2; Δ=139)
        + [(-0.04, 0.04)]     # w_lrP (z0*z3; Δ=114)
        + [(-0.05, 0.05)]     # w_bszD (z1*z2; Δ=151)
        + [(-0.02, 0.02)]     # w_bszP (z1*z3; Δ=120)
    ),

    # sl_7: 31p E + 2*exp(poly2), HAS clamp [-50,50] on each exponent
    # [E, w1[0..14], w2[0..14]] — each poly2 has same structure as sl_1
    # Each exp term ≤ lm_loss-E ≤ 1.7; same coefficient analysis as sl_1.
    # Quad/cross bounds expanded to (-0.03,0.03) since some coefficients found near ±0.02.
    # Fit: E=1.75, w1_bias=4.26, w2_bias=3.17; quad coeffs up to ±0.02.
    "sl_7": (
        [(0, 2.5)]         # E
        + [(-15, 15)] + [(-0.35, 0.35)] * 4 + [(-0.03, 0.03)] * 10  # w1
        + [(-15, 15)] + [(-0.35, 0.35)] * 4 + [(-0.03, 0.03)] * 10  # w2
    ),

    # sl_8: 20p softplus/tanh model (all internally clamped via softplus/tanh)
    # [L0, cP, aP, cD, aD, cR, aR, phi0, phi_b, phi_p, phi_d,
    #  k_lr, a_lr, psi0, psi_p, psi_d, k_ns, a_ns, delta0, k_dp]
    # All cP,cD,cR,k_lr,k_ns,k_dp pre-softplus; aP,aD,aR pre-softplus (positive rates).
    # phi0: log(lr_opt) intercept (fit: -14.1); psi0: log(bsz_opt) intercept (fit: -1.4).
    # delta0: (log(D/P)) offset; s-p∈[0.6,6.1]; fit: 6.8.
    # All fit values within bounds; no expansion needed.
    "sl_8": (
        [(-1, 3)]          # L0 (fit: -0.44)
        + [(-3, 12), (-5, 5)] * 3  # cP,aP, cD,aD, cR,aR (pre-softplus)
        + [(-20, 3)]       # phi0 (fit: -14.1)
        + [(-5, 5)] * 3    # phi_b, phi_p, phi_d
        + [(-5, 10)]       # k_lr (pre-softplus; fit: -4.9)
        + [(-5, 5)]        # a_lr (pre-tanh; fit: 3.5)
        + [(-15, 5)]       # psi0 (log(bsz_opt) intercept; fit: -1.4)
        + [(-5, 5)] * 2    # psi_p, psi_d
        + [(-5, 10)]       # k_ns (pre-softplus; fit: 0.74)
        + [(-5, 5)]        # a_ns (pre-tanh; fit: 4.2)
        + [(-5, 10)]       # delta0 (s-p offset; fit: 6.8)
        + [(-5, 10)]       # k_dp (pre-softplus; fit: -4.9)
    ),

    # sl_9: 15p direct poly2(log10), no exp transform
    # [c0, c_lr, c_bsz, c_D, c_P, c_lr2, c_bsz2, c_D2, c_P2, c_lr_bsz..c_D_P]
    # log10: z0∈[-3.91,-1.66], z1∈[1.20,3.61], z2∈[9.30,11.0], z3∈[7.78,9.03]
    # loss ∈ [2.08, 3.70], Δ=1.62. Linear: |c|*Δz ≤ 1.62 → |c_D|≤1.62/1.7=0.95.
    # Quad: |c|*Δ(z2^2)=|c|*34.5 ≤ 1.62 → |c|≤0.047. Cross: Δ(z2*z3)=27 → |c|≤0.06.
    # Bias must absorb z3 contribution at mean: c_P*z3_mean≈-0.31*8.4=-2.6. Fit: bias=16.09.
    # Fit: c0=16.09, c_D=-2.06, c_P=-0.31, quad max=0.14, cross max=0.13.
    "sl_9": (
        [(-5, 30)]         # c0 (bias; fit: 16.09)
        + [(-4, 4)] * 4    # c_lr, c_bsz, c_D, c_P (fit max: |c_D|=2.06)
        + [(-0.3, 0.3)] * 10  # quad + cross coefficients (fit max: 0.14)
    ),

    # sl_10: 18p direct poly2(log) + power features, no exp
    # [c0..c14, w_D, w_P, w_bsz]  (natural log, not log10)
    # Natural log: z0∈[-9,-4], z1∈[3,8], z2∈[21,25], z3∈[18,21]
    # Δz3=2.9 → max|c_P|≤1.62/2.9=0.56. Quad: Δ(z2^2)=183 → |c|≤0.009.
    # Bias absorbs z3 contribution: at mean z3=19.3, c3=-1.34 → 19.3*(-1.34)=-25.9; bias∈(-5,35).
    # Power: D^(-0.5)∈[3.2e-6,2.2e-5], Δ=1.92e-5 → |w_D|≤1.62/1.92e-5=84375; fit: 1.68e4.
    # P^(-0.5)∈[3.1e-5,1.3e-4], Δ=9.9e-5 → |w_P|≤1.62/9.9e-5=16364; fit: -3469.
    # 1/bsz∈[2.4e-4,0.0625], Δ=0.062 → |w_bsz|≤26; fit: 1.0.
    "sl_10": (
        [(-5, 35)]         # c0 (bias; fit: 16.27)
        + [(-2, 2)] * 4    # c_lr, c_bsz, c_D, c_P (fit: c_P=-1.34)
        + [(-0.1, 0.1)] * 10  # quad + cross (fit max: 0.044)
        + [(-5e4, 5e4)]    # w_D (D^-0.5; fit: 1.68e4)
        + [(-2e4, 2e4)]    # w_P (P^-0.5; fit: -3469)
        + [(-15, 15)]      # w_bsz (1/bsz; fit: 1.0)
    ),
}

LAW_REGISTRY = {
    "sl_1": sl_1, "sl_2": sl_2, "sl_3": sl_3, "sl_4": sl_4, "sl_5": sl_5,
    "sl_6": sl_6, "sl_7": sl_7, "sl_8": sl_8, "sl_9": sl_9, "sl_10": sl_10,
}
PARAM_COUNTS = {
    "sl_1": 15, "sl_2": 26, "sl_3": 24, "sl_4": 20, "sl_5": 19,
    "sl_6": 14, "sl_7": 31, "sl_8": 20, "sl_9": 15, "sl_10": 18,
}
