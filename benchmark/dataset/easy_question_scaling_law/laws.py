"""Scaling laws for easy-question (Brier score vs. compute) prediction.

X columns: [log_flops (x)]
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


# sl_1 (3p): y0 + k * (x - m)^2
# theta: [y0, k, m]
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    y0, k, m = [theta[:, i] for i in range(3)]
    diff = x[None, :] - m[:, None]
    diff2 = diff ** 2
    pred = y0[:, None] + k[:, None] * diff2

    # Jacobian: d/d[y0, k, m]
    ones = pred * 0.0 + 1.0
    d_y0 = ones                              # d/dy0 = 1
    d_k = diff2                              # d/dk = (x-m)^2
    d_m = -2.0 * k[:, None] * diff           # d/dm = -2k(x-m)
    jac = ops.stack([d_y0, d_k, d_m], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_2 (4p): d + a*(x-c)^2 / (1 + b*(x-c)^2)
# theta: [a, b, c, d]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    a, b, c, d = [theta[:, i] for i in range(4)]
    diff = x[None, :] - c[:, None]
    diff_sq = diff ** 2
    denom = 1.0 + b[:, None] * diff_sq
    frac = diff_sq / denom
    pred = d[:, None] + a[:, None] * frac

    # Jacobian: d/d[a, b, c, d]
    ones = pred * 0.0 + 1.0
    denom2 = denom ** 2
    d_a = frac                                                          # d/da = (x-c)^2 / denom
    d_b = -a[:, None] * diff_sq ** 2 / denom2                          # d/db = -a*(x-c)^4 / denom^2
    # d(pred)/dc = a * d/dc[(x-c)^2 / denom]
    # Using quotient rule: d/dc[u/v] = (u'v - uv') / v^2
    # u = (x-c)^2, u' = -2(x-c); v = 1+b*(x-c)^2, v' = -2b(x-c)
    # = (-2(x-c)*denom - (x-c)^2*(-2b*(x-c))) / denom^2
    # = (-2(x-c)*denom + 2b*(x-c)^3) / denom^2
    # = -2(x-c)*(denom - b*(x-c)^2) / denom^2
    # = -2(x-c)*1 / denom^2 = -2*diff / denom^2
    d_c = a[:, None] * (-2.0 * diff) / denom2
    d_d = ones                                                          # d/dd = 1
    jac = ops.stack([d_a, d_b, d_c, d_d], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_3 (6p): A + C1 * sigmoid((x - (c-d)) / s) + C2 * sigmoid(((c+d) - x) / s)
# Double-sigmoid U-shape
# theta: [A, C1, C2, c, d, s]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    A, C1, C2, c, d, s = [theta[:, i] for i in range(6)]
    s_safe = ops.clamp_min(s, _EPS)

    def sigmoid(z):
        return 1.0 / (1.0 + ops.exp(-z))

    arg1 = (x[None, :] - (c[:, None] - d[:, None])) / s_safe[:, None]
    arg2 = ((c[:, None] + d[:, None]) - x[None, :]) / s_safe[:, None]
    sig1 = sigmoid(arg1)
    sig2 = sigmoid(arg2)
    pred = A[:, None] + C1[:, None] * sig1 + C2[:, None] * sig2

    # Jacobian: d/d[A, C1, C2, c, d, s]
    ones = pred * 0.0 + 1.0
    dsig1 = sig1 * (1.0 - sig1)  # d(sig1)/d(arg1)
    dsig2 = sig2 * (1.0 - sig2)  # d(sig2)/d(arg2)

    inv_s = 1.0 / s_safe[:, None]

    d_A = ones
    d_C1 = sig1
    d_C2 = sig2
    # d(arg1)/dc = -1/s, d(arg2)/dc = 1/s
    d_c = C1[:, None] * dsig1 * (-inv_s) + C2[:, None] * dsig2 * inv_s
    # d(arg1)/dd = 1/s, d(arg2)/dd = 1/s
    d_d = C1[:, None] * dsig1 * inv_s + C2[:, None] * dsig2 * inv_s
    # d(arg1)/ds = -arg1/s, d(arg2)/ds = -arg2/s
    d_s = C1[:, None] * dsig1 * (-arg1 * inv_s) + C2[:, None] * dsig2 * (-arg2 * inv_s)
    jac = ops.stack([d_A, d_C1, d_C2, d_c, d_d, d_s], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_4 (6p): f + e*(x-c) + b*(x-c)^2 + d*(x-c)^4
# Quartic potential
# theta: [f, e, b, d, c, _unused] -> actually 5 free params, pad to 6 for consistency
# Use 5p: theta: [f, e, b, d, c]
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    f, e, b, d, c = [theta[:, i] for i in range(5)]
    diff = x[None, :] - c[:, None]
    diff2 = diff ** 2
    diff4 = diff2 ** 2
    pred = f[:, None] + e[:, None] * diff + b[:, None] * diff2 + d[:, None] * diff4

    # Jacobian: d/d[f, e, b, d, c]
    ones = pred * 0.0 + 1.0
    d_f = ones
    d_e = diff
    d_b = diff2
    d_d = diff4
    # d(pred)/dc = -e - 2b*(x-c) - 4d*(x-c)^3
    d_c = -e[:, None] - 2.0 * b[:, None] * diff - 4.0 * d[:, None] * (diff2 * diff)
    jac = ops.stack([d_f, d_e, d_b, d_d, d_c], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_5 (5p): p1*x + p2 + p3 * exp(-(x - p4)^2 / (2 * p5^2))
# Linear trend + Gaussian bump
# theta: [p1, p2, p3, p4, p5]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    p1, p2, p3, p4, p5 = [theta[:, i] for i in range(5)]
    p5_safe = ops.clamp_min(p5, _EPS)
    diff = x[None, :] - p4[:, None]
    p5s = p5_safe[:, None]
    exponent = -(diff ** 2) / (2.0 * p5s ** 2)
    gauss_raw = ops.exp(exponent)  # exp(-(x-p4)^2/(2*p5^2))
    linear = p1[:, None] * x[None, :] + p2[:, None]
    gauss = p3[:, None] * gauss_raw
    pred = linear + gauss

    # Jacobian: d/d[p1, p2, p3, p4, p5]
    ones = pred * 0.0 + 1.0
    d_p1 = x[None, :] + ones * 0.0  # broadcast to (B, M)
    d_p2 = ones
    d_p3 = gauss_raw
    # d(gauss)/dp4 = p3 * gauss_raw * (x-p4)/p5^2
    d_p4 = gauss * diff / (p5s ** 2)
    # d(gauss)/dp5 = p3 * gauss_raw * (x-p4)^2 / p5^3
    d_p5 = gauss * (diff ** 2) / (p5s ** 3)
    jac = ops.stack([d_p1, d_p2, d_p3, d_p4, d_p5], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_6 (6p): y0 + a * sigmoid(k*(x-t1)) - b * sigmoid(k*(x-t2))
# Difference of two sigmoids (performance hump)
# theta: [y0, a, b, k, t1, t2]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    y0, a, b, k, t1, t2 = [theta[:, i] for i in range(6)]

    def sigmoid(z):
        return 1.0 / (1.0 + ops.exp(-z))

    z1 = k[:, None] * (x[None, :] - t1[:, None])
    z2 = k[:, None] * (x[None, :] - t2[:, None])
    sig1 = sigmoid(z1)
    sig2 = sigmoid(z2)
    pred = y0[:, None] + a[:, None] * sig1 - b[:, None] * sig2

    # Jacobian: d/d[y0, a, b, k, t1, t2]
    ones = pred * 0.0 + 1.0
    dsig1 = sig1 * (1.0 - sig1)
    dsig2 = sig2 * (1.0 - sig2)

    d_y0 = ones
    d_a = sig1
    d_b = -sig2
    # d(z1)/dk = (x-t1), d(z2)/dk = (x-t2)
    d_k = (a[:, None] * dsig1 * (x[None, :] - t1[:, None])
           - b[:, None] * dsig2 * (x[None, :] - t2[:, None]))
    # d(z1)/dt1 = -k
    d_t1 = a[:, None] * dsig1 * (-k[:, None])
    # d(z2)/dt2 = -k
    d_t2 = -b[:, None] * dsig2 * (-k[:, None])
    jac = ops.stack([d_y0, d_a, d_b, d_k, d_t1, d_t2], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_7 (6p): b_inf + s*exp(-alpha*x) + a*exp(-0.5*((x - mu)/sigma)^2)
# Exponential decay + Gaussian bump
# theta: [b_inf, s, alpha, a, mu, sigma]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    b_inf, s, alpha, a, mu, sigma = [theta[:, i] for i in range(6)]
    sigma_safe = ops.clamp_min(sigma, _EPS)
    exp_decay = ops.exp(-alpha[:, None] * x[None, :])
    s_exp = s[:, None] * exp_decay
    diff = x[None, :] - mu[:, None]
    sig_s = sigma_safe[:, None]
    gauss_raw = ops.exp(-0.5 * (diff / sig_s) ** 2)
    a_gauss = a[:, None] * gauss_raw
    pred = b_inf[:, None] + s_exp + a_gauss

    # Jacobian: d/d[b_inf, s, alpha, a, mu, sigma]
    ones = pred * 0.0 + 1.0
    d_b_inf = ones
    d_s = exp_decay
    d_alpha = s_exp * (-x[None, :])
    d_a = gauss_raw
    # d(gauss)/dmu = a * gauss_raw * (x-mu)/sigma^2
    d_mu = a_gauss * diff / (sig_s ** 2)
    # d(gauss)/dsigma = a * gauss_raw * (x-mu)^2 / sigma^3
    d_sigma = a_gauss * (diff ** 2) / (sig_s ** 3)
    jac = ops.stack([d_b_inf, d_s, d_alpha, d_a, d_mu, d_sigma], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_8 (5p): c0 + c1*exp(alpha*x) + c2*exp(gamma*x)
# Bi-exponential
# theta: [c0, c1, alpha, c2, gamma]
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    c0, c1, alpha, c2, gamma = [theta[:, i] for i in range(5)]
    exp1 = ops.exp(alpha[:, None] * x[None, :])
    exp2 = ops.exp(gamma[:, None] * x[None, :])
    term1 = c1[:, None] * exp1
    term2 = c2[:, None] * exp2
    pred = c0[:, None] + term1 + term2

    # Jacobian: d/d[c0, c1, alpha, c2, gamma]
    ones = pred * 0.0 + 1.0
    d_c0 = ones
    d_c1 = exp1
    d_alpha = term1 * x[None, :]
    d_c2 = exp2
    d_gamma = term2 * x[None, :]
    jac = ops.stack([d_c0, d_c1, d_alpha, d_c2, d_gamma], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_9 (5p): (a*x^2 + b*x + c) / (d*x^2 + e*x + 1)
# Padé rational function
# theta: [a, b, c, d, e]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    a, b, c, d, e = [theta[:, i] for i in range(5)]
    x_ = x[None, :]
    x2 = x_ ** 2
    numer = a[:, None] * x2 + b[:, None] * x_ + c[:, None]
    denom = d[:, None] * x2 + e[:, None] * x_ + 1.0
    pred = numer / denom

    # Jacobian: d/d[a, b, c, d, e]
    # pred = numer / denom
    # d(pred)/d(theta_i) = (d(numer)/d(theta_i) * denom - numer * d(denom)/d(theta_i)) / denom^2
    inv_denom = 1.0 / denom
    inv_denom2 = inv_denom ** 2
    d_a = x2 * inv_denom                           # d(numer)/da = x^2, d(denom)/da = 0
    d_b = x_ * inv_denom                            # d(numer)/db = x
    d_c = inv_denom                                  # d(numer)/dc = 1
    d_d = -numer * x2 * inv_denom2                   # d(denom)/dd = x^2
    d_e = -numer * x_ * inv_denom2                   # d(denom)/de = x
    jac = ops.stack([d_a, d_b, d_c, d_d, d_e], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# sl_10 (3p): a*x + b*exp(-x) + c
# Linear + inverse-flops (since exp(-log_flops) = 1/flops)
# theta: [a, b, c]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    x = X[:, 0]
    a, b, c = [theta[:, i] for i in range(3)]
    exp_neg_x = ops.exp(-x[None, :])
    pred = a[:, None] * x[None, :] + b[:, None] * exp_neg_x + c[:, None]

    # Jacobian: d/d[a, b, c]
    ones = pred * 0.0 + 1.0
    d_a = x[None, :] + ones * 0.0  # broadcast to (B, M)
    d_b = exp_neg_x + ones * 0.0   # broadcast to (B, M)
    d_c = ones
    jac = ops.stack([d_a, d_b, d_c], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


PARAM_BOUNDS = {
    # Dataset: x = log_flops in [-0.9, 2.9], y = brier_score in [-0.77, -0.0007]
    # sl_1: [y0, k, m] — y0 + k*(x-m)^2
    # y0: vertex value ~[-0.9, 0.05]; k: curvature ~[-0.04, 0.08]; m: vertex position
    # m can lie far outside data range when fitting monotone curves (degenerate in m)
    "sl_1": [(-1, 0.1), (-0.15, 0.15), (-6, 7)],
    # sl_2: [a, b, c, d] — d + a*(x-c)^2 / (1 + b*(x-c)^2)
    # d: baseline ~[-0.72, -0.07]; a: ~[-0.95, 4]; b: saturation ~[-0.04, 20]; c: center
    "sl_2": [(-2, 6), (-0.05, 25), (-2, 4), (-0.9, 0.1)],
    # sl_3: [A, C1, C2, c, d, s] — A + C1*sigmoid + C2*sigmoid (double sigmoid U-shape)
    # A: baseline ~[-1.4, 0.2]; C1 ~[-0.3, 1.1]; C2 ~[-0.6, 0.5]; c: center; d: half-width; s: scale
    "sl_3": [(-1.5, 0.5), (-0.5, 1.5), (-0.8, 0.7), (-1, 5), (0, 7), (0.005, 2)],
    # sl_4: [f, e, b, d, c] — quartic polynomial about c
    # f: value at c ~[-1.4, 0.06]; e: slope ~[-0.7, 0.7]; b: quadratic ~[-0.16, 0.14];
    # d: quartic ~[-0.021, 0.012]; c: center ~[-2.8, 6.4]
    "sl_4": [(-1.5, 0.2), (-0.8, 0.8), (-0.25, 0.25), (-0.03, 0.02), (-5, 8)],
    # sl_5: [p1, p2, p3, p4, p5] — linear trend + Gaussian bump
    # p1: slope ~[-0.06, 0.55]; p2: intercept ~[-1.5, 0.2]; p3: Gaussian amp ~[-1.5, 1.5];
    # p4: center ~[-1.1, 3.6]; p5: width ~[0.18, 2.8]
    "sl_5": [(-0.2, 0.7), (-2, 0.3), (-2.5, 2), (-2, 5), (0.05, 4)],
    # sl_6: [y0, a, b, k, t1, t2] — difference of two sigmoids
    # y0: baseline ~[-1.1, -0.003]; a, b: amplitudes; k: steepness (degenerate for |k|>>1/range);
    # t1, t2: transition points ~[-1.3, 4.7]
    "sl_6": [(-1.5, 0.2), (-3, 3), (-3, 3), (-10, 10), (-2, 6), (-2, 6)],
    # sl_7: [b_inf, s, alpha, a, mu, sigma] — exponential decay + Gaussian bump
    # x in [-0.9, 2.9]; b_inf: limit at x→inf; s: exp amplitude ~[-0.16, 0.03];
    # alpha: decay rate (degenerate as s→0 for large alpha); a: Gaussian amp; mu: center; sigma: width
    "sl_7": [(-0.5, 0.5), (-0.5, 0.5), (-3, 8), (-2, 2), (-4, 4), (0.01, 7)],
    # sl_8: [c0, c1, alpha, c2, gamma] — bi-exponential; x in [-0.9, 2.9]
    # c0: constant ~[-0.54, 0.3]; c1, c2: amplitudes; alpha, gamma: exponents (negative = decay)
    # At x=-0.9 with alpha=-8: exp(7.2)≈1327; keep |coeff| small or alpha near 0
    "sl_8": [(-1, 1), (-3, 3), (-8, 2), (-3, 3), (-8, 2)],
    # sl_9: [a, b, c, d, e] — Padé rational (a*x^2+b*x+c)/(d*x^2+e*x+1)
    # a ~[-0.30, 0.24]; b ~[-0.53, 0.52]; c ~[-0.54, -0.05]; d ~[-0.48, 0.83]; e ~[-0.86, 1.40]
    "sl_9": [(-0.5, 0.4), (-0.7, 0.7), (-0.7, 0.1), (-0.7, 1.0), (-1.2, 1.8)],
    # sl_10: [a, b, c] — a*x + b*exp(-x) + c
    # a: slope ~[-0.08, 0.18]; b: exp coeff ~[-0.24, 0.24]; c: intercept ~[-0.78, 0.06]
    "sl_10": [(-0.2, 0.3), (-0.4, 0.4), (-0.9, 0.2)],
}

LAW_REGISTRY = {
    "sl_1": sl_1, "sl_2": sl_2, "sl_3": sl_3, "sl_4": sl_4, "sl_5": sl_5,
    "sl_6": sl_6, "sl_7": sl_7, "sl_8": sl_8, "sl_9": sl_9, "sl_10": sl_10,
}
PARAM_COUNTS = {
    "sl_1": 3, "sl_2": 4, "sl_3": 6, "sl_4": 5, "sl_5": 5,
    "sl_6": 6, "sl_7": 6, "sl_8": 5, "sl_9": 5, "sl_10": 3,
}
