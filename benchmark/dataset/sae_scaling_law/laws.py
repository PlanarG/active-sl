"""Scaling laws for Sparse Autoencoders.

X columns: [n (training tokens), k (SAE dictionary size)]

Law:
  L(n,k) = exp(alpha + beta_k*log(k) + beta_n*log(n) + gamma*log(k)*log(n))
         + exp(zeta + eta*log(k))

theta: [alpha, beta_k, beta_n, gamma, zeta, eta]
"""

from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12


def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    n = ops.clamp_min(X[:, 0], _EPS)  # (M,)
    k = ops.clamp_min(X[:, 1], _EPS)  # (M,)

    alpha = theta[:, 0]   # (B,)
    beta_k = theta[:, 1]
    beta_n = theta[:, 2]
    gamma = theta[:, 3]
    zeta = theta[:, 4]
    eta = theta[:, 5]

    log_n = xp.log(n)[None, :]   # (1, M)
    log_k = xp.log(k)[None, :]   # (1, M)

    exponent1 = (
        alpha[:, None]
        + beta_k[:, None] * log_k
        + beta_n[:, None] * log_n
        + gamma[:, None] * log_k * log_n
    )
    exponent1 = ops.clamp(exponent1, min=-50.0, max=50.0)

    exponent2 = zeta[:, None] + eta[:, None] * log_k
    exponent2 = ops.clamp(exponent2, min=-50.0, max=50.0)

    t1 = ops.exp(exponent1)  # (B, M)
    t2 = ops.exp(exponent2)  # (B, M)

    pred = t1 + t2  # (B, M)

    # Jacobian: (B, M, 6)
    # t1 = exp(exponent1), so d(t1)/d(param) = t1 * d(exponent1)/d(param)
    # t2 = exp(exponent2), so d(t2)/d(param) = t2 * d(exponent2)/d(param)
    # exponent1 = alpha + beta_k*log_k + beta_n*log_n + gamma*log_k*log_n
    # exponent2 = zeta + eta*log_k

    zeros = pred * 0.0  # (B, M)

    d_alpha = t1                           # d/d_alpha: t1 * 1
    d_beta_k = t1 * log_k                  # d/d_beta_k: t1 * log_k
    d_beta_n = t1 * log_n                  # d/d_beta_n: t1 * log_n
    d_gamma = t1 * log_k * log_n           # d/d_gamma: t1 * log_k * log_n
    d_zeta = zeros + t2                    # d/d_zeta: t2 * 1
    d_eta = t2 * log_k                     # d/d_eta: t2 * log_k

    jac = ops.stack([d_alpha, d_beta_k, d_beta_n, d_gamma, d_zeta, d_eta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


LAW_REGISTRY = {"sl_1": sl_1}
PARAM_COUNTS = {"sl_1": 6}
# Parameter bounds: [alpha, beta_k, beta_n, gamma, zeta, eta]
# Data ranges: log(n) in [7.6, 13.2], log(k) in [3.5, 6.2], loss in [0.24, 0.63]
# Each exp term ~ 0.1-0.5 => exponents ~ -2.3 to -0.7
# Wide bounds to allow the optimizer freedom while preventing overflow
PARAM_BOUNDS = {
    "sl_1": [(-20, 20), (-5, 5), (-5, 5), (-1, 1), (-20, 20), (-5, 5)],
}
