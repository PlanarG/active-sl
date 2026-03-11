from typing import Literal

import benchmark.dataset.utils as utils

_EPS = 1e-12
_M_REF = 1.0
_T_REF = 1.0
_U_REF = 1.0

# Scaling law 1:
#   A / N^alpha + B / D^beta + E * (U^gamma * N^delta)
# theta: [A, alpha, B, beta, E, gamma, delta]
def sl_1(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    # X: (M, 3)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]

    # theta: (B, 7)
    A = theta[:, 0]
    alpha = theta[:, 1]
    Bcoef = theta[:, 2]
    beta = theta[:, 3]
    Ecoef = theta[:, 4]
    gamma = theta[:, 5]
    delta = theta[:, 6]

    N_b = N[None, :]  # (1, M)
    D_b = D[None, :]
    U_b = U[None, :]

    log_N = xp.log(ops.clamp_min(N_b, _EPS))
    log_D = xp.log(ops.clamp_min(D_b, _EPS))
    log_U = xp.log(ops.clamp_min(U_b, _EPS))

    N_neg_alpha = N_b ** (-alpha[:, None])          # (B, M)
    D_neg_beta = D_b ** (-beta[:, None])             # (B, M)
    U_gamma = U_b ** gamma[:, None]                  # (B, M)
    N_delta = N_b ** delta[:, None]                  # (B, M)

    term1 = A[:, None] * N_neg_alpha                 # (B, M)
    term2 = Bcoef[:, None] * D_neg_beta              # (B, M)
    term3 = Ecoef[:, None] * U_gamma * N_delta       # (B, M)

    pred = term1 + term2 + term3

    # Jacobian: (B, M, 7)
    d_A = N_neg_alpha                                # ∂/∂A = N^(-alpha)
    d_alpha = -term1 * log_N                         # ∂/∂alpha = -A*N^(-alpha)*log(N)
    d_B = D_neg_beta                                 # ∂/∂B = D^(-beta)
    d_beta = -term2 * log_D                          # ∂/∂beta = -B*D^(-beta)*log(D)
    d_E = U_gamma * N_delta                          # ∂/∂E = U^gamma * N^delta
    d_gamma = term3 * log_U                          # ∂/∂gamma = term3 * log(U)
    d_delta = term3 * log_N                          # ∂/∂delta = term3 * log(N)

    jac = ops.stack([d_A, d_alpha, d_B, d_beta, d_E, d_gamma, d_delta], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac

# Scaling law 2:
#   a + b * U^p + c * N^q + d * D^r
# theta: [a, b, c, d, p, q, r]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    # X: (M, 3)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]

    # theta: (B, 7)
    a0 = theta[:, 0]
    bcoef = theta[:, 1]
    ccoef = theta[:, 2]
    dcoef = theta[:, 3]
    p = theta[:, 4]
    q = theta[:, 5]
    r = theta[:, 6]

    U_b = U[None, :]
    N_b = N[None, :]
    D_b = D[None, :]

    log_U = xp.log(ops.clamp_min(U_b, _EPS))
    log_N = xp.log(ops.clamp_min(N_b, _EPS))
    log_D = xp.log(ops.clamp_min(D_b, _EPS))

    U_p = U_b ** p[:, None]        # (B, M)
    N_q = N_b ** q[:, None]        # (B, M)
    D_r = D_b ** r[:, None]        # (B, M)

    term2 = bcoef[:, None] * U_p
    term3 = ccoef[:, None] * N_q
    term4 = dcoef[:, None] * D_r

    pred = a0[:, None] + term2 + term3 + term4

    # Jacobian: (B, M, 7)
    ones = pred * 0.0 + 1.0
    d_a = ones                         # ∂/∂a = 1
    d_b = U_p                          # ∂/∂b = U^p
    d_c = N_q                          # ∂/∂c = N^q
    d_d = D_r                          # ∂/∂d = D^r
    d_p = term2 * log_U                # ∂/∂p = b*U^p*log(U)
    d_q = term3 * log_N                # ∂/∂q = c*N^q*log(N)
    d_r = term4 * log_D                # ∂/∂r = d*D^r*log(D)

    jac = ops.stack([d_a, d_b, d_c, d_d, d_p, d_q, d_r], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac

# Scaling law 3 (data-constrained style):
#   loss = A / eff_N^alpha + B / eff_D^alpha + C
# where
#   U_D = U
#   R_D = D / U_D - 1
#   U_N = min(rho * U_D, N)
#   R_N = max(N / U_N - 1, 0)
#   eff_N = U_N + tau_N * U_N * (1 - exp(-R_N / tau_N))
#   eff_D = U_D + tau_D * U_D * (1 - exp(-R_D / tau_D))
#
# theta: [A, tau_N, B, tau_D, alpha, C, rho]
def sl_3(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    # X: (M, 3)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]

    # theta: (B, 7)
    A = theta[:, 0]
    tau_N = theta[:, 1]
    Bcoef = theta[:, 2]
    tau_D = theta[:, 3]
    alpha = theta[:, 4]
    C = theta[:, 5]
    rho = theta[:, 6]

    U_b = U[None, :]   # (1, M)
    N_b = N[None, :]
    D_b = D[None, :]

    # avoid divide-by-zero
    U_D = ops.clamp_min(U_b, _EPS)

    R_D = D_b / U_D - 1.0                            # (1, M) or (B, M)

    rho_U_D = rho[:, None] * U_D                      # (B, M)
    U_N_raw = ops.minimum(rho_U_D, N_b)               # (B, M)
    U_N = ops.clamp_min(U_N_raw, _EPS)

    R_N_raw = N_b / U_N - 1.0
    R_N = ops.clamp_min(R_N_raw, 0.0)                 # (B, M)

    tau_N_b = tau_N[:, None]                           # (B, 1)
    tau_D_b = tau_D[:, None]

    exp_RN = ops.exp(-R_N / tau_N_b)                   # (B, M)
    exp_RD = ops.exp(-R_D / tau_D_b)                   # (B, M)

    eff_N_raw = U_N + tau_N_b * U_N * (1.0 - exp_RN)   # (B, M)
    eff_D_raw = U_D + tau_D_b * U_D * (1.0 - exp_RD)   # (B, M)

    # Masks for clamp: derivative is 0 where clamp is active
    mask_eff_N = (eff_N_raw > _EPS) * 1.0               # (B, M)
    mask_eff_D = (eff_D_raw > _EPS) * 1.0               # (B, M)

    eff_N = ops.clamp_min(eff_N_raw, _EPS)
    eff_D = ops.clamp_min(eff_D_raw, _EPS)

    log_eff_N = xp.log(ops.clamp_min(eff_N, _EPS))
    log_eff_D = xp.log(ops.clamp_min(eff_D, _EPS))

    eff_N_neg_alpha = eff_N ** (-alpha[:, None])       # (B, M)
    eff_D_neg_alpha = eff_D ** (-alpha[:, None])       # (B, M)

    termN = A[:, None] * eff_N_neg_alpha               # (B, M)
    termD = Bcoef[:, None] * eff_D_neg_alpha           # (B, M)

    pred = termN + termD + C[:, None]

    # --- Jacobian ---
    # ∂pred/∂A = eff_N^(-alpha)
    d_A = eff_N_neg_alpha

    # ∂pred/∂B = eff_D^(-alpha)
    d_B = eff_D_neg_alpha

    # ∂pred/∂alpha = -termN * log(eff_N) - termD * log(eff_D)
    d_alpha = -termN * log_eff_N - termD * log_eff_D

    # ∂pred/∂C = 1
    ones = pred * 0.0 + 1.0
    d_C = ones

    # For tau_N: ∂pred/∂tau_N = ∂pred/∂eff_N * ∂eff_N/∂tau_N
    # ∂pred/∂eff_N = -alpha * A * eff_N^(-alpha-1) = -alpha * termN / eff_N
    dpred_deffN = -alpha[:, None] * termN / eff_N      # (B, M)

    # ∂eff_N/∂tau_N = U_N * (1 - exp(-R_N/tau_N) - (R_N/tau_N)*exp(-R_N/tau_N))
    #               = U_N * (1 - exp_RN - (R_N/tau_N)*exp_RN)
    #               = U_N * (1 - exp_RN*(1 + R_N/tau_N))
    deffN_dtauN = U_N * (1.0 - exp_RN * (1.0 + R_N / tau_N_b))  # (B, M)
    d_tau_N = dpred_deffN * deffN_dtauN * mask_eff_N

    # For tau_D: ∂pred/∂tau_D = ∂pred/∂eff_D * ∂eff_D/∂tau_D
    dpred_deffD = -alpha[:, None] * termD / eff_D      # (B, M)
    deffD_dtauD = U_D * (1.0 - exp_RD * (1.0 + R_D / tau_D_b))  # (B, M)
    d_tau_D = dpred_deffD * deffD_dtauD * mask_eff_D

    # For rho: ∂pred/∂rho = ∂pred/∂eff_N * ∂eff_N/∂U_N * ∂U_N/∂rho
    # + ∂pred/∂eff_N * ∂eff_N/∂R_N * ∂R_N/∂U_N * ∂U_N/∂rho
    #
    # U_N = min(rho*U_D, N). ∂U_N/∂rho = U_D when rho*U_D < N, else 0.
    # mask: 1 where rho*U_D < N (i.e., the min selects rho*U_D)
    mask_rho = (rho_U_D < N_b) * 1.0                  # (B, M), 1 or 0

    # ∂U_N/∂rho = U_D * mask_rho
    dUN_drho = U_D * mask_rho                          # (B, M)

    # ∂eff_N/∂U_N via chain rule (U_N appears in eff_N, R_N):
    # eff_N = U_N * (1 + tau_N * (1 - exp(-R_N/tau_N)))
    # R_N = max(N/U_N - 1, 0)
    # ∂R_N/∂U_N = -N/U_N^2 when R_N > 0, else 0
    mask_RN = (R_N_raw > 0.0) * 1.0                   # (B, M)
    dRN_dUN = -N_b / (U_N ** 2) * mask_RN             # (B, M)

    # ∂eff_N/∂U_N (direct, holding R_N constant):
    # = 1 + tau_N*(1 - exp_RN)
    deffN_dUN_direct = 1.0 + tau_N_b * (1.0 - exp_RN)  # (B, M)

    # ∂eff_N/∂R_N = tau_N * U_N * (R_N/tau_N derivative of (1-exp(-R_N/tau_N)))
    # = tau_N * U_N * (1/tau_N)*exp(-R_N/tau_N) = U_N * exp_RN
    deffN_dRN = U_N * exp_RN                           # (B, M)

    # total ∂eff_N/∂U_N = deffN_dUN_direct + deffN_dRN * dRN_dUN
    deffN_dUN_total = deffN_dUN_direct + deffN_dRN * dRN_dUN  # (B, M)

    d_rho = dpred_deffN * deffN_dUN_total * dUN_drho * mask_eff_N

    # order: [A, tau_N, B, tau_D, alpha, C, rho]
    jac = ops.stack([d_A, d_tau_N, d_B, d_tau_D, d_alpha, d_C, d_rho], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 4:
#   L0 + A * M_n^(-a) + B * T_eff_n^(-b)
# where
#   M_n = max(M / _M_REF, _EPS)
#   T_n = max(T / _T_REF, _EPS)
#   U_n = max(U / _U_REF, _EPS)
#   q = T_n / max(s * U_n * M_n^d, _EPS)
#   T_eff_n = T_n / (1 + q)
#
# theta: [L0, A, a, B, b, s, d]
def sl_4(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    # X: (M, 3)
    U, M, T = X[:, 0], X[:, 1], X[:, 2]

    # theta: (B, 7)
    L0 = theta[:, 0]
    A = theta[:, 1]
    a = theta[:, 2]
    Bcoef = theta[:, 3]
    b = theta[:, 4]
    s = theta[:, 5]
    d = theta[:, 6]

    U_b = U[None, :]
    M_b = M[None, :]
    T_b = T[None, :]

    M_n = ops.clamp_min(M_b / _M_REF, _EPS)
    T_n = ops.clamp_min(T_b / _T_REF, _EPS)
    U_n = ops.clamp_min(U_b / _U_REF, _EPS)

    log_M_n = xp.log(ops.clamp_min(M_n, _EPS))

    scale = s[:, None] * U_n * (M_n ** d[:, None])     # (B, M)
    scale = ops.clamp_min(scale, _EPS)

    q_val = T_n / scale                                 # (B, M)
    denom = 1.0 + q_val                                 # (B, M)
    T_eff_n = T_n / denom                               # (B, M)
    T_eff_n = ops.clamp_min(T_eff_n, _EPS)

    log_T_eff_n = xp.log(ops.clamp_min(T_eff_n, _EPS))

    Mn_neg_a = M_n ** (-a[:, None])                     # (B, M)
    Teff_neg_b = T_eff_n ** (-b[:, None])               # (B, M)

    termM = A[:, None] * Mn_neg_a                       # (B, M)
    termT = Bcoef[:, None] * Teff_neg_b                 # (B, M)

    pred = L0[:, None] + termM + termT

    # --- Jacobian ---
    ones = pred * 0.0 + 1.0

    # ∂/∂L0 = 1
    d_L0 = ones

    # ∂/∂A = M_n^(-a)
    d_A = Mn_neg_a

    # ∂/∂a = -termM * log(M_n)
    d_a = -termM * log_M_n

    # ∂/∂B = T_eff_n^(-b)
    d_B = Teff_neg_b

    # ∂/∂b = -termT * log(T_eff_n)
    d_b = -termT * log_T_eff_n

    # For s, d: need ∂pred/∂T_eff_n * ∂T_eff_n/∂(s or d)
    # ∂pred/∂T_eff_n = -b * B * T_eff_n^(-b-1) = -b * termT / T_eff_n
    dpred_dTeff = -b[:, None] * termT / T_eff_n         # (B, M)

    # T_eff_n = T_n / (1 + T_n/scale) = T_n * scale / (scale + T_n)
    # ∂T_eff_n/∂scale = T_n * (scale + T_n - scale) / (scale + T_n)^2
    #                 = T_n^2 / (scale + T_n)^2
    # But scale + T_n = scale * denom, and T_eff_n = T_n / denom, so:
    # T_n^2 / (scale * denom)^2 = (T_n/denom)^2 / scale^2 = ... let's do directly:
    # ∂T_eff_n/∂scale = T_n^2 / (scale + T_n)^2
    scale_plus_Tn = scale + T_n
    dTeff_dscale = T_n ** 2 / (scale_plus_Tn ** 2)      # (B, M)

    # scale = s * U_n * M_n^d
    # ∂scale/∂s = U_n * M_n^d = scale / s[:, None]
    dscale_ds = scale / s[:, None]                       # (B, M)

    # ∂scale/∂d = s * U_n * M_n^d * log(M_n) = scale * log(M_n)
    dscale_dd = scale * log_M_n                          # (B, M)

    d_s = dpred_dTeff * dTeff_dscale * dscale_ds
    d_d = dpred_dTeff * dTeff_dscale * dscale_dd

    # order: [L0, A, a, B, b, s, d]
    jac = ops.stack([d_L0, d_A, d_a, d_B, d_b, d_s, d_d], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac

# Scaling law 5:
#   L = A / N^alpha + B / D_eff^beta + E
# where
#   D_eff = U^gamma * D^(1 - gamma)
#
# theta: [A, alpha, B, beta, E, gamma]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)

    # X: (M, 3)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]

    # theta: (B, 6)
    A = theta[:, 0]
    alpha = theta[:, 1]
    Bcoef = theta[:, 2]
    beta = theta[:, 3]
    E = theta[:, 4]
    gamma = theta[:, 5]

    U_b = ops.clamp_min(U[None, :], _EPS)
    N_b = ops.clamp_min(N[None, :], _EPS)
    D_b = ops.clamp_min(D[None, :], _EPS)

    log_U = xp.log(ops.clamp_min(U_b, _EPS))
    log_D = xp.log(ops.clamp_min(D_b, _EPS))
    log_N = xp.log(ops.clamp_min(N_b, _EPS))

    D_eff = (U_b ** gamma[:, None]) * (D_b ** (1.0 - gamma[:, None]))
    D_eff = ops.clamp_min(D_eff, _EPS)

    log_D_eff = xp.log(ops.clamp_min(D_eff, _EPS))

    N_neg_alpha = N_b ** (-alpha[:, None])               # (B, M)
    D_eff_neg_beta = D_eff ** (-beta[:, None])           # (B, M)

    termN = A[:, None] * N_neg_alpha
    termD = Bcoef[:, None] * D_eff_neg_beta

    pred = termN + termD + E[:, None]

    # --- Jacobian ---
    ones = pred * 0.0 + 1.0

    # ∂/∂A = N^(-alpha)
    d_A = N_neg_alpha

    # ∂/∂alpha = -termN * log(N)
    d_alpha = -termN * log_N

    # ∂/∂B = D_eff^(-beta)
    d_B = D_eff_neg_beta

    # ∂/∂beta = -termD * log(D_eff)
    d_beta = -termD * log_D_eff

    # ∂/∂E = 1
    d_E = ones

    # ∂/∂gamma: D_eff = U^gamma * D^(1-gamma)
    # log(D_eff) = gamma*log(U) + (1-gamma)*log(D)
    # ∂log(D_eff)/∂gamma = log(U) - log(D) = log(U/D)
    # ∂D_eff/∂gamma = D_eff * (log(U) - log(D))
    # ∂pred/∂gamma = ∂pred/∂D_eff * ∂D_eff/∂gamma
    # ∂pred/∂D_eff = -beta * B * D_eff^(-beta-1) = -beta * termD / D_eff
    dpred_dDeff = -beta[:, None] * termD / D_eff
    dDeff_dgamma = D_eff * (log_U - log_D)
    d_gamma = dpred_dDeff * dDeff_dgamma

    # order: [A, alpha, B, beta, E, gamma]
    jac = ops.stack([d_A, d_alpha, d_B, d_beta, d_E, d_gamma], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac

# Scaling law 6 (8p): Chinchilla + repeat-penalty factor
#   R = D / U;  factor = 1 + C * max(R - 1, 0)^c * N^d
#   D_eff = D / factor
#   loss = E + A * N^(-alpha) + B * D_eff^(-beta)
# theta: [E, A, alpha, B, beta, C, c, d]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]
    Ep = theta[:, 0]
    A = theta[:, 1]
    alpha = theta[:, 2]
    Bcoef = theta[:, 3]
    beta = theta[:, 4]
    Cc = theta[:, 5]
    c = theta[:, 6]
    d = theta[:, 7]

    U_b = ops.clamp_min(U[None, :], _EPS)
    N_b = ops.clamp_min(N[None, :], _EPS)
    D_b = ops.clamp_min(D[None, :], _EPS)

    log_N = xp.log(ops.clamp_min(N_b, _EPS))

    R = D_b / U_b                                     # (1, M)
    repeat_excess = ops.clamp_min(R - 1.0, 0.0)       # (1, M)
    log_re = xp.log(ops.clamp_min(repeat_excess, _EPS))

    re_c = repeat_excess ** c[:, None]                 # (B, M)
    N_d = N_b ** d[:, None]                            # (B, M)
    penalty = Cc[:, None] * re_c * N_d                 # (B, M)
    factor = 1.0 + penalty                             # (B, M)
    factor_safe = ops.clamp_min(factor, _EPS)

    D_eff = D_b / factor_safe                          # (B, M)
    D_eff = ops.clamp_min(D_eff, _EPS)

    log_D_eff = xp.log(ops.clamp_min(D_eff, _EPS))

    N_neg_alpha = N_b ** (-alpha[:, None])             # (B, M)
    D_eff_neg_beta = D_eff ** (-beta[:, None])         # (B, M)

    termN = A[:, None] * N_neg_alpha                   # (B, M)
    termD = Bcoef[:, None] * D_eff_neg_beta            # (B, M)

    pred = Ep[:, None] + termN + termD

    # --- Jacobian ---
    ones = pred * 0.0 + 1.0

    # ∂/∂E = 1
    d_E = ones

    # ∂/∂A = N^(-alpha)
    d_A = N_neg_alpha

    # ∂/∂alpha = -termN * log(N)
    d_alpha = -termN * log_N

    # ∂/∂B = D_eff^(-beta)
    d_B = D_eff_neg_beta

    # ∂/∂beta = -termD * log(D_eff)
    d_beta = -termD * log_D_eff

    # For C, c, d: need ∂pred/∂D_eff * ∂D_eff/∂factor * ∂factor/∂param
    # ∂pred/∂D_eff = -beta * B * D_eff^(-beta-1) = -beta * termD / D_eff
    dpred_dDeff = -beta[:, None] * termD / D_eff       # (B, M)

    # D_eff = D / factor  =>  ∂D_eff/∂factor = -D / factor^2 = -D_eff / factor
    dDeff_dfactor = -D_eff / factor_safe               # (B, M)

    dpred_dfactor = dpred_dDeff * dDeff_dfactor        # (B, M)

    # factor = 1 + C * re^c * N^d
    # ∂factor/∂C = re^c * N^d = penalty / Cc[:, None]
    # But Cc could be 0, so compute directly:
    dfactor_dC = re_c * N_d                            # (B, M)

    # ∂factor/∂c = C * re^c * log(re) * N^d = penalty * log(re)
    dfactor_dc = penalty * log_re                      # (B, M)

    # ∂factor/∂d = C * re^c * N^d * log(N) = penalty * log(N)
    dfactor_dd = penalty * log_N                       # (B, M)

    d_Cc = dpred_dfactor * dfactor_dC
    d_c = dpred_dfactor * dfactor_dc
    d_d = dpred_dfactor * dfactor_dd

    # order: [E, A, alpha, B, beta, C, c, d]
    jac = ops.stack([d_E, d_A, d_alpha, d_B, d_beta, d_Cc, d_c, d_d], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 7 (7p): Multiplicative (N*U) product + additive terms
#   loss = L0 + A * (N * U)^alpha_pu + B * D^alpha_t + C * N^alpha_p
# theta: [L0, A, alpha_pu, B, alpha_t, C, alpha_p]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]
    L0 = theta[:, 0]
    A = theta[:, 1]
    alpha_pu = theta[:, 2]
    Bcoef = theta[:, 3]
    alpha_t = theta[:, 4]
    Cc = theta[:, 5]
    alpha_p = theta[:, 6]

    U_b = ops.clamp_min(U[None, :], _EPS)
    N_b = ops.clamp_min(N[None, :], _EPS)
    D_b = ops.clamp_min(D[None, :], _EPS)
    NU = ops.clamp_min(N_b * U_b, _EPS)

    log_NU = xp.log(ops.clamp_min(NU, _EPS))
    log_D = xp.log(ops.clamp_min(D_b, _EPS))
    log_N = xp.log(ops.clamp_min(N_b, _EPS))

    NU_apu = NU ** alpha_pu[:, None]                   # (B, M)
    D_at = D_b ** alpha_t[:, None]                     # (B, M)
    N_ap = N_b ** alpha_p[:, None]                     # (B, M)

    term2 = A[:, None] * NU_apu
    term3 = Bcoef[:, None] * D_at
    term4 = Cc[:, None] * N_ap

    pred = L0[:, None] + term2 + term3 + term4

    # --- Jacobian ---
    ones = pred * 0.0 + 1.0

    d_L0 = ones
    d_A = NU_apu
    d_alpha_pu = term2 * log_NU
    d_B = D_at
    d_alpha_t = term3 * log_D
    d_C = N_ap
    d_alpha_p = term4 * log_N

    # order: [L0, A, alpha_pu, B, alpha_t, C, alpha_p]
    jac = ops.stack([d_L0, d_A, d_alpha_pu, d_B, d_alpha_t, d_C, d_alpha_p], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 8 (7p): Log-ratio vocabulary saturation
#   vocab_ratio = log(U / D + 1)
#   loss = a + b / D^alpha + c / N^beta + d * |vocab_ratio|^gamma
# theta: [a, b, c, d, alpha, beta, gamma]
def sl_8(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]
    a0 = theta[:, 0]
    b0 = theta[:, 1]
    c0 = theta[:, 2]
    d0 = theta[:, 3]
    alpha = theta[:, 4]
    beta = theta[:, 5]
    gamma = theta[:, 6]

    U_b = ops.clamp_min(U[None, :], _EPS)
    N_b = ops.clamp_min(N[None, :], _EPS)
    D_b = ops.clamp_min(D[None, :], _EPS)

    log_D = xp.log(ops.clamp_min(D_b, _EPS))
    log_N = xp.log(ops.clamp_min(N_b, _EPS))

    D_neg_alpha = D_b ** (-alpha[:, None])             # (B, M)
    N_neg_beta = N_b ** (-beta[:, None])               # (B, M)

    termD = b0[:, None] * D_neg_alpha                  # b/D^alpha
    termN = c0[:, None] * N_neg_beta                   # c/N^beta

    vocab_ratio = xp.log(U_b / D_b + 1.0)             # (1, M) or (B, M)
    abs_vr = ops.clamp_min(xp.abs(vocab_ratio) if hasattr(xp, 'abs') else ops.maximum(vocab_ratio, -vocab_ratio), _EPS)
    log_abs_vr = xp.log(ops.clamp_min(abs_vr, _EPS))

    abs_vr_gamma = abs_vr ** gamma[:, None]            # (B, M)
    termV = d0[:, None] * abs_vr_gamma                 # d*|vr|^gamma

    pred = a0[:, None] + termD + termN + termV

    # --- Jacobian ---
    ones = pred * 0.0 + 1.0

    d_a = ones
    d_b = D_neg_alpha                                  # ∂/∂b = 1/D^alpha
    d_c = N_neg_beta                                   # ∂/∂c = 1/N^beta
    d_d = abs_vr_gamma                                 # ∂/∂d = |vr|^gamma
    d_alpha = -termD * log_D                           # ∂/∂alpha = -b*D^(-alpha)*log(D)
    d_beta = -termN * log_N                            # ∂/∂beta = -c*N^(-beta)*log(N)
    d_gamma = termV * log_abs_vr                       # ∂/∂gamma = d*|vr|^gamma*log(|vr|)

    # order: [a, b, c, d, alpha, beta, gamma]
    jac = ops.stack([d_a, d_b, d_c, d_d, d_alpha, d_beta, d_gamma], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 9 (7p): Multiplicative data-quality modulation
#   loss = A / N^alpha + B / D^beta * (1 + C / U^gamma) + L_inf
# theta: [A, alpha, B, beta, C, gamma, L_inf]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]
    A = theta[:, 0]
    alpha = theta[:, 1]
    Bcoef = theta[:, 2]
    beta = theta[:, 3]
    Cc = theta[:, 4]
    gamma = theta[:, 5]
    L_inf = theta[:, 6]

    U_b = ops.clamp_min(U[None, :], _EPS)
    N_b = ops.clamp_min(N[None, :], _EPS)
    D_b = ops.clamp_min(D[None, :], _EPS)

    log_N = xp.log(ops.clamp_min(N_b, _EPS))
    log_D = xp.log(ops.clamp_min(D_b, _EPS))
    log_U = xp.log(ops.clamp_min(U_b, _EPS))

    N_neg_alpha = N_b ** (-alpha[:, None])             # (B, M)
    D_neg_beta = D_b ** (-beta[:, None])               # (B, M)
    U_neg_gamma = U_b ** (-gamma[:, None])             # (B, M)

    termN = A[:, None] * N_neg_alpha                   # A/N^alpha
    quality = 1.0 + Cc[:, None] * U_neg_gamma          # 1 + C/U^gamma
    data_base = Bcoef[:, None] * D_neg_beta            # B/D^beta
    data_term = data_base * quality                    # B/D^beta * (1 + C/U^gamma)

    pred = termN + data_term + L_inf[:, None]

    # --- Jacobian ---
    ones = pred * 0.0 + 1.0

    # ∂/∂A = N^(-alpha)
    d_A = N_neg_alpha

    # ∂/∂alpha = -termN * log(N)
    d_alpha = -termN * log_N

    # ∂/∂B = D^(-beta) * quality
    d_B = D_neg_beta * quality

    # ∂/∂beta = -data_term * log(D)
    d_beta = -data_term * log_D

    # ∂/∂C = B/D^beta * U^(-gamma) = data_base * U^(-gamma)
    d_C = data_base * U_neg_gamma

    # ∂/∂gamma = B/D^beta * C * (-U^(-gamma)) * log(U)
    #          = -data_base * C * U^(-gamma) * log(U)
    d_gamma = -data_base * Cc[:, None] * U_neg_gamma * log_U

    # ∂/∂L_inf = 1
    d_Linf = ones

    # order: [A, alpha, B, beta, C, gamma, L_inf]
    jac = ops.stack([d_A, d_alpha, d_B, d_beta, d_C, d_gamma, d_Linf], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


# Scaling law 10 (7p): Generalized mean (Lq-norm) data term
#   loss = L0 + A * N^(-a) + B * (D^(-b*q) + (k*U)^(-b*q))^(1/q)
# theta: [L0, A, B, a, b, k, q]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    xp = ops.xp
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]
    L0 = theta[:, 0]
    A = theta[:, 1]
    Bcoef = theta[:, 2]
    a = theta[:, 3]
    b = theta[:, 4]
    k = theta[:, 5]
    q = theta[:, 6]

    U_b = ops.clamp_min(U[None, :], _EPS)
    N_b = ops.clamp_min(N[None, :], _EPS)
    D_b = ops.clamp_min(D[None, :], _EPS)

    log_N = xp.log(ops.clamp_min(N_b, _EPS))
    log_D = xp.log(ops.clamp_min(D_b, _EPS))

    q_safe = ops.clamp_min(q, _EPS)
    bq = b[:, None] * q_safe[:, None]                 # (B, M-broadcast) -> (B, 1)

    t_D = D_b ** (-bq)                                # (B, M)
    kU = ops.clamp_min(k[:, None] * U_b, _EPS)        # (B, M)
    log_kU = xp.log(ops.clamp_min(kU, _EPS))
    t_U = kU ** (-bq)                                 # (B, M)

    S = t_D + t_U                                     # (B, M)
    S_safe = ops.clamp_min(S, _EPS)
    log_S = xp.log(ops.clamp_min(S_safe, _EPS))

    inv_q = 1.0 / q_safe[:, None]                     # (B, 1)
    gm = S_safe ** inv_q                               # (B, M)

    N_neg_a = N_b ** (-a[:, None])                     # (B, M)

    termN = A[:, None] * N_neg_a                       # (B, M)
    termG = Bcoef[:, None] * gm                        # (B, M)

    pred = L0[:, None] + termN + termG

    # --- Jacobian ---
    ones = pred * 0.0 + 1.0

    # ∂/∂L0 = 1
    d_L0 = ones

    # ∂/∂A = N^(-a)
    d_A = N_neg_a

    # ∂/∂B = gm
    d_B = gm

    # ∂/∂a = -termN * log(N)
    d_a = -termN * log_N

    # For b, k, q we need derivatives through gm = S^(1/q)
    # Let's compute ∂gm/∂S first:
    # gm = S^(1/q)  =>  ∂gm/∂S = (1/q) * S^(1/q - 1) = gm / (q * S)
    dgm_dS = gm / (q_safe[:, None] * S_safe)          # (B, M)

    # ∂S/∂b: S = D^(-bq) + (kU)^(-bq)
    # t_D = D^(-bq), ∂t_D/∂b = -q * t_D * log(D)  (since ∂(-bq)/∂b = -q)
    # t_U = (kU)^(-bq), ∂t_U/∂b = -q * t_U * log(kU)
    dS_db = -q_safe[:, None] * (t_D * log_D + t_U * log_kU)  # (B, M)

    # Also need ∂gm/∂b via the exponent: gm = S^(1/q), S depends on b
    # ∂gm/∂b = dgm_dS * dS_db
    dgm_db = dgm_dS * dS_db
    d_b = Bcoef[:, None] * dgm_db                     # (B, M)

    # ∂S/∂k: t_U = (kU)^(-bq)
    # ∂t_U/∂k = -bq * (kU)^(-bq-1) * U = -bq * t_U / (kU) * U = -bq * t_U / k[:, None]
    # (since kU = k*U, ∂(kU)/∂k = U, and ∂t_U/∂(kU) = -bq * (kU)^(-bq-1))
    dS_dk = -bq * t_U / k[:, None]                    # (B, M)
    dgm_dk = dgm_dS * dS_dk
    d_k = Bcoef[:, None] * dgm_dk                     # (B, M)

    # ∂gm/∂q: gm = S^(1/q), where both S and the exponent 1/q depend on q.
    # Use: log(gm) = (1/q) * log(S)
    # ∂log(gm)/∂q = ∂(1/q)/∂q * log(S) + (1/q) * ∂log(S)/∂q
    #             = (-1/q^2) * log(S) + (1/q) * (1/S) * ∂S/∂q
    # ∂gm/∂q = gm * [(-1/q^2)*log(S) + (1/(q*S))*∂S/∂q]
    #
    # ∂S/∂q: t_D = D^(-bq) => ∂t_D/∂q = -b * t_D * log(D)
    #         t_U = (kU)^(-bq) => ∂t_U/∂q = -b * t_U * log(kU)
    dS_dq = -b[:, None] * (t_D * log_D + t_U * log_kU)  # (B, M)

    q2 = q_safe[:, None] ** 2
    dgm_dq = gm * (-log_S / q2 + dS_dq / (q_safe[:, None] * S_safe))
    d_q = Bcoef[:, None] * dgm_dq                     # (B, M)

    # order: [L0, A, B, a, b, k, q]
    jac = ops.stack([d_L0, d_A, d_B, d_a, d_b, d_k, d_q], axis=-1)

    if pred.shape[0] == 1:
        return pred[0], jac[0]
    return pred, jac


PARAM_BOUNDS = {
    # Dataset: U (unique_tokens) ~ 1e8–2e11, N (params) ~ 7e6–9e9,
    #          D (tokens) ~ 1e8–9e11, Loss ~ 2.3–8.1
    #
    # Bound derivation:
    #   - Decay exponents (alpha, beta, etc.): (0.05, 2.0) — physically positive,
    #     upper limit avoids numerically useless landscape regions.
    #   - Mixed/signed exponents (gamma, delta, alpha_pu, …): tight range around
    #     observed optimal, typically (-2, 0.5) for negative-decay parameters.
    #   - Coefficients for N-decay terms A/N^alpha:
    #       A_max ≈ L_max * N_min^alpha_max = 8 * (7e6)^2 ≈ 4e14 → use 1e9 with
    #       alpha restricted to ≤ 2 (optimizer stays near typical alpha ≈ 0.3–1.0).
    #   - Loss-floor constants: (-3, 10) — loss ∈ [2.3, 8.1], floor < total loss.
    #   - Structural params (tau, rho, s, k, q): derived from data ratios / physics.
    #
    # Overflow: all float64-safe; exp(-R/tau) underflows to 0 (never NaN);
    # power terms at extreme bounds are O(1e50) at worst, well below 1e308.

    # sl_1: [A, alpha, B, beta, E, gamma, delta]
    # A/N^alpha + B/D^beta + E*(U^gamma * N^delta)
    # Optimal approx: A~5e5 (alpha~0.82), B~1e5 (beta~0.56), E~12, gamma~-0.04, delta~-0.03
    # E*U^gamma*N^delta is a small "repeat-floor" term; restrict |gamma|,|delta|<=0.5
    # so the floor stays O(1–100) and E doesn't need astronomically large values.
    "sl_1": [(1e-3, 1e9), (0.05, 2.0), (1e-3, 1e8), (0.05, 2.0),
             (-100, 200), (-0.5, 0.5), (-0.5, 0.5)],

    # sl_2: [a, b, c, d, p, q, r]
    # a + b*U^p + c*N^q + d*D^r
    # Optimal approx: a~2, b~17–700, c~3400–5300, d~1e5, p~-0.14 to -0.38,
    #   q~-0.48 to -0.51, r~-0.56 to -0.57
    # Coefficients: b*U^p ~ O(1) at U~1e9, p~-0.4 → b ~ 3/U^(-0.4) ~ 3e4 max;
    #   similarly c ~ 3e5, d ~ 1e6. Use 10x margin.
    "sl_2": [(-3, 10), (-1e6, 1e6), (-1e6, 1e6), (-1e7, 1e7),
             (-1.5, 0.5), (-1.5, 0.5), (-1.5, 0.5)],

    # sl_3: [A, tau_N, B, tau_D, alpha, C, rho]
    # Muennighoff data-constrained formula with effective N/D via exponential saturation.
    # Optimal approx: A~2345, tau_N~0.07, B~14500, tau_D~31, alpha~0.45, C~2.3, rho~0.82
    # tau: dimensionless repeat counts; tau_N can be very small (<<1) or up to O(100).
    # rho: fraction U/N at crossover; rho*U_D is compared to N, range from 0.001 to 100.
    "sl_3": [(1e-3, 1e7), (1e-3, 500), (1e-3, 1e7), (1e-3, 500),
             (0.05, 2.0), (-3, 10), (1e-3, 100)],

    # sl_4: [L0, A, a, B, b, s, d]
    # L0 + A*M_n^(-a) + B*T_eff_n^(-b), T_eff_n = T_n/(1+T_n/(s*U_n*M_n^d))
    # _M_REF=_T_REF=_U_REF=1 so M_n=N~7e6–9e9, T_n=D~1e8–9e11, U_n=U~1e8–2e11.
    # Optimal approx: L0~2.5, A~4e6 (a~0.92), B~12500 (b~0.44), s~197, d~-0.13
    # s*U_n*M_n^d ~ D at crossover; s ~ D/(U*N^d) ~ 1e10/(4e9*(2.5e8)^(-0.13)) ~ 200.
    "sl_4": [(-3, 8), (1e-3, 1e11), (0.05, 2.5), (1e-3, 1e9),
             (0.05, 2.5), (1e-5, 1e7), (-2.0, 1.5)],

    # sl_5: [A, alpha, B, beta, E, gamma]
    # A/N^alpha + B/D_eff^beta + E, D_eff = U^gamma * D^(1-gamma)
    # gamma in [0,1]: D_eff is a geometric mean of U and D (U^gamma*D^(1-gamma)).
    # Optimal approx: A~173 (alpha~0.28), B~1.7e6 (beta~0.71), E~2.3, gamma~0.34
    # D_eff at gamma=0.34: D_eff ~ (4e9)^0.34*(1e10)^0.66 ~ 7e9; B/D_eff^0.71 ~ O(1).
    "sl_5": [(1e-3, 1e10), (0.05, 2.0), (1e-3, 1e10), (0.05, 2.0),
             (-3, 10), (0.0, 1.0)],

    # sl_6: [E, A, alpha, B, beta, C, c, d]
    # E + A*N^(-alpha) + B*D_eff^(-beta),
    # D_eff = D / max(1 + C*(max(D/U-1,0))^c * N^d, eps)
    # Optimal approx: E~3, A~1.6e6 (alpha~0.86), B~7e8 (beta~1.04), C~0.3, c~0.83, d~0.02
    # D/U-1 ranges 0–8999; factor = 1+C*(8999)^c*N^d; with C=0.3,c=0.83,d=0.02 → factor~700.
    # C>=0 (repeat penalty must increase factor); c in [0,2]; d in [-1,1].
    "sl_6": [(-3, 10), (1e-3, 1e11), (0.05, 2.0), (1e-3, 1e12),
             (0.05, 2.0), (0, 1e4), (0, 2.0), (-1.0, 1.0)],

    # sl_7: [L0, A, alpha_pu, B, alpha_t, C, alpha_p]
    # L0 + A*(N*U)^alpha_pu + B*D^alpha_t + C*N^alpha_p
    # N*U range: ~7e14–1.5e21; exponents alpha_pu, alpha_t, alpha_p are negative (decay).
    # Optimal approx: L0~2.4, A~1100 (alpha_pu~-0.18), B~95000 (alpha_t~-0.55),
    #   C~7e11 (alpha_p~-1.77, nearly zero contribution at typical N).
    "sl_7": [(-3, 10), (1e-3, 1e9), (-2.0, 0.5), (1e-3, 1e9),
             (-2.0, 0.5), (-1e13, 1e13), (-2.5, 0.5)],

    # sl_8: [a, b, c, d, alpha, beta, gamma]
    # a + b/D^alpha + c/N^beta + d*|log(U/D+1)|^gamma
    # vocab_ratio = log(U/D+1) in [0, ~7.5] over this dataset.
    # WARNING: as gamma->0, |vr|^gamma->1, making a and d unidentifiable (a+d = const).
    #   Restrict gamma >= 0.05 to reduce degeneracy; fix a in (-3, 10).
    # Optimal approx (non-degenerate): a~3.5–3.9, b~8000–9600, c~6400–9000,
    #   d~-1.7, alpha~0.42, beta~0.54, gamma~0.1–0.4.
    "sl_8": [(-3, 10), (-1e8, 1e8), (-1e8, 1e8), (-200, 200),
             (0.05, 2.0), (0.05, 2.0), (0.05, 5.0)],

    # sl_9: [A, alpha, B, beta, C, gamma, L_inf]
    # A/N^alpha + B/D^beta * (1 + C/U^gamma) + L_inf
    # Optimal approx: A~90 (alpha~0.22), B~16000 (beta~0.47),
    #   C~1.1e9 (gamma~1.20), L_inf~2.0
    # C/U^gamma at U_typ=4e9, gamma=1.2: C/(4e9)^1.2 ~ 1.1e9/2.4e10 ~ 0.046 (small factor).
    # C can be large because U is also large; allow up to 1e11.
    "sl_9": [(1e-3, 1e7), (0.05, 2.0), (1e-3, 1e7), (0.05, 2.0),
             (0, 1e11), (0.05, 2.0), (-3, 10)],

    # sl_10: [L0, A, B, a, b, k, q]
    # L0 + A*N^(-a) + B*(D^(-b*q) + (k*U)^(-b*q))^(1/q)
    # q: generalized-mean exponent; k: scales U to match D in the Lq-norm.
    # Optimal approx: L0~2.3, A~14500 (a~0.63), B~1900 (b~0.37), k~23, q~12–16.
    # k*U_typ=23*4e9=9e10 ~ D_typ=1e10 (same order, reasonable crossover).
    # q can be large (O(10–20)) meaning the law approaches a hard min over D and k*U.
    "sl_10": [(-3, 8), (1e-3, 1e9), (1e-3, 1e9), (0.05, 2.0),
              (0.05, 2.0), (1e-3, 1e4), (0.1, 50.0)],
}

LAW_REGISTRY = {
    "sl_1": sl_1, "sl_2": sl_2, "sl_3": sl_3, "sl_4": sl_4, "sl_5": sl_5,
    "sl_6": sl_6, "sl_7": sl_7, "sl_8": sl_8, "sl_9": sl_9, "sl_10": sl_10,
}
PARAM_COUNTS = {
    "sl_1": 7, "sl_2": 7, "sl_3": 7, "sl_4": 7, "sl_5": 6,
    "sl_6": 8, "sl_7": 7, "sl_8": 7, "sl_9": 7, "sl_10": 7,
}
