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

    term1 = A[:, None] / (N[None, :] ** alpha[:, None])
    term2 = Bcoef[:, None] / (D[None, :] ** beta[:, None])
    term3 = Ecoef[:, None] * (
        (U[None, :] ** gamma[:, None]) * (N[None, :] ** delta[:, None])
    )

    pred = term1 + term2 + term3
    return pred[0] if pred.shape[0] == 1 else pred

# Scaling law 2:
#   a + b * U^p + c * N^q + d * D^r
# theta: [a, b, c, d, p, q, r]
def sl_2(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
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

    term1 = a0[:, None]
    term2 = bcoef[:, None] * (U[None, :] ** p[:, None])
    term3 = ccoef[:, None] * (N[None, :] ** q[:, None])
    term4 = dcoef[:, None] * (D[None, :] ** r[:, None])

    pred = term1 + term2 + term3 + term4
    return pred[0] if pred.shape[0] == 1 else pred

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

    U_b = U[None, :]
    N_b = N[None, :]
    D_b = D[None, :]

    # avoid divide-by-zero
    U_D = ops.clamp_min(U_b, _EPS)

    R_D = D_b / U_D - 1.0

    U_N = ops.minimum(rho[:, None] * U_D, N_b)
    U_N = ops.clamp_min(U_N, _EPS)

    R_N = ops.clamp_min(N_b / U_N - 1.0, 0.0)

    eff_N = U_N + tau_N[:, None] * U_N * (1.0 - ops.exp(-R_N / tau_N[:, None]))
    eff_D = U_D + tau_D[:, None] * U_D * (1.0 - ops.exp(-R_D / tau_D[:, None]))

    eff_N = ops.clamp_min(eff_N, _EPS)
    eff_D = ops.clamp_min(eff_D, _EPS)

    pred = (
        A[:, None] / (eff_N ** alpha[:, None])
        + Bcoef[:, None] / (eff_D ** alpha[:, None])
        + C[:, None]
    )

    return pred[0] if pred.shape[0] == 1 else pred


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

    scale = s[:, None] * U_n * (M_n ** d[:, None])
    scale = ops.clamp_min(scale, _EPS)

    q = T_n / scale
    T_eff_n = T_n / (1.0 + q)
    T_eff_n = ops.clamp_min(T_eff_n, _EPS)

    pred = (
        L0[:, None]
        + A[:, None] * (M_n ** (-a[:, None]))
        + Bcoef[:, None] * (T_eff_n ** (-b[:, None]))
    )

    return pred[0] if pred.shape[0] == 1 else pred

# Scaling law 5:
#   L = A / N^alpha + B / D_eff^beta + E
# where
#   D_eff = U^gamma * D^(1 - gamma)
#
# theta: [A, alpha, B, beta, E, gamma]
def sl_5(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
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

    D_eff = (U_b ** gamma[:, None]) * (D_b ** (1.0 - gamma[:, None]))
    D_eff = ops.clamp_min(D_eff, _EPS)

    pred = (
        A[:, None] / (N_b ** alpha[:, None])
        + Bcoef[:, None] / (D_eff ** beta[:, None])
        + E[:, None]
    )

    return pred[0] if pred.shape[0] == 1 else pred

# Scaling law 6 (8p): Chinchilla + repeat-penalty factor
#   R = D / U;  factor = 1 + C * max(R - 1, 0)^c * N^d
#   D_eff = D / factor
#   loss = E + A * N^(-alpha) + B * D_eff^(-beta)
# theta: [E, A, alpha, B, beta, C, c, d]
def sl_6(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
    X = ops.asarray(X, atleast_2d=True)
    theta = ops.asarray(theta, atleast_2d=True)
    U, N, D = X[:, 0], X[:, 1], X[:, 2]
    E = theta[:, 0]
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
    R = D_b / U_b
    repeat_excess = ops.clamp_min(R - 1.0, 0.0)
    factor = 1.0 + Cc[:, None] * (repeat_excess ** c[:, None]) * (N_b ** d[:, None])
    D_eff = D_b / ops.clamp_min(factor, _EPS)
    D_eff = ops.clamp_min(D_eff, _EPS)
    pred = E[:, None] + A[:, None] * (N_b ** (-alpha[:, None])) + Bcoef[:, None] * (D_eff ** (-beta[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 7 (7p): Multiplicative (N*U) product + additive terms
#   loss = L0 + A * (N * U)^alpha_pu + B * D^alpha_t + C * N^alpha_p
# theta: [L0, A, alpha_pu, B, alpha_t, C, alpha_p]
def sl_7(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
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
    pred = (L0[:, None]
            + A[:, None] * (NU ** alpha_pu[:, None])
            + Bcoef[:, None] * (D_b ** alpha_t[:, None])
            + Cc[:, None] * (N_b ** alpha_p[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


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
    vocab_ratio = xp.log(U_b / D_b + 1.0)
    abs_vr = ops.clamp_min(xp.abs(vocab_ratio) if hasattr(xp, 'abs') else ops.maximum(vocab_ratio, -vocab_ratio), _EPS)
    pred = (a0[:, None]
            + b0[:, None] / (D_b ** alpha[:, None])
            + c0[:, None] / (N_b ** beta[:, None])
            + d0[:, None] * (abs_vr ** gamma[:, None]))
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 9 (7p): Multiplicative data-quality modulation
#   loss = A / N^alpha + B / D^beta * (1 + C / U^gamma) + L_inf
# theta: [A, alpha, B, beta, C, gamma, L_inf]
def sl_9(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
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
    data_term = Bcoef[:, None] / (D_b ** beta[:, None]) * (1.0 + Cc[:, None] / (U_b ** gamma[:, None]))
    pred = A[:, None] / (N_b ** alpha[:, None]) + data_term + L_inf[:, None]
    return pred[0] if pred.shape[0] == 1 else pred


# Scaling law 10 (7p): Generalized mean (Lq-norm) data term
#   loss = L0 + A * N^(-a) + B * (D^(-b*q) + (k*U)^(-b*q))^(1/q)
# theta: [L0, A, B, a, b, k, q]
def sl_10(theta, X, backend: Literal["numpy", "jax", "torch"] = "jax"):
    ops = utils.get_ops(backend)
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
    q_safe = ops.clamp_min(q, _EPS)
    bq = b[:, None] * q_safe[:, None]
    t_D = D_b ** (-bq)
    kU = ops.clamp_min(k[:, None] * U_b, _EPS)
    t_U = kU ** (-bq)
    gm = (t_D + t_U) ** (1.0 / q_safe[:, None])
    pred = L0[:, None] + A[:, None] * (N_b ** (-a[:, None])) + Bcoef[:, None] * gm
    return pred[0] if pred.shape[0] == 1 else pred


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