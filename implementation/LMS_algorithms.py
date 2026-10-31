"""
Adaptive MISO filtering algorithms (RLS, NLMS, and VS-LMS) and Late Fusion bucket function.

Implements the adaptive filtering stage of the PPG heart signal reconstruction pipeline:
- Primary input `d[n]`: Preprocessed PPG channel (`s[n] + m[n]`, cardiac signal + motion artifact).
- Reference input `a[n]`: 3-axis accelerometer tapped-delay vector `[a_1^T[n], a_2^T[n], a_3^T[n]]^T`.
- MISO motion artifact estimate:
      m_hat[n] = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]
- Cleared cardiac signal (a priori error / innovation output):
      e[n] = d[n] - m_hat[n] = s[n] + m[n] - w^T[n] a[n]

Includes:
1. `rls`: Recursive Least Squares (RLS) MISO adaptive filter with all hyperparameters tunable
   (`M`, forgetting factor `lam` / `lambda_`, regularization parameter `delta`, and `w_init`).
2. `nlms`: Normalized MISO LMS configured with the optimal hyperparameters obtained in Project 1
   (`M = 22`, `mu = 0.01288`, `delta = 0.5`).
3. `vs_lms`: Variable Step-Size MISO LMS configured with the optimal hyperparameters obtained in Project 1
   (`M = 18`, `mu_max = 0.06558`, `mu_min = 0.00791`, `alpha = 0.98503`, `gamma = 0.00053`, `delta = 1.0`).
4. `lms_bucket`: Bucket function that runs the selected adaptive filter (`rls`, `nlms`, or `vs_lms`)
   in parallel on `ppg1` and `ppg2` and fuses the resulting error signals via `snr_weighted_late_fusion`.
"""

import numpy as np

try:
    from .reference_construction import build_U
    from .fusion import snr_weighted_late_fusion
except ImportError:
    from reference_construction import build_U
    from fusion import snr_weighted_late_fusion

# Divergence threshold to detect numerical instability during adaptation
DIV_THR: float = 1e3

# ============================================================================
# Optimal Hyperparameters Obtained in Project 1 (5-Fold Subject-Wise GroupKFold)
# ============================================================================
OPTIMAL_NLMS_PARAMS: dict = {
    "M": 22,
    "mu": 0.01288,
    "delta": 0.5,
}
OPTIMAL_NLMS_HR_CONFIG: dict = {
    "hr_algorithm": "periodogram",
    "hr_params": {"nfft": 4096, "window_type": "hann"},
}

OPTIMAL_VS_LMS_PARAMS: dict = {
    "M": 18,
    "mu_max": 0.06557805690203233,
    "mu_min": 0.0079057537359447,
    "alpha": 0.9850297284624578,
    "gamma": 0.0005342576707258032,
    "delta": 1.0,
}
OPTIMAL_VS_LMS_HR_CONFIG: dict = {
    "hr_algorithm": "periodogram",
    "hr_params": {"nfft": 8192, "window_type": "hamming"},
}


def _prepare_regressor(acc_or_U: np.ndarray, N: int, M: int) -> np.ndarray:
    """Return the (N, 3 * M) regressor matrix U from either `acc` (3, N) or prebuilt `U`."""
    arr = np.asarray(acc_or_U, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == N and arr.shape[1] != N:
        return arr
    return build_U(arr, M)


# ============================================================================
# 1. Recursive Least Squares (RLS) MISO Adaptive Filter (All Hyperparameters Tunable)
# ============================================================================
def rls(
    d: np.ndarray,
    acc: np.ndarray,
    M: int = 12,
    lam: float = 0.99,
    delta: float = 1.0,
    lambda_: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recursive Least Squares (RLS) MISO adaptive filter.

    Minimizes the exponentially weighted deterministic least-squares cost function:
        J[n] = sum_{i=0}^{n} lam^{n-i} |d[i] - w^T[n] a[i]|^2 + delta * lam^{n+1} ||w[n]||^2

    At each sample `n = 0, ..., N - 1`:
        m_hat[n] = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]     = d[n] - m_hat[n]
        pi[n]    = P[n] a[n]
        k[n]     = pi[n] / (lam + a^T[n] pi[n])
        w[n + 1] = w[n] + k[n] * e[n]
        P[n + 1] = lam^{-1} * (P[n] - k[n] pi^T[n])

    where `P[0] = delta^{-1} I_{3M}` is the initial inverse correlation matrix.

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape `(N,)`.
    acc : np.ndarray
        Reference 3-axis accelerometer matrix of shape `(3, N)`, or prebuilt
        regressor matrix `U` of shape `(N, 3 * M)`.
    M : int, default=12
        Filter order (number of FIR taps per accelerometer axis; total taps = 3 * M).
    lam : float, default=0.99
        Exponential forgetting factor `lambda` in `(0, 1]`. Controls the effective memory
        length `N_eff = 1 / (1 - lam)` of the sample correlation matrix.
    delta : float, default=1.0
        Regularization parameter `delta > 0` initializing the inverse correlation matrix
        `P[0] = (1 / delta) * I_{3M}` (corresponding to initial correlation `Phi[0] = delta * I`).
    lambda_ : float, optional
        Alias for `lam`.
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(3 * M,)`. Defaults to zeros.

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal (a priori estimation error) `e[n] = d[n] - m_hat[n]` of shape `(N,)`.
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape `(N,)`.
    w : np.ndarray
        Final MISO filter weight vector of shape `(3 * M,)`.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    lam_val = float(lambda_ if lambda_ is not None else lam)
    if not (0.0 < lam_val <= 1.0):
        raise ValueError(f"Forgetting factor 'lam' must be in (0, 1], got {lam_val}.")
    inv_lam = 1.0 / lam_val

    delta_val = float(delta)
    if delta_val <= 0.0:
        raise ValueError(f"Regularization parameter 'delta' must be > 0, got {delta_val}.")

    P = (1.0 / delta_val) * np.eye(L, dtype=float)
    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
    e = np.empty(N, dtype=float)
    m_hat = np.empty(N, dtype=float)

    dot = np.dot
    for n in range(N):
        a_n = U[n]
        m_n = float(dot(w, a_n))
        e_n = d[n] - m_n
        m_hat[n] = m_n
        e[n] = e_n

        if not np.isfinite(e_n) or abs(e_n) > DIV_THR:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

        pi = dot(P, a_n)
        denom = lam_val + float(dot(a_n, pi))
        if not np.isfinite(denom) or denom <= 1e-12:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

        k = pi * (1.0 / denom)
        w += e_n * k
        P -= k[:, None] * pi[None, :]
        P *= inv_lam

    return e, m_hat, w


def _rls_dual_channel(
    ppg1: np.ndarray,
    ppg2: np.ndarray,
    U: np.ndarray,
    M: int = 12,
    lam: float = 0.99,
    delta: float = 1.0,
    lambda_: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run parallel MISO RLS on `ppg1` and `ppg2` sharing the inverse correlation matrix `P[n]`.

    Because both PPG channels share the exact same 3-axis accelerometer reference `U`,
    the gain vector `k[n]` and inverse correlation matrix `P[n]` depend strictly on `U`,
    `lam`, and `delta`, allowing both branches to be updated simultaneously with identical
    mathematical results to two independent calls to `rls`.
    """
    d1 = np.asarray(ppg1, dtype=float)
    d2 = np.asarray(ppg2, dtype=float)
    N = len(d1)
    L = U.shape[1]

    lam_val = float(lambda_ if lambda_ is not None else lam)
    if not (0.0 < lam_val <= 1.0):
        raise ValueError(f"Forgetting factor 'lam' must be in (0, 1], got {lam_val}.")
    inv_lam = 1.0 / lam_val

    delta_val = float(delta)
    if delta_val <= 0.0:
        raise ValueError(f"Regularization parameter 'delta' must be > 0, got {delta_val}.")

    P = (1.0 / delta_val) * np.eye(L, dtype=float)
    if w_init is None:
        w1 = np.zeros(L, dtype=float)
        w2 = np.zeros(L, dtype=float)
    else:
        w1 = np.array(w_init, dtype=float, copy=True)
        w2 = np.array(w_init, dtype=float, copy=True)

    e1 = np.empty(N, dtype=float)
    e2 = np.empty(N, dtype=float)
    m_hat1 = np.empty(N, dtype=float)
    m_hat2 = np.empty(N, dtype=float)

    dot = np.dot
    alive1 = True
    alive2 = True

    for n in range(N):
        a_n = U[n]

        if alive1:
            m1_n = float(dot(w1, a_n))
            e1_n = d1[n] - m1_n
            m_hat1[n] = m1_n
            e1[n] = e1_n
            if not np.isfinite(e1_n) or abs(e1_n) > DIV_THR:
                e1[n:] = np.nan
                m_hat1[n:] = np.nan
                alive1 = False

        if alive2:
            m2_n = float(dot(w2, a_n))
            e2_n = d2[n] - m2_n
            m_hat2[n] = m2_n
            e2[n] = e2_n
            if not np.isfinite(e2_n) or abs(e2_n) > DIV_THR:
                e2[n:] = np.nan
                m_hat2[n:] = np.nan
                alive2 = False

        if not alive1 and not alive2:
            break

        pi = dot(P, a_n)
        denom = lam_val + float(dot(a_n, pi))
        if not np.isfinite(denom) or denom <= 1e-12:
            if alive1:
                e1[n:] = np.nan
                m_hat1[n:] = np.nan
            if alive2:
                e2[n:] = np.nan
                m_hat2[n:] = np.nan
            break

        k = pi * (1.0 / denom)
        if alive1:
            w1 += e1_n * k
        if alive2:
            w2 += e2_n * k

        P -= k[:, None] * pi[None, :]
        P *= inv_lam

    return e1, m_hat1, w1, e2, m_hat2, w2


# ============================================================================
# 2. Normalized MISO LMS (NLMS) — Optimal Hyperparameters from Project 1
# ============================================================================
def nlms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int = OPTIMAL_NLMS_PARAMS["M"],
    mu: float = OPTIMAL_NLMS_PARAMS["mu"],
    delta: float = OPTIMAL_NLMS_PARAMS["delta"],
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized MISO Least Mean Squares (NLMS) adaptive filter using Project 1 optimal parameters.

    Default hyperparameters correspond to the optimal 5-fold Group K-Fold configuration
    obtained in Project 1 (`M = 22`, `mu = 0.01288`, `delta = 0.5`):
        m_hat[n] = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]     = d[n] - m_hat[n]
        w[n + 1] = w[n] + (mu / (delta + ||a[n]||^2)) * e[n] * a[n]
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    inv_denom = float(mu) / (float(delta) + np.einsum("ij,ij->i", U, U))

    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
    e = np.empty(N, dtype=float)
    m_hat = np.empty(N, dtype=float)

    dot = np.dot
    for n in range(N):
        a_n = U[n]
        m_n = float(dot(w, a_n))
        e_n = d[n] - m_n
        m_hat[n] = m_n
        e[n] = e_n

        if not np.isfinite(e_n) or abs(e_n) > DIV_THR:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

        w += (inv_denom[n] * e_n) * a_n

    return e, m_hat, w


# ============================================================================
# 3. Variable Step-Size MISO LMS (VS-LMS) — Optimal Hyperparameters from Project 1
# ============================================================================
def vs_lms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int = OPTIMAL_VS_LMS_PARAMS["M"],
    mu_max: float = OPTIMAL_VS_LMS_PARAMS["mu_max"],
    mu_min: float = OPTIMAL_VS_LMS_PARAMS["mu_min"],
    alpha: float = OPTIMAL_VS_LMS_PARAMS["alpha"],
    gamma: float = OPTIMAL_VS_LMS_PARAMS["gamma"],
    delta: float = OPTIMAL_VS_LMS_PARAMS["delta"],
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kwong-Johnston Variable Step-Size MISO LMS (VS-LMS) using Project 1 optimal parameters.

    Default hyperparameters correspond to the optimal 5-fold Group K-Fold configuration
    obtained in Project 1 (`M = 18`, `mu_max = 0.06558`, `mu_min = 0.00791`,
    `alpha = 0.98503`, `gamma = 0.00053`, `delta = 1.0`):
        m_hat[n]  = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]      = d[n] - m_hat[n]
        w[n + 1]  = w[n] + (mu[n] / (delta + ||a[n]||^2)) * e[n] * a[n]
        mu[n + 1] = clip(alpha * mu[n] + gamma * e^2[n], mu_min, mu_max)
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    mu_max = float(mu_max)
    effective_mu_min = float(min(mu_min, mu_max * 0.5))
    mu_n = mu_max
    alpha = float(alpha)
    gamma = float(gamma)

    inv_power = 1.0 / (float(delta) + np.einsum("ij,ij->i", U, U))

    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
    e = np.empty(N, dtype=float)
    m_hat = np.empty(N, dtype=float)

    dot = np.dot
    for n in range(N):
        a_n = U[n]
        m_n = float(dot(w, a_n))
        e_n = d[n] - m_n
        m_hat[n] = m_n
        e[n] = e_n

        if not np.isfinite(e_n) or abs(e_n) > DIV_THR:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

        w += (mu_n * inv_power[n] * e_n) * a_n

        mu_next = alpha * mu_n + gamma * (e_n * e_n)
        if mu_next > mu_max:
            mu_n = mu_max
        elif mu_next < effective_mu_min:
            mu_n = effective_mu_min
        else:
            mu_n = mu_next

    return e, m_hat, w


# ============================================================================
# Registry of Adaptive Algorithms & Bucket Function with Late Fusion
# ============================================================================
LMS_REGISTRY: dict[str, callable] = {
    "rls": rls,
    "nlms": nlms,
    "vs_lms": vs_lms,
    "vslms": vs_lms,
}


def lms_bucket(
    ppg1: np.ndarray,
    ppg2: np.ndarray,
    acc: np.ndarray,
    algorithm: str = "rls",
    M: int | None = None,
    w_init: np.ndarray | None = None,
    return_details: bool = False,
    **hyperparams,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Bucket function to select an adaptive algorithm (`rls`, `nlms`, or `vs_lms`) and apply Late Fusion.

    Executes Dual PPG Parallel Filtering + SNR-Weighted Late Fusion:
    1. Builds the shared 3-axis MISO regressor matrix `U = build_U(acc, M)` of shape `(N, 3 * M)`.
    2. Runs the chosen adaptive filter (`rls`, `nlms`, or `vs_lms`) on `ppg1` -> `e1` and `ppg2` -> `e2`.
    3. Fuses `e1` and `e2` via `snr_weighted_late_fusion(e1, e2)` -> `e_fused`.
    """
    algo_key = algorithm.strip().lower().replace("-", "_")
    if algo_key not in LMS_REGISTRY:
        valid = ", ".join(sorted(LMS_REGISTRY.keys()))
        raise ValueError(f"Unknown adaptive filter algorithm '{algorithm}'. Valid options: {valid}")

    params = dict(hyperparams)
    if M is None:
        if "M" in params:
            M = int(params.pop("M"))
        elif algo_key == "nlms":
            M = OPTIMAL_NLMS_PARAMS["M"]
        elif algo_key in ("vs_lms", "vslms"):
            M = OPTIMAL_VS_LMS_PARAMS["M"]
        else:
            M = 12
    else:
        params.pop("M", None)
        M = int(M)

    N = len(ppg1)
    U = _prepare_regressor(acc, N, M)

    if algo_key == "rls":
        e1, m_hat1, w1, e2, m_hat2, w2 = _rls_dual_channel(
            ppg1, ppg2, U, M=M, w_init=w_init, **params
        )
    else:
        lms_fn = LMS_REGISTRY[algo_key]
        e1, m_hat1, w1 = lms_fn(ppg1, U, M=M, w_init=w_init, **params)
        e2, m_hat2, w2 = lms_fn(ppg2, U, M=M, w_init=w_init, **params)

    e_fused, fusion_info = snr_weighted_late_fusion(e1, e2, return_weights=True)

    if return_details:
        return e_fused, {
            "e1": e1,
            "e2": e2,
            "m_hat1": m_hat1,
            "m_hat2": m_hat2,
            "w1": w1,
            "w2": w2,
            "fusion_info": fusion_info,
        }
    return e_fused
