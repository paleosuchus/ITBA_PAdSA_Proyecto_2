"""
Adaptive MISO filtering algorithms (Re-Architected RLS, LMS, NLMS, and VS-LMS) and Late Fusion bucket.

Implements the adaptive filtering stage of the PPG heart signal reconstruction pipeline:
- Primary input `d[n]`: Preprocessed PPG channel (`s[n] + m[n]`, cardiac signal + motion artifact).
- Reference input `a[n]`: 3-axis accelerometer dilated tapped-delay vector `[a_1^T[n], a_2^T[n], a_3^T[n]]^T`.
- MISO motion artifact estimate:
      m_hat[n] = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]
- Cleared cardiac signal (a priori error / innovation output):
      e[n] = d[n] - m_hat[n] = s[n] + m[n] - w^T[n] a[n]

Feature-Matched Pipeline (Shared Structural Parameters):
To ensure a strictly fair comparison between RLS and the LMS family, all four adaptive filters
(`rls`, `standard_lms`/`lms`, `nlms`, and `vs_lms`) support the same structural pipeline features:
- `delay_step`: Optional tap dilation stride `D >= 1` offering a conditioning/regularization
  alternative and computation-time saver by reducing the parameter count `L = 3M` needed for a
  given temporal aperture.
- `huber_c`: Huber M-estimator robust innovation clipping protecting `w[n]` against impulsive spikes.
- `warm_start_ratio`: Strictly intra-window reverse-time warm-up pass eliminating cold-start
  transients at `n = 0` without cross-window state propagation.

Includes:
1. `rls`: Re-architected Recursive Least Squares (RLS) MISO adaptive filter with all hyperparameters
   tunable (`M`, `delay_step`, `lam` / `N_eff`, `delta`, `diag_load`, `huber_c`, `warm_start_ratio`).
2. `standard_lms` (`lms`): Standard MISO LMS configured with the optimal step size obtained in
   Project 1 (`M = 11`, `mu = 0.00070`) plus shared structural parameters.
3. `nlms`: Normalized MISO LMS configured with the optimal hyperparameters obtained in Project 1
   (`M = 22`, `mu = 0.01288`, `delta = 0.5`) plus shared structural parameters.
4. `vs_lms`: Variable Step-Size MISO LMS configured with the optimal hyperparameters obtained in Project 1
   (`M = 18`, `mu_max = 0.06558`, `mu_min = 0.00791`, `alpha = 0.98503`, `gamma = 0.00053`, `delta = 1.0`)
   plus shared structural parameters.
5. `lms_bucket`: Bucket function that runs the selected adaptive filter (`rls`, `lms`/`standard_lms`,
   `nlms`, or `vs_lms`) in parallel on `ppg1` and `ppg2` and fuses the resulting error signals via
   `snr_weighted_late_fusion`.
"""

import numpy as np
from scipy.linalg import blas as _blas

_dgemv = _blas.dgemv
_dger = _blas.dger

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
OPTIMAL_LMS_PARAMS: dict = {
    "M": 11,
    "mu": 0.00070,
}
OPTIMAL_LMS_HR_CONFIG: dict = {
    "hr_algorithm": "periodogram",
    "hr_params": {"nfft": 4096, "window_type": "hamming"},
}

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


def _prepare_regressor(
    acc_or_U: np.ndarray, N: int, M: int, delay_step: int = 1
) -> np.ndarray:
    """Return the (N, 3 * M) regressor matrix U from either `acc` (3, N) or prebuilt `U`."""
    arr = np.asarray(acc_or_U, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == N and arr.shape[1] != N:
        return arr
    return build_U(arr, M, delay_step=delay_step)


def _resolve_rls_forgetting(
    lam: float = 0.996,
    lambda_: float | None = None,
    N_eff: float | None = None,
) -> float:
    """Resolve forgetting factor `lam` from either `N_eff` (effective memory horizon) or `lam`."""
    if N_eff is not None:
        n_eff_val = max(10.0, float(N_eff))
        lam_val = 1.0 - (1.0 / n_eff_val)
    else:
        lam_val = float(lambda_ if lambda_ is not None else lam)
    if not (0.0 < lam_val <= 1.0):
        raise ValueError(f"Forgetting factor 'lam' must be in (0, 1], got {lam_val}.")
    return lam_val


def _huber_clip(
    e_n: float, sigma_e_sq: float, huber_thr: float, use_huber: bool
) -> tuple[float, float]:
    """Apply recursive Huber M-estimator clipping to innovation `e_n` and update scale."""
    if not use_huber:
        return e_n, sigma_e_sq
    bound = huber_thr * np.sqrt(sigma_e_sq)
    e_upd = bound if e_n > bound else (-bound if e_n < -bound else e_n)
    sigma_e_sq_next = 0.98 * sigma_e_sq + 0.02 * (e_upd * e_upd)
    return e_upd, sigma_e_sq_next


def _run_lms_family(
    d: np.ndarray,
    U: np.ndarray,
    step_weights: np.ndarray | None,
    fixed_mu: float | None = None,
    vs_params: tuple[float, float, float, float] | None = None,
    huber_c: float | None = None,
    warm_start_ratio: float = 0.0,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared feature-matched execution engine for `standard_lms`, `nlms`, and `vs_lms`.

    Supports intra-window reverse-time warm-starting (`warm_start_ratio`) and
    Huber M-estimator innovation clipping (`huber_c`) identically to `rls`.
    """
    N, L = U.shape
    use_huber = huber_c is not None and float(huber_c) > 0.0
    huber_thr = float(huber_c) if use_huber else 0.0

    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
    dot = np.dot

    if vs_params is not None:
        mu_max, effective_mu_min, alpha, gamma = vs_params
        mu_n = mu_max
    else:
        mu_max = effective_mu_min = alpha = gamma = 0.0
        mu_n = 0.0

    # Intra-window reverse-time warm-start pass (strictly within the current isolated window)
    n_warm = int(round(float(warm_start_ratio) * N)) if w_init is None else 0
    if n_warm > 1:
        n_warm = min(N, n_warm)
        sigma_e_sq = 1.0
        for n in range(n_warm - 1, -1, -1):
            a_n = U[n]
            e_n = d[n] - float(dot(w, a_n))
            if not np.isfinite(e_n) or abs(e_n) > DIV_THR:
                w.fill(0.0)
                if vs_params is not None:
                    mu_n = mu_max
                break
            e_upd, sigma_e_sq = _huber_clip(e_n, sigma_e_sq, huber_thr, use_huber)
            if vs_params is not None:
                w += (mu_n * step_weights[n] * e_upd) * a_n
                mu_next = alpha * mu_n + gamma * (e_upd * e_upd)
                mu_n = mu_max if mu_next > mu_max else (effective_mu_min if mu_next < effective_mu_min else mu_next)
            elif step_weights is not None:
                w += (step_weights[n] * e_upd) * a_n
            else:
                w += (fixed_mu * e_upd) * a_n

    e = np.empty(N, dtype=float)
    m_hat = np.empty(N, dtype=float)
    sigma_e_sq = 1.0

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

        e_upd, sigma_e_sq = _huber_clip(e_n, sigma_e_sq, huber_thr, use_huber)
        if vs_params is not None:
            w += (mu_n * step_weights[n] * e_upd) * a_n
            mu_next = alpha * mu_n + gamma * (e_upd * e_upd)
            mu_n = mu_max if mu_next > mu_max else (effective_mu_min if mu_next < effective_mu_min else mu_next)
        elif step_weights is not None:
            w += (step_weights[n] * e_upd) * a_n
        else:
            w += (fixed_mu * e_upd) * a_n

    return e, m_hat, w


# ============================================================================
# 1. Re-Architected Recursive Least Squares (RLS) MISO Adaptive Filter
# ============================================================================
def rls(
    d: np.ndarray,
    acc: np.ndarray,
    M: int = 12,
    lam: float = 0.996,
    delta: float = 1.0,
    delay_step: int = 2,
    N_eff: float | None = None,
    diag_load: float = 0.25,
    huber_c: float | None = 2.5,
    warm_start_ratio: float = 0.35,
    lambda_: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-architected MISO Recursive Least Squares (RLS) adaptive filter.

    Addresses the failure modes of naive RLS on isolated 8s narrowband PPG/IMU windows:
    1. Dimension-Invariant Input-Power Regularization:
       Initializes `P[0] = (delta * L)^{-1} I_L` where `L = 3 * M = E[||a[n]||^2]`, making
       `delta` invariant to filter order `M` and stabilizing even dense high-order `D = 1` filters.
    2. Persistent Diagonal Loading (`diag_load`):
       Applies convex covariance shrinkage `P[n+1] <- (1 - tau) P[n+1] + tau P_0 I_L` with
       `tau = (1 - lam) * diag_load`, bounding the null-space eigenvalues of `P[n]` so
       `lam < 1` never triggers covariance explosion or cardiac signal cancellation.
    3. Optional Dilated Tap Delay Stride (`delay_step = D`):
       Constructs regressor taps spaced by `D` samples (`a_k[n - i*D]`), acting as an optional
       conditioning/regularization solution and computation-time saver that shrinks `O(L^2)`
       matrix operations while preserving temporal aperture `T_aperture = (M - 1) * D / fs`.
    4. Huber M-Estimator Innovation Clipping (`huber_c`) & Intra-Window Warm-Starting (`warm_start_ratio`):
       Clips impulsive innovation spikes in the weight update (`|e_tilde[n]| <= huber_c * sigma_e[n]`)
       and runs an intra-window backward warm-up pass over the first `warm_start_ratio * N` samples
       so `w[0]` starts already converged at `n = 0` without any cross-window state propagation.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M, delay_step=delay_step)
    L = U.shape[1]

    lam_val = _resolve_rls_forgetting(lam=lam, lambda_=lambda_, N_eff=N_eff)
    inv_lam = 1.0 / lam_val

    delta_val = float(delta)
    if delta_val <= 0.0:
        raise ValueError(f"Regularization parameter 'delta' must be > 0, got {delta_val}.")

    # Dimension-invariant prior inverse variance P_0 = 1 / (delta * L)
    p0_diag = 1.0 / (delta_val * float(max(1, L)))
    tau_leak = float(np.clip((1.0 - lam_val) * float(diag_load), 0.0, 0.25))
    one_minus_tau = 1.0 - tau_leak
    p_scale = inv_lam * one_minus_tau
    diag_add = tau_leak * p0_diag

    use_huber = huber_c is not None and float(huber_c) > 0.0
    huber_thr = float(huber_c) if use_huber else 0.0

    # Fortran-contiguous P enables zero-copy in-place BLAS dgemv / dger updates
    P = np.zeros((L, L), dtype=np.float64, order="F")
    P_flat = P.ravel(order="K")
    P_flat[:: L + 1] = p0_diag
    w = np.zeros(L, dtype=np.float64) if w_init is None else np.array(w_init, dtype=np.float64, copy=True)

    dot = np.dot

    # Intra-window reverse-time warm-start pass (strictly within the current isolated window)
    n_warm = int(round(float(warm_start_ratio) * N)) if w_init is None else 0
    if n_warm > 1:
        n_warm = min(N, n_warm)
        sigma_e_sq = 1.0
        for n in range(n_warm - 1, -1, -1):
            a_n = U[n]
            e_n = d[n] - float(dot(w, a_n))
            if not np.isfinite(e_n) or abs(e_n) > DIV_THR:
                w.fill(0.0)
                P.fill(0.0)
                P_flat[:: L + 1] = p0_diag
                break
            e_upd, sigma_e_sq = _huber_clip(e_n, sigma_e_sq, huber_thr, use_huber)

            pi = _dgemv(1.0, P, a_n)
            denom = lam_val + float(dot(a_n, pi))
            if denom <= 1e-12:
                break
            inv_denom = 1.0 / denom
            w += (e_upd * inv_denom) * pi
            P = _dger(-inv_denom, pi, pi, a=P, overwrite_a=1)
            P *= p_scale
            if tau_leak > 0.0:
                P_flat[:: L + 1] += diag_add

        # Reset P toward well-conditioned prior covariance while keeping warm-started weights w[0]
        P *= 0.5
        P_flat[:: L + 1] += 0.5 * p0_diag

    e = np.empty(N, dtype=np.float64)
    m_hat = np.empty(N, dtype=np.float64)
    sigma_e_sq = 1.0

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

        e_upd, sigma_e_sq = _huber_clip(e_n, sigma_e_sq, huber_thr, use_huber)

        pi = _dgemv(1.0, P, a_n)
        denom = lam_val + float(dot(a_n, pi))
        if not np.isfinite(denom) or denom <= 1e-12:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

        inv_denom = 1.0 / denom
        w += (e_upd * inv_denom) * pi
        P = _dger(-inv_denom, pi, pi, a=P, overwrite_a=1)
        P *= p_scale
        if tau_leak > 0.0:
            P_flat[:: L + 1] += diag_add

    return e, m_hat, w


def _rls_dual_channel(
    ppg1: np.ndarray,
    ppg2: np.ndarray,
    U: np.ndarray,
    M: int = 12,
    lam: float = 0.996,
    delta: float = 1.0,
    delay_step: int = 2,
    N_eff: float | None = None,
    diag_load: float = 0.25,
    huber_c: float | None = 2.5,
    warm_start_ratio: float = 0.35,
    lambda_: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run parallel MISO RLS on `ppg1` and `ppg2` sharing the inverse correlation matrix `P[n]`.

    Because both PPG channels share the exact same 3-axis accelerometer reference `U`,
    the gain vector `k[n]` and regularized inverse correlation matrix `P[n]` depend strictly
    on `U`, `lam`, `delta`, and `diag_load`, allowing both branches to be updated simultaneously
    with identical mathematical results to two independent calls to `rls`.
    """
    d1 = np.asarray(ppg1, dtype=float)
    d2 = np.asarray(ppg2, dtype=float)
    N = len(d1)
    L = U.shape[1]

    lam_val = _resolve_rls_forgetting(lam=lam, lambda_=lambda_, N_eff=N_eff)
    inv_lam = 1.0 / lam_val

    delta_val = float(delta)
    if delta_val <= 0.0:
        raise ValueError(f"Regularization parameter 'delta' must be > 0, got {delta_val}.")

    p0_diag = 1.0 / (delta_val * float(max(1, L)))
    tau_leak = float(np.clip((1.0 - lam_val) * float(diag_load), 0.0, 0.25))
    one_minus_tau = 1.0 - tau_leak
    p_scale = inv_lam * one_minus_tau
    diag_add = tau_leak * p0_diag

    use_huber = huber_c is not None and float(huber_c) > 0.0
    huber_thr = float(huber_c) if use_huber else 0.0

    # Fortran-contiguous P enables zero-copy in-place BLAS dgemv / dger updates
    P = np.zeros((L, L), dtype=np.float64, order="F")
    P_flat = P.ravel(order="K")
    P_flat[:: L + 1] = p0_diag
    if w_init is None:
        w1 = np.zeros(L, dtype=np.float64)
        w2 = np.zeros(L, dtype=np.float64)
    else:
        w1 = np.array(w_init, dtype=np.float64, copy=True)
        w2 = np.array(w_init, dtype=np.float64, copy=True)

    dot = np.dot

    # Intra-window reverse-time warm-start pass (strictly within the current isolated window)
    n_warm = int(round(float(warm_start_ratio) * N)) if w_init is None else 0
    if n_warm > 1:
        n_warm = min(N, n_warm)
        sig1_sq = 1.0
        sig2_sq = 1.0
        for n in range(n_warm - 1, -1, -1):
            a_n = U[n]
            e1_n = d1[n] - float(dot(w1, a_n))
            e2_n = d2[n] - float(dot(w2, a_n))
            if (
                not np.isfinite(e1_n)
                or abs(e1_n) > DIV_THR
                or not np.isfinite(e2_n)
                or abs(e2_n) > DIV_THR
            ):
                w1.fill(0.0)
                w2.fill(0.0)
                P.fill(0.0)
                P_flat[:: L + 1] = p0_diag
                break

            e1_upd, sig1_sq = _huber_clip(e1_n, sig1_sq, huber_thr, use_huber)
            e2_upd, sig2_sq = _huber_clip(e2_n, sig2_sq, huber_thr, use_huber)

            pi = _dgemv(1.0, P, a_n)
            denom = lam_val + float(dot(a_n, pi))
            if denom <= 1e-12:
                break
            inv_denom = 1.0 / denom
            w1 += (e1_upd * inv_denom) * pi
            w2 += (e2_upd * inv_denom) * pi
            P = _dger(-inv_denom, pi, pi, a=P, overwrite_a=1)
            P *= p_scale
            if tau_leak > 0.0:
                P_flat[:: L + 1] += diag_add

        P *= 0.5
        P_flat[:: L + 1] += 0.5 * p0_diag

    e1 = np.empty(N, dtype=np.float64)
    e2 = np.empty(N, dtype=np.float64)
    m_hat1 = np.empty(N, dtype=np.float64)
    m_hat2 = np.empty(N, dtype=np.float64)

    alive1 = True
    alive2 = True
    sig1_sq = 1.0
    sig2_sq = 1.0

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
            else:
                e1_upd, sig1_sq = _huber_clip(e1_n, sig1_sq, huber_thr, use_huber)

        if alive2:
            m2_n = float(dot(w2, a_n))
            e2_n = d2[n] - m2_n
            m_hat2[n] = m2_n
            e2[n] = e2_n
            if not np.isfinite(e2_n) or abs(e2_n) > DIV_THR:
                e2[n:] = np.nan
                m_hat2[n:] = np.nan
                alive2 = False
            else:
                e2_upd, sig2_sq = _huber_clip(e2_n, sig2_sq, huber_thr, use_huber)

        if not alive1 and not alive2:
            break

        pi = _dgemv(1.0, P, a_n)
        denom = lam_val + float(dot(a_n, pi))
        if not np.isfinite(denom) or denom <= 1e-12:
            if alive1:
                e1[n:] = np.nan
                m_hat1[n:] = np.nan
            if alive2:
                e2[n:] = np.nan
                m_hat2[n:] = np.nan
            break

        inv_denom = 1.0 / denom
        if alive1:
            w1 += (e1_upd * inv_denom) * pi
        if alive2:
            w2 += (e2_upd * inv_denom) * pi

        P = _dger(-inv_denom, pi, pi, a=P, overwrite_a=1)
        P *= p_scale
        if tau_leak > 0.0:
            P_flat[:: L + 1] += diag_add

    return e1, m_hat1, w1, e2, m_hat2, w2


# ============================================================================
# 2. Standard MISO LMS (Feature-Matched Pipeline) — Optimal Project 1 Step Size
# ============================================================================
def standard_lms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int = OPTIMAL_LMS_PARAMS["M"],
    mu: float = OPTIMAL_LMS_PARAMS["mu"],
    delay_step: int = 1,
    huber_c: float | None = None,
    warm_start_ratio: float = 0.0,
    my: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard MISO LMS using Project 1 optimal parameters with shared structural features."""
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M, delay_step=delay_step)
    step_mu = float(my if my is not None else mu)
    return _run_lms_family(
        d,
        U,
        step_weights=None,
        fixed_mu=step_mu,
        huber_c=huber_c,
        warm_start_ratio=warm_start_ratio,
        w_init=w_init,
    )


# ============================================================================
# 3. Normalized MISO LMS (NLMS, Feature-Matched Pipeline) — Optimal Project 1 Params
# ============================================================================
def nlms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int = OPTIMAL_NLMS_PARAMS["M"],
    mu: float = OPTIMAL_NLMS_PARAMS["mu"],
    delta: float = OPTIMAL_NLMS_PARAMS["delta"],
    delay_step: int = 1,
    huber_c: float | None = None,
    warm_start_ratio: float = 0.0,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized MISO LMS using Project 1 optimal parameters with shared structural features.

    Default step hyperparameters correspond to the optimal 5-fold Group K-Fold configuration
    obtained in Project 1 (`M = 22`, `mu = 0.01288`, `delta = 0.5`), while supporting matched
    `delay_step`, `huber_c`, and `warm_start_ratio` for a fair feature-matched comparison with RLS.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M, delay_step=delay_step)
    inv_denom = float(mu) / (float(delta) + np.einsum("ij,ij->i", U, U))
    return _run_lms_family(
        d,
        U,
        step_weights=inv_denom,
        huber_c=huber_c,
        warm_start_ratio=warm_start_ratio,
        w_init=w_init,
    )


# ============================================================================
# 4. Variable Step-Size MISO LMS (VS-LMS, Feature-Matched Pipeline)
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
    delay_step: int = 1,
    huber_c: float | None = None,
    warm_start_ratio: float = 0.0,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kwong-Johnston Variable Step-Size MISO LMS using Project 1 optimal params + shared structural features.

    Default step hyperparameters correspond to the optimal 5-fold Group K-Fold configuration
    obtained in Project 1 (`M = 18`, `mu_max = 0.06558`, `mu_min = 0.00791`,
    `alpha = 0.98503`, `gamma = 0.00053`, `delta = 1.0`), while supporting matched
    `delay_step`, `huber_c`, and `warm_start_ratio` for a fair feature-matched comparison with RLS.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M, delay_step=delay_step)

    mu_max = float(mu_max)
    effective_mu_min = float(min(mu_min, mu_max * 0.5))
    inv_power = 1.0 / (float(delta) + np.einsum("ij,ij->i", U, U))
    return _run_lms_family(
        d,
        U,
        step_weights=inv_power,
        vs_params=(mu_max, effective_mu_min, float(alpha), float(gamma)),
        huber_c=huber_c,
        warm_start_ratio=warm_start_ratio,
        w_init=w_init,
    )


# ============================================================================
# Registry of Adaptive Algorithms & Bucket Function with Late Fusion
# ============================================================================
LMS_REGISTRY: dict[str, callable] = {
    "rls": rls,
    "lms": standard_lms,
    "standard_lms": standard_lms,
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
    """Bucket function to select an adaptive algorithm (`rls`, `lms`/`standard_lms`, `nlms`, `vs_lms`) and apply Late Fusion.

    Executes Dual PPG Parallel Filtering + SNR-Weighted Late Fusion:
    1. Builds the shared 3-axis MISO regressor matrix `U = build_U(acc, M, delay_step)` of shape `(N, 3 * M)`.
    2. Runs the chosen adaptive filter (`rls`, `standard_lms`/`lms`, `nlms`, or `vs_lms`) on `ppg1` -> `e1` and `ppg2` -> `e2`.
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
        elif algo_key in ("lms", "standard_lms"):
            M = OPTIMAL_LMS_PARAMS["M"]
        elif algo_key == "nlms":
            M = OPTIMAL_NLMS_PARAMS["M"]
        elif algo_key in ("vs_lms", "vslms"):
            M = OPTIMAL_VS_LMS_PARAMS["M"]
        else:
            M = 12
    else:
        params.pop("M", None)
        M = int(M)

    delay_step = int(params.get("delay_step", 2 if algo_key == "rls" else 1))
    N = len(ppg1)
    U = _prepare_regressor(acc, N, M, delay_step=delay_step)

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
