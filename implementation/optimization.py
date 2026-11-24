"""
Hyperparameter Optimization module for the Re-Architected MISO RLS Adaptive Filter,
Feature-Matched LMS Family (`nlms`, `vs_lms`), and Heart Rate Detector using
Subject-Wise Group K-Fold Cross-Validation and Conditional TPE.

Architecture:
1. Reparameterized Conditional Tree Search Space (`rls`, `nlms`, `vs_lms` + HR Detector):
   - Adaptive Filter (`rls`):
     * `M` in [6, 50]: Number of FIR taps per accelerometer axis (total MISO taps L = 3 * M in [18, 150]).
       The wide upper bound allows dense `delay_step = 1` models to span long temporal apertures
       (>390 ms) whenever input-power-scaled regularization and diagonal loading stabilize collinearity.
     * `delay_step` in {1, 2, 3, 4}: Optional tap dilation stride D (in samples). Acts as a
       conditioning/regularization aid and computation-time saver by reducing the O(L^2) covariance
       matrix size required to reach a given temporal aperture T_aperture = (M - 1) * D / fs.
     * `N_eff` in [150.0, 2500.0] (log-scale): Effective memory horizon N_eff = 1 / (1 - lam),
       reparameterizing the forgetting factor lam = 1 - 1 / N_eff so TPE explores multi-beat
       horizons (1.2 s to 20 s) smoothly without boundary pinning.
     * `delta` in [0.05, 50.0] (log-scale): Dimension-invariant input-power regularization
       initializing P[0] = (delta * L)^{-1} I_L.
     * `diag_load` in [0.05, 1.0] (log-scale): Persistent diagonal shrinkage bounding null-space
       eigenvalues of P[n] when lam < 1.
     * `huber_c` in [1.5, 4.0]: Huber M-estimator threshold clipping impulsive innovation spikes.
     * `warm_start_ratio` in {0.0, 0.25, 0.5}: Intra-window reverse-time warm-start fraction
       eliminating cold-start transients at n = 0 without cross-window state propagation.
   - Categorical Heart Rate Detector Choice:
     * `hr_algorithm` in {'periodogram', 'elgendi'}
   - Conditioned HR Sub-Parameters:
     * `periodogram`: `nfft` in {4096, 8192}, `window_type` in {'hann', 'hamming', 'blackman'}
     * `elgendi`:     `w1_sec` in [0.08, 0.14], `w2_sec` in [0.55, 0.80], `beta_offset` in [0.01, 0.08]

2. Joint Reoptimization on the Updated Structural Pipeline:
   - `rls`, `nlms`, and `vs_lms` are each optimized from scratch via 2-phase ConditionalTPE over the
     same filter-order range `M in [6, 50]` and shared structural options (`delay_step`, `huber_c`,
     `warm_start_ratio`) alongside their step-size/regularization hyperparameters and HR detector.

3. Validation Protocol:
   - Subject-Wise Group K-Fold (`subject_wise_group_kfold` using `sklearn.model_selection.GroupKFold`):
     partitions windows strictly by participant ID to prevent data leakage.
"""

import json
import os
import numpy as np

try:
    from .dataset_preparation import FS, subject_wise_group_kfold
    from .preprocessing import preprocess_signal
    from .reference_construction import build_U
    from .LMS_algorithms import (
        LMS_REGISTRY,
        OPTIMAL_LMS_HR_CONFIG,
        OPTIMAL_LMS_PARAMS,
        OPTIMAL_NLMS_HR_CONFIG,
        OPTIMAL_NLMS_PARAMS,
        OPTIMAL_VS_LMS_HR_CONFIG,
        OPTIMAL_VS_LMS_PARAMS,
        _rls_dual_channel,
    )
    from .fusion import estimate_cardiac_snr, snr_weighted_late_fusion
    from .bpm_detection import (
        estimate_bpm_window,
        objective_function,
        zero_padded_periodogram_bpm,
    )
    from .pipeline import HeartRateEstimatorModel
except ImportError:
    from dataset_preparation import FS, subject_wise_group_kfold
    from preprocessing import preprocess_signal
    from reference_construction import build_U
    from LMS_algorithms import (
        LMS_REGISTRY,
        OPTIMAL_LMS_HR_CONFIG,
        OPTIMAL_LMS_PARAMS,
        OPTIMAL_NLMS_HR_CONFIG,
        OPTIMAL_NLMS_PARAMS,
        OPTIMAL_VS_LMS_HR_CONFIG,
        OPTIMAL_VS_LMS_PARAMS,
        _rls_dual_channel,
    )
    from fusion import estimate_cardiac_snr, snr_weighted_late_fusion
    from bpm_detection import (
        estimate_bpm_window,
        objective_function,
        zero_padded_periodogram_bpm,
    )
    from pipeline import HeartRateEstimatorModel

# Maximum filter taps per axis and bounded delay-tensor cache lag (avoids Colab OOM)
MAX_M: int = 50
MAX_DELAY_STEP: int = 4
PRECOMPUTED_MAX_LAG: int = 96
MAX_LAG: int = PRECOMPUTED_MAX_LAG

# ============================================================================
# Conditional Tree Hyperparameter Search Space Definition (RLS, NLMS, VS-LMS, LMS + HR)
# ============================================================================
_RLS_SUBSPACE: dict = {
    "M": ("int", 6, MAX_M),
    "delay_step": ("choice", [1, 2, 3, 4]),
    "N_eff": ("log_float", 150.0, 2500.0),
    "delta": ("log_float", 0.05, 50.0),
    "diag_load": ("log_float", 0.05, 1.0),
    "huber_c": ("float", 1.5, 4.0),
    "warm_start_ratio": ("choice", [0.0, 0.25, 0.5]),
}

CONDITIONAL_SEARCH_SPACE: dict = {
    "lms_algorithm": ["nlms", "vs_lms", "rls"],
    "hr_algorithm": ["periodogram", "elgendi"],
    "rls_subspace": _RLS_SUBSPACE,
    "lms_subspaces": {
        "rls": _RLS_SUBSPACE,
        "lms": {
            "M": ("int", 6, MAX_M),
            "delay_step": ("choice", [1, 2, 3, 4]),
            "mu": ("log_float", 1e-4, 3e-3),
            "huber_c": ("float", 1.5, 4.0),
            "warm_start_ratio": ("choice", [0.0, 0.25, 0.5]),
        },
        "nlms": {
            "M": ("int", 6, MAX_M),
            "delay_step": ("choice", [1, 2, 3, 4]),
            "mu": ("log_float", 0.005, 0.25),
            "delta": ("log_float", 0.1, 5.0),
            "huber_c": ("float", 1.5, 4.0),
            "warm_start_ratio": ("choice", [0.0, 0.25, 0.5]),
        },
        "vs_lms": {
            "M": ("int", 6, MAX_M),
            "delay_step": ("choice", [1, 2, 3, 4]),
            "mu_max": ("log_float", 0.01, 0.25),
            "mu_min": ("log_float", 0.0005, 0.02),
            "alpha": ("float", 0.94, 0.995),
            "gamma": ("log_float", 1e-4, 5e-3),
            "delta": ("log_float", 0.2, 5.0),
            "huber_c": ("float", 1.5, 4.0),
            "warm_start_ratio": ("choice", [0.0, 0.25, 0.5]),
        },
    },
    "hr_subspaces": {
        "periodogram": {
            "nfft": ("choice", [4096, 8192]),
            "window_type": ("choice", ["hann", "hamming", "blackman"]),
        },
        "elgendi": {
            "w1_sec": ("float", 0.08, 0.14),
            "w2_sec": ("float", 0.55, 0.80),
            "beta_offset": ("float", 0.01, 0.08),
        },
    },
}

PROJECT2_RLS_REFERENCE: dict = {
    "lms_algorithm": "rls",
    "lms_params": {
        "M": 7,
        "delay_step": 4,
        "N_eff": 1409.281222670354,
        "delta": 5.11410341560248,
        "diag_load": 0.14339015026730578,
        "huber_c": 2.5014879715311853,
        "warm_start_ratio": 0.0,
        "lam": 0.9992904184176206,
    },
    "hr_algorithm": "periodogram",
    "hr_params": {"nfft": 8192, "window_type": "hann"},
}

# Reference 5-fold Group K-Fold CV results from Project 1 for direct comparison
PROJECT1_CV_REFERENCE: dict = {
    "baseline_cv": {
        "score_objective": 24.66,
        "score_aae": 12.52,
        "score_rmse": 22.45,
        "outlier_pct": 29.9,
    },
    "lms": {
        "hyperparameters": {
            "lms_algorithm": "lms",
            "lms_params": dict(OPTIMAL_LMS_PARAMS),
            "hr_algorithm": OPTIMAL_LMS_HR_CONFIG["hr_algorithm"],
            "hr_params": dict(OPTIMAL_LMS_HR_CONFIG["hr_params"]),
        },
        "score_objective": 12.88,
        "score_aae": 5.18,
        "score_rmse": 16.93,
        "outlier_pct": 6.9,
    },
    "nlms": {
        "hyperparameters": {
            "lms_algorithm": "nlms",
            "lms_params": dict(OPTIMAL_NLMS_PARAMS),
            "hr_algorithm": OPTIMAL_NLMS_HR_CONFIG["hr_algorithm"],
            "hr_params": dict(OPTIMAL_NLMS_HR_CONFIG["hr_params"]),
        },
        "score_objective": 11.43,
        "score_aae": 4.72,
        "score_rmse": 14.75,
        "outlier_pct": 6.6,
    },
    "vs_lms": {
        "hyperparameters": {
            "lms_algorithm": "vs_lms",
            "lms_params": {
                k: v for k, v in OPTIMAL_VS_LMS_PARAMS.items() if k != "delta"
            },
            "hr_algorithm": OPTIMAL_VS_LMS_HR_CONFIG["hr_algorithm"],
            "hr_params": dict(OPTIMAL_VS_LMS_HR_CONFIG["hr_params"]),
        },
        "score_objective": 11.32,
        "score_aae": 4.56,
        "score_rmse": 15.02,
        "outlier_pct": 6.4,
    },
}


# ============================================================================
# Tree-structured Parzen Estimator (TPE) Sampler for RLS + Conditional HR Tree
# ============================================================================
class ConditionalTPE:
    """Tree-structured Parzen Estimator (TPE) for RLS hyperparameters and HR detector selection."""

    def __init__(
        self,
        seed: int = 42,
        gamma_quantile: float = 0.25,
        n_startup_trials: int = 5,
        n_ei_candidates: int = 24,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.gamma_quantile = gamma_quantile
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.trials: list[tuple[dict, float]] = []

    def observe(self, config: dict, loss: float) -> None:
        """Record a completed trial configuration and its objective loss (lower is better)."""
        if np.isfinite(loss):
            self.trials.append((config, float(loss)))

    def _sample_param_prior(self, spec: tuple):
        """Sample a single parameter from its uniform/log-uniform/categorical prior."""
        ptype = spec[0]
        if ptype == "int":
            _, low, high = spec
            return int(self.rng.integers(low, high + 1))
        if ptype == "float":
            _, low, high = spec
            return float(self.rng.uniform(low, high))
        if ptype == "log_float":
            _, low, high = spec
            return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        if ptype == "choice":
            choices = spec[1]
            return choices[int(self.rng.integers(0, len(choices)))]
        raise ValueError(f"Unknown parameter specification: {spec}")

    def _sample_tpe_1d(self, spec: tuple, good_vals: list, bad_vals: list):
        """Sample a 1D parameter using Parzen KDEs over `good_vals` vs `bad_vals`."""
        if len(good_vals) < 2 or len(bad_vals) < 2:
            return self._sample_param_prior(spec)

        ptype = spec[0]
        if ptype == "choice":
            choices = spec[1]
            k = len(choices)
            good_counts = np.array([sum(1 for v in good_vals if v == c) for c in choices], dtype=float)
            bad_counts = np.array([sum(1 for v in bad_vals if v == c) for c in choices], dtype=float)
            p_good = (good_counts + 1.0) / (len(good_vals) + k)
            p_bad = (bad_counts + 1.0) / (len(bad_vals) + k)
            ei_weights = p_good / np.maximum(p_bad, 1e-12)
            probs = ei_weights / np.sum(ei_weights)
            return choices[int(self.rng.choice(k, p=probs))]

        _, low, high = spec
        is_log = ptype == "log_float"
        lo_t = np.log(low) if is_log else float(low)
        hi_t = np.log(high) if is_log else float(high)

        g_arr = np.log(np.asarray(good_vals, dtype=float)) if is_log else np.asarray(good_vals, dtype=float)
        b_arr = np.log(np.asarray(bad_vals, dtype=float)) if is_log else np.asarray(bad_vals, dtype=float)

        span = hi_t - lo_t
        sigma_g = max(1.06 * (np.std(g_arr) + 1e-6) * (len(g_arr) ** (-0.2)), 0.05 * span)
        sigma_b = max(1.06 * (np.std(b_arr) + 1e-6) * (len(b_arr) ** (-0.2)), 0.05 * span)

        comp_indices = self.rng.integers(0, len(g_arr), size=self.n_ei_candidates)
        candidates = np.clip(
            g_arr[comp_indices] + self.rng.normal(0.0, sigma_g, size=self.n_ei_candidates),
            lo_t,
            hi_t,
        )

        diff_g = (candidates[:, None] - g_arr[None, :]) / sigma_g
        diff_b = (candidates[:, None] - b_arr[None, :]) / sigma_b
        p_l = np.mean(np.exp(-0.5 * (diff_g**2)), axis=1) / sigma_g + 1e-12
        p_g = np.mean(np.exp(-0.5 * (diff_b**2)), axis=1) / sigma_b + 1e-12

        best_cand = float(candidates[int(np.argmax(np.log(p_l) - np.log(p_g)))])
        if is_log:
            return float(np.exp(best_cand))
        if ptype == "int":
            return int(np.clip(round(best_cand), low, high))
        return best_cand

    def _split_good_bad(
        self, subset: list[tuple[dict, float]] | None = None
    ) -> tuple[list[dict], list[dict]]:
        """Partition observed configurations into good (`l`) and bad (`g`) sets."""
        pool = self.trials if subset is None else subset
        if not pool:
            return [], []
        losses = np.array([loss for _, loss in pool], dtype=float)
        n_good = max(1, int(np.ceil(self.gamma_quantile * len(losses))))
        order = np.argsort(losses)
        return [pool[i][0] for i in order[:n_good]], [pool[i][0] for i in order[n_good:]]

    def sample_lms_node(self, lms_algo: str = "rls") -> dict:
        """Sample hyperparameters for `lms_algo` ('rls', 'nlms', 'vs_lms', or 'lms')."""
        use_tpe = len(self.trials) >= self.n_startup_trials
        good_cfgs, bad_cfgs = self._split_good_bad() if use_tpe else ([], [])

        subspace = CONDITIONAL_SEARCH_SPACE["lms_subspaces"][lms_algo]
        good_branch = [
            c["lms_params"]
            for c in good_cfgs
            if c.get("lms_algorithm") == lms_algo and "lms_params" in c
        ]
        bad_branch = [
            c["lms_params"]
            for c in bad_cfgs
            if c.get("lms_algorithm") == lms_algo and "lms_params" in c
        ]

        params = {}
        for pname, pspec in subspace.items():
            if len(good_branch) >= 2 and len(bad_branch) >= 2:
                params[pname] = self._sample_tpe_1d(
                    pspec,
                    [p[pname] for p in good_branch if pname in p],
                    [p[pname] for p in bad_branch if pname in p],
                )
            else:
                params[pname] = self._sample_param_prior(pspec)

        if lms_algo == "rls":
            params["lam"] = float(1.0 - (1.0 / float(params["N_eff"])))
        elif lms_algo == "vs_lms" and params["mu_min"] >= params["mu_max"]:
            params["mu_min"] = min(params["mu_min"], params["mu_max"] * 0.25)

        return {"lms_algorithm": lms_algo, "lms_params": params}

    def sample_rls_node(self) -> dict:
        """Sample re-architected RLS hyperparameters."""
        return self.sample_lms_node("rls")

    def sample_full_tree(self, lms_algo: str = "rls") -> dict:
        """Sample a complete conditional tree configuration (`lms_algo` hyperparameters + HR detector branch)."""
        lms_branch = self.sample_lms_node(lms_algo)
        use_tpe = len(self.trials) >= self.n_startup_trials
        good_cfgs, bad_cfgs = self._split_good_bad() if use_tpe else ([], [])

        hr_spec = ("choice", CONDITIONAL_SEARCH_SPACE["hr_algorithm"])
        if use_tpe:
            hr_algo = self._sample_tpe_1d(
                hr_spec,
                [c["hr_algorithm"] for c in good_cfgs if "hr_algorithm" in c],
                [c["hr_algorithm"] for c in bad_cfgs if "hr_algorithm" in c],
            )
        else:
            hr_algo = self._sample_param_prior(hr_spec)

        hr_subspace = CONDITIONAL_SEARCH_SPACE["hr_subspaces"][hr_algo]
        hr_trials = [
            (c, loss)
            for c, loss in self.trials
            if c.get("hr_algorithm") == hr_algo and "hr_params" in c
        ]
        if len(hr_trials) >= 4:
            good_hr_cfgs, bad_hr_cfgs = self._split_good_bad(hr_trials)
            good_hr = [c["hr_params"] for c in good_hr_cfgs]
            bad_hr = [c["hr_params"] for c in bad_hr_cfgs]
        else:
            good_hr = [
                c["hr_params"]
                for c in good_cfgs
                if c.get("hr_algorithm") == hr_algo and "hr_params" in c
            ]
            bad_hr = [
                c["hr_params"]
                for c in bad_cfgs
                if c.get("hr_algorithm") == hr_algo and "hr_params" in c
            ]

        hr_params = {}
        for pname, pspec in hr_subspace.items():
            if len(good_hr) >= 2 and len(bad_hr) >= 2:
                hr_params[pname] = self._sample_tpe_1d(
                    pspec,
                    [p[pname] for p in good_hr],
                    [p[pname] for p in bad_hr],
                )
            else:
                hr_params[pname] = self._sample_param_prior(pspec)

        return {
            "lms_algorithm": lms_algo,
            "lms_params": lms_branch["lms_params"],
            "hr_algorithm": hr_algo,
            "hr_params": hr_params,
        }


# ============================================================================
# Early Pruner Across Cross-Validation Folds
# ============================================================================
class MedianFoldPruner:
    """Early pruning mechanism for multi-fold cross-validation trials."""

    def __init__(self, n_warmup_trials: int = 4, percentile: float = 50.0) -> None:
        self.n_warmup_trials = n_warmup_trials
        self.percentile = percentile
        self.step_history: dict[int, list[float]] = {}

    def report_and_check_prune(
        self, step: int, running_score: float, allow_prune: bool = True
    ) -> bool:
        """Report intermediate running score at fold `step` and return True if trial should prune."""
        if not np.isfinite(running_score):
            return True

        history = self.step_history.setdefault(step, [])
        should_prune = False
        if allow_prune and len(history) >= self.n_warmup_trials:
            threshold = float(np.percentile(history, self.percentile))
            if running_score > threshold:
                should_prune = True

        history.append(float(running_score))
        return should_prune


# ============================================================================
# Preprocessed Window & 3-Axis MISO Dilated Delay Tensor Cache
# ============================================================================
def preprocess_dataset_windows(
    X: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    max_lag: int = MAX_LAG,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Preprocess all 5D windows `(n_windows, 5, N)` and precompute `U_max` of shape `(N, 3, max_lag)`."""
    ppg1_all, ppg2_all, acc_all = preprocess_signal(X, lo=lo, hi=hi, fs=fs)
    cached = []
    for i in range(len(X)):
        ppg1 = ppg1_all[i]
        ppg2 = ppg2_all[i]
        acc = acc_all[i]
        N = len(ppg1)
        U_max_3d = build_U(acc, max_lag, delay_step=1).reshape(N, 3, max_lag)
        cached.append((ppg1, ppg2, acc, U_max_3d))
    return cached


def _filter_cached_window(
    cached_win: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lms_algo: str,
    M: int,
    lms_params: dict,
) -> np.ndarray:
    """Run parallel 3-axis MISO adaptive filtering + SNR-Weighted Late Fusion using precomputed `U_max_3d`."""
    ppg1, ppg2, acc, U_max_3d = cached_win
    N = len(ppg1)
    delay_step = int(lms_params.get("delay_step", 2 if lms_algo == "rls" else 1))
    span = M * delay_step
    if span <= U_max_3d.shape[2]:
        U = U_max_3d[:, :, :span:delay_step].reshape(N, 3 * M)
    else:
        U = build_U(acc, M, delay_step=delay_step)

    if lms_algo == "rls":
        e1, _, _, e2, _, _ = _rls_dual_channel(ppg1, ppg2, U, M=M, **lms_params)
    else:
        lms_fn = LMS_REGISTRY[lms_algo]
        e1, _, _ = lms_fn(ppg1, U, M=M, **lms_params)
        e2, _, _ = lms_fn(ppg2, U, M=M, **lms_params)
    return snr_weighted_late_fusion(e1, e2, return_weights=False)


# ============================================================================
# Phase 1 & Phase 2 Optimization Functions (RLS Only)
# ============================================================================
def run_phase1_snr_screening(
    preprocessed_windows: list[tuple],
    y: np.ndarray,
    screening_indices: np.ndarray,
    lms_algo: str = "rls",
    n_trials: int = 16,
    top_k: int = 3,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Phase 1: Fast Screening of `lms_algo` hyperparameters on a subset of training windows.

    Evaluates candidate configurations on `screening_indices` by measuring both
    in-band cardiac spectral SNR and periodogram BPM accuracy against ground-truth `y`.
    """
    tpe = ConditionalTPE(seed=seed, gamma_quantile=0.25, n_startup_trials=6)
    phase1_history: list[dict] = []

    for trial_idx in range(n_trials):
        candidate = tpe.sample_lms_node(lms_algo)
        algo_params = dict(candidate["lms_params"])
        M = int(algo_params.pop("M"))

        snr_scores = []
        bpm_preds = []
        bpm_targets = y[screening_indices]
        diverged = False

        for idx in screening_indices:
            cached_win = preprocessed_windows[int(idx)]
            e_fused = _filter_cached_window(cached_win, lms_algo, M, algo_params)
            if not np.all(np.isfinite(e_fused)):
                diverged = True
                break
            snr_scores.append(estimate_cardiac_snr(e_fused))
            bpm_preds.append(zero_padded_periodogram_bpm(e_fused, nfft=4096))

        if diverged or not snr_scores:
            mean_snr = 0.0
            screen_obj = 1e6
        else:
            mean_snr = float(np.mean(snr_scores))
            screen_obj = objective_function(bpm_preds, bpm_targets)

        tpe.observe(candidate, screen_obj)
        phase1_history.append(
            {
                "trial": trial_idx,
                "lms_algorithm": lms_algo,
                "lms_params": dict(candidate["lms_params"]),
                "mean_snr": mean_snr,
                "screen_objective": screen_obj,
            }
        )

    sorted_trials = sorted(phase1_history, key=lambda r: r["screen_objective"])
    top_configs = [
        {"lms_algorithm": lms_algo, "lms_params": dict(t["lms_params"])}
        for t in sorted_trials[:top_k]
    ]
    return top_configs, phase1_history


def run_phase2_tpe_optimization(
    preprocessed_windows: list[tuple],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed_configs: list[dict],
    lms_algo: str = "rls",
    n_trials: int = 18,
    prune_percentile: float = 50.0,
    seed: int = 42,
) -> tuple[dict, float, list[dict]]:
    """Phase 2: End-to-End TPE Optimization of `lms_algo` + HR Detector (`periodogram` vs `elgendi`) across Group K-Folds."""
    warm_start_queue: list[dict] = []
    for cfg in seed_configs:
        for hr_algo in CONDITIONAL_SEARCH_SPACE["hr_algorithm"]:
            hr_default = (
                {"nfft": 8192, "window_type": "hamming"}
                if hr_algo == "periodogram"
                else {"w1_sec": 0.111, "w2_sec": 0.667, "beta_offset": 0.02}
            )
            warm_start_queue.append(
                {
                    "lms_algorithm": lms_algo,
                    "lms_params": dict(cfg["lms_params"]),
                    "hr_algorithm": hr_algo,
                    "hr_params": hr_default,
                }
            )

    tpe = ConditionalTPE(
        seed=seed + 100,
        gamma_quantile=0.25,
        n_startup_trials=max(4, len(warm_start_queue)),
    )
    pruner = MedianFoldPruner(
        n_warmup_trials=max(4, len(warm_start_queue)),
        percentile=prune_percentile,
    )

    best_config: dict | None = None
    best_score: float = float("inf")
    phase2_history: list[dict] = []

    for trial_idx in range(n_trials):
        is_warm_start = trial_idx < len(warm_start_queue)
        if is_warm_start:
            config = warm_start_queue[trial_idx]
        else:
            config = tpe.sample_full_tree(lms_algo)

        algo_params = dict(config["lms_params"])
        M = int(algo_params.pop("M"))
        hr_algo = config["hr_algorithm"]
        hr_params = dict(config["hr_params"])

        fold_obj_scores = []
        fold_aaes = []
        fold_rmses = []
        fold_outliers = []
        pruned = False

        for fold_idx, (_, val_indices) in enumerate(splits):
            val_preds = []
            val_targets = y[val_indices]

            for w_idx in val_indices:
                cached_win = preprocessed_windows[int(w_idx)]
                e_fused = _filter_cached_window(cached_win, lms_algo, M, algo_params)
                bpm_hat = estimate_bpm_window(e_fused, hr_algorithm=hr_algo, **hr_params)
                val_preds.append(bpm_hat)

            fold_obj = objective_function(val_preds, val_targets)
            preds_arr = np.asarray(val_preds, dtype=float)
            if np.all(np.isfinite(preds_arr)):
                abs_err = np.abs(preds_arr - val_targets)
                fold_aae = float(np.mean(abs_err))
                fold_rmse = float(np.sqrt(np.mean(abs_err**2)))
                fold_out = float(np.mean(abs_err > 10.0) * 100.0)
            else:
                fold_aae = float("inf")
                fold_rmse = float("inf")
                fold_out = 100.0

            fold_obj_scores.append(fold_obj)
            fold_aaes.append(fold_aae)
            fold_rmses.append(fold_rmse)
            fold_outliers.append(fold_out)
            running_obj = float(np.mean(fold_obj_scores))

            if pruner.report_and_check_prune(
                step=fold_idx,
                running_score=running_obj,
                allow_prune=not is_warm_start,
            ):
                pruned = True
                break

        final_obj = float(np.mean(fold_obj_scores)) if fold_obj_scores else float("inf")
        final_aae = float(np.mean(fold_aaes)) if fold_aaes else float("inf")
        final_rmse = float(np.mean(fold_rmses)) if fold_rmses else float("inf")
        final_out = float(np.mean(fold_outliers)) if fold_outliers else 100.0

        if not pruned and np.isfinite(final_obj):
            tpe.observe(config, final_obj)

        if not pruned and final_obj < best_score:
            best_score = final_obj
            best_config = config

        phase2_history.append(
            {
                "trial": trial_idx,
                "config": config,
                "score_objective": final_obj,
                "score_aae": final_aae,
                "score_rmse": final_rmse,
                "outlier_pct": final_out,
                "folds_evaluated": len(fold_obj_scores),
                "pruned": pruned,
            }
        )

    if best_config is None and phase2_history:
        best_entry = min(phase2_history, key=lambda r: r["score_objective"])
        best_config = best_entry["config"]
        best_score = best_entry["score_objective"]

    return best_config, best_score, phase2_history


def evaluate_model_cv(
    preprocessed_windows: list[tuple],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    lms_algo: str,
    M: int,
    lms_params: dict,
    hr_algo: str,
    hr_params: dict,
) -> dict:
    """Evaluate any configured adaptive filter + HR detector across Subject-Wise Group K-Folds."""
    fold_objs, fold_aaes, fold_rmses, fold_outs = [], [], [], []
    for _, val_indices in splits:
        val_preds = []
        val_targets = y[val_indices]
        for w_idx in val_indices:
            cached_win = preprocessed_windows[int(w_idx)]
            e_fused = _filter_cached_window(cached_win, lms_algo, M, lms_params)
            val_preds.append(estimate_bpm_window(e_fused, hr_algorithm=hr_algo, **hr_params))
        preds_arr = np.asarray(val_preds, dtype=float)
        if np.all(np.isfinite(preds_arr)):
            abs_err = np.abs(preds_arr - val_targets)
            fold_objs.append(objective_function(preds_arr, val_targets))
            fold_aaes.append(float(np.mean(abs_err)))
            fold_rmses.append(float(np.sqrt(np.mean(abs_err**2))))
            fold_outs.append(float(np.mean(abs_err > 10.0) * 100.0))
        else:
            fold_objs.append(float("inf"))
            fold_aaes.append(float("inf"))
            fold_rmses.append(float("inf"))
            fold_outs.append(100.0)

    return {
        "score_objective": float(np.mean(fold_objs)),
        "score_aae": float(np.mean(fold_aaes)),
        "score_rmse": float(np.mean(fold_rmses)),
        "outlier_pct": float(np.mean(fold_outs)),
    }


def evaluate_baseline_without_lms(
    preprocessed_windows: list[tuple],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Evaluate BPM estimation on preprocessed windows BEFORE adaptive filtering across CV folds."""
    fold_objs, fold_aaes, fold_rmses, fold_outs = [], [], [], []
    for _, val_indices in splits:
        val_preds = []
        val_targets = y[val_indices]
        for w_idx in val_indices:
            ppg1, ppg2, _, _ = preprocessed_windows[int(w_idx)]
            d_fused = snr_weighted_late_fusion(ppg1, ppg2, return_weights=False)
            val_preds.append(zero_padded_periodogram_bpm(d_fused, nfft=8192))
        preds_arr = np.asarray(val_preds, dtype=float)
        abs_err = np.abs(preds_arr - val_targets)
        fold_objs.append(objective_function(preds_arr, val_targets))
        fold_aaes.append(float(np.mean(abs_err)))
        fold_rmses.append(float(np.sqrt(np.mean(abs_err**2))))
        fold_outs.append(float(np.mean(abs_err > 10.0) * 100.0))

    return {
        "score_objective": float(np.mean(fold_objs)),
        "score_aae": float(np.mean(fold_aaes)),
        "score_rmse": float(np.mean(fold_rmses)),
        "outlier_pct": float(np.mean(fold_outs)),
    }


def summarize_algorithm_comparisons(
    phase1_history: list[dict],
    phase2_history: list[dict],
) -> dict:
    """Summarize the best RLS performance for each HR detection algorithm (`periodogram` vs `elgendi`)."""
    best_phase1 = (
        max(phase1_history, key=lambda e: e["mean_snr"]) if phase1_history else {}
    )

    best_phase2_by_hr = {}
    for entry in phase2_history:
        cfg = entry["config"]
        pair_key = f"rls + {cfg['hr_algorithm']}"
        obj = entry["score_objective"]
        if (
            pair_key not in best_phase2_by_hr
            or (not entry["pruned"] and best_phase2_by_hr[pair_key]["pruned"])
            or (
                entry["pruned"] == best_phase2_by_hr[pair_key]["pruned"]
                and obj < best_phase2_by_hr[pair_key]["score_objective"]
            )
        ):
            best_phase2_by_hr[pair_key] = entry

    return {
        "phase1_best_rls": best_phase1,
        "phase2_by_hr": best_phase2_by_hr,
    }


# ============================================================================
# Master Execution Function: Optimizes Re-Architected RLS & Includes Project 1 Models
# ============================================================================
def optimize_hr_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    phase1_trials: int = 24,
    phase1_sample_size: int = 150,
    phase1_top_k: int = 4,
    phase2_trials: int = 24,
    prune_percentile: float = 50.0,
    retrain_algorithms: tuple[str, ...] = ("nlms", "vs_lms"),
    reoptimize_rls: bool = True,
    seed: int = 42,
) -> dict:
    """Optimize `nlms`, `vs_lms`, and `rls` hyperparameters via Group K-Fold CV.

    Uses `M in [6, MAX_M]` (`MAX_M = 50`) for all adaptive filters so dense (`delay_step = 1`)
    and dilated (`delay_step > 1`) solutions compete fairly across the full search space.
    When `reoptimize_rls=False`, evaluates the stored `PROJECT2_RLS_REFERENCE` configuration.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    splits = subject_wise_group_kfold(X, n_splits=n_splits)
    preprocessed_windows = preprocess_dataset_windows(X)

    rng = np.random.default_rng(seed)
    n_total = len(X)
    screening_indices = (
        np.arange(n_total)
        if n_total <= phase1_sample_size
        else np.sort(rng.choice(n_total, size=phase1_sample_size, replace=False))
    )

    baseline_cv = evaluate_baseline_without_lms(preprocessed_windows, y, splits)

    results_by_lms: dict[str, dict] = {}
    all_p1_hist: list[dict] = []
    all_p2_hist: list[dict] = []
    rls_p1_hist: list[dict] = []
    rls_p2_hist: list[dict] = []
    top_rls_cfg: list[dict] = []

    for idx_algo, algo_name in enumerate(retrain_algorithms):
        algo_seed = seed + (idx_algo + 1) * 100
        top_cfg, p1_hist = run_phase1_snr_screening(
            preprocessed_windows=preprocessed_windows,
            y=y,
            screening_indices=screening_indices,
            lms_algo=algo_name,
            n_trials=phase1_trials,
            top_k=phase1_top_k,
            seed=algo_seed,
        )
        best_cfg, _, p2_hist = run_phase2_tpe_optimization(
            preprocessed_windows=preprocessed_windows,
            y=y,
            splits=splits,
            seed_configs=top_cfg,
            lms_algo=algo_name,
            n_trials=phase2_trials,
            prune_percentile=prune_percentile,
            seed=algo_seed,
        )
        all_p1_hist.extend(p1_hist)
        all_p2_hist.extend(p2_hist)

        unpruned = [t for t in p2_hist if not t["pruned"]]
        best_trial = min(unpruned if unpruned else p2_hist, key=lambda r: r["score_objective"])

        params_copy = dict(best_cfg["lms_params"])
        best_M = int(params_copy.pop("M"))
        model = HeartRateEstimatorModel(
            lms_algorithm=algo_name,
            M=best_M,
            lms_params=params_copy,
            hr_algorithm=best_cfg["hr_algorithm"],
            hr_params=best_cfg["hr_params"],
            fs=FS,
        )
        results_by_lms[algo_name] = {
            "model": model,
            "hyperparameters": best_cfg,
            "score_objective": float(best_trial["score_objective"]),
            "score_aae": float(best_trial["score_aae"]),
            "score_rmse": float(best_trial["score_rmse"]),
            "outlier_pct": float(best_trial["outlier_pct"]),
        }

    if reoptimize_rls:
        top_rls_cfg, rls_p1_hist = run_phase1_snr_screening(
            preprocessed_windows=preprocessed_windows,
            y=y,
            screening_indices=screening_indices,
            lms_algo="rls",
            n_trials=phase1_trials,
            top_k=phase1_top_k,
            seed=seed,
        )
        best_rls_cfg, best_rls_obj, rls_p2_hist = run_phase2_tpe_optimization(
            preprocessed_windows=preprocessed_windows,
            y=y,
            splits=splits,
            seed_configs=top_rls_cfg,
            lms_algo="rls",
            n_trials=phase2_trials,
            prune_percentile=prune_percentile,
            seed=seed,
        )
        unpruned = [t for t in rls_p2_hist if not t["pruned"]]
        best_rls_trial = min(unpruned if unpruned else rls_p2_hist, key=lambda r: r["score_objective"])
        rls_cv = {
            "score_objective": float(best_rls_trial["score_objective"]),
            "score_aae": float(best_rls_trial["score_aae"]),
            "score_rmse": float(best_rls_trial["score_rmse"]),
            "outlier_pct": float(best_rls_trial["outlier_pct"]),
        }
    else:
        best_rls_cfg = {
            "lms_algorithm": "rls",
            "lms_params": dict(PROJECT2_RLS_REFERENCE["lms_params"]),
            "hr_algorithm": PROJECT2_RLS_REFERENCE["hr_algorithm"],
            "hr_params": dict(PROJECT2_RLS_REFERENCE["hr_params"]),
        }
        rls_params_eval = dict(best_rls_cfg["lms_params"])
        rls_M_eval = int(rls_params_eval.pop("M"))
        rls_cv = evaluate_model_cv(
            preprocessed_windows,
            y,
            splits,
            lms_algo="rls",
            M=rls_M_eval,
            lms_params=rls_params_eval,
            hr_algo=best_rls_cfg["hr_algorithm"],
            hr_params=best_rls_cfg["hr_params"],
        )
        best_rls_obj = float(rls_cv["score_objective"])

    rls_params_copy = dict(best_rls_cfg["lms_params"])
    best_M = int(rls_params_copy.pop("M"))
    rls_model = HeartRateEstimatorModel(
        lms_algorithm="rls",
        M=best_M,
        lms_params=rls_params_copy,
        hr_algorithm=best_rls_cfg["hr_algorithm"],
        hr_params=best_rls_cfg["hr_params"],
        fs=FS,
    )
    results_by_lms["rls"] = {
        "model": rls_model,
        "hyperparameters": best_rls_cfg,
        **rls_cv,
    }

    comparison_summary = summarize_algorithm_comparisons(rls_p1_hist, rls_p2_hist)

    boundary_warnings: list[str] = []
    for algo_key, entry in results_by_lms.items():
        hp_cfg = entry["hyperparameters"]
        algo_m = int(hp_cfg["lms_params"]["M"])
        if algo_m >= MAX_M:
            boundary_warnings.append(
                f"{algo_key}: optimal M={algo_m} reached MAX_M={MAX_M}; consider raising MAX_M."
            )
        if algo_key == "rls":
            delta_val = float(hp_cfg["lms_params"].get("delta", 0.0))
            if delta_val >= 49.0:
                boundary_warnings.append(
                    f"rls: optimal delta={delta_val:.3f} is near upper bound 50.0."
                )

    return {
        "rls_model": rls_model,
        "best_model": rls_model,
        "best_hyperparameters": best_rls_cfg,
        "best_score": float(best_rls_obj),
        "results_by_lms": results_by_lms,
        "baseline_cv": baseline_cv,
        "comparison_summary": comparison_summary,
        "boundary_warnings": boundary_warnings,
        "max_m_searched": MAX_M,
        "phase1_top_rls": top_rls_cfg,
        "phase1_trials": rls_p1_hist + all_p1_hist,
        "phase2_trials": rls_p2_hist + all_p2_hist,
    }


def _to_json_native(obj):
    """Recursively convert NumPy scalars/arrays to native Python JSON types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_optimization_config(
    opt_results: dict,
    filepath: str,
    test_metrics: dict | None = None,
) -> str:
    """Serialize optimized CV models and metrics to JSON for downstream comparison notebooks."""
    algorithms_payload: dict = {}
    for algo_key, info in opt_results.get("results_by_lms", {}).items():
        hp = info["hyperparameters"]
        lms_params = dict(hp["lms_params"])
        M_val = int(lms_params.pop("M"))
        entry = {
            "M": M_val,
            "lms_params": _to_json_native(lms_params),
            "hr_algorithm": hp["hr_algorithm"],
            "hr_params": _to_json_native(hp["hr_params"]),
            "cv": {
                "score_objective": float(info["score_objective"]),
                "score_aae": float(info["score_aae"]),
                "score_rmse": float(info["score_rmse"]),
                "outlier_pct": float(info["outlier_pct"]),
            },
        }
        if test_metrics and algo_key in test_metrics:
            entry["test"] = _to_json_native(test_metrics[algo_key])
        algorithms_payload[algo_key] = entry

    payload = {
        "max_m_searched": int(opt_results.get("max_m_searched", MAX_M)),
        "boundary_warnings": list(opt_results.get("boundary_warnings", [])),
        "baseline_cv": _to_json_native(opt_results.get("baseline_cv", {})),
        "algorithms": algorithms_payload,
    }
    if test_metrics and "base" in test_metrics:
        payload["baseline_test"] = _to_json_native(test_metrics["base"])

    out_dir = os.path.dirname(os.path.abspath(filepath))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return os.path.abspath(filepath)


def load_optimization_config(filepath: str) -> dict:
    """Load an `optimized_models_config.json` artifact saved by `save_optimization_config`."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
