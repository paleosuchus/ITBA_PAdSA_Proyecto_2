"""
End-to-end heart signal reconstruction and BPM estimation pipeline for individual windows.

Processes each 5-dimensional window `[ppg1, ppg2, accx, accy, accz]` of length N independently:
1. Preprocessing (`preprocessing.py`):
   - 4th-order zero-phase Butterworth bandpass filter (0.5 - 4.0 Hz / 30 - 240 BPM).
   - Amplitude normalization (zero mean, unit standard deviation per channel).
2. Reference Signal Construction (`reference_construction.py`):
   - 3M-tap MISO delay-line regressor construction: m_hat[n] = sum_{k=1}^3 w_k^T[n] a_k[n].
3. Adaptive MISO Filtering + Late Fusion (`LMS_algorithms.py` & `fusion.py`):
   - Parallel filtering of `ppg1` and `ppg2` against `acc` (3, N) using `rls` (or the Project 1
     optimal `nlms` / `vs_lms` benchmarks) followed by SNR-Weighted Late Fusion.
4. Heart Rate Detection (`bpm_detection.py`):
   - Estimates BPM per window via `zero_padded_periodogram_bpm` or `elgendi_systolic_peak_bpm`.
"""

import numpy as np

try:
    from .dataset_preparation import FS
    from .preprocessing import preprocess_signal
    from .LMS_algorithms import (
        OPTIMAL_NLMS_HR_CONFIG,
        OPTIMAL_NLMS_PARAMS,
        OPTIMAL_VS_LMS_HR_CONFIG,
        OPTIMAL_VS_LMS_PARAMS,
        lms_bucket,
    )
    from .fusion import snr_weighted_late_fusion
    from .bpm_detection import estimate_bpm_window
except ImportError:
    from dataset_preparation import FS
    from preprocessing import preprocess_signal
    from LMS_algorithms import (
        OPTIMAL_NLMS_HR_CONFIG,
        OPTIMAL_NLMS_PARAMS,
        OPTIMAL_VS_LMS_HR_CONFIG,
        OPTIMAL_VS_LMS_PARAMS,
        lms_bucket,
    )
    from fusion import snr_weighted_late_fusion
    from bpm_detection import estimate_bpm_window


def reconstruct_heart_signal(
    window_5d: np.ndarray,
    lms_algorithm: str = "rls",
    M: int | None = None,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    return_details: bool = False,
    **lms_params,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Reconstruct the clean cardiac signal `e[n]` from a single 5D window `(5, N)`.

    Parameters
    ----------
    window_5d : np.ndarray
        5-dimensional signal window of shape `(5, N)` with channels
        `[ppg1, ppg2, accx, accy, accz]`.
    lms_algorithm : {'rls', 'nlms', 'vs_lms'}, default='rls'
        Adaptive filtering algorithm to execute in the parallel filtering stage.
    M : int, optional
        Number of FIR taps per accelerometer axis (total MISO weights = 3 * M).
    lo : float, default=0.5
        Bandpass lower cutoff frequency in Hz (30 BPM = 0.5 Hz).
    hi : float, default=4.0
        Bandpass upper cutoff frequency in Hz (240 BPM = 4.0 Hz).
    fs : float, default=125
        Sampling frequency in Hz.
    return_details : bool, default=False
        If True, returns `(e_fused, details_dict)`.
    **lms_params
        Hyperparameters for the selected adaptive filter.

    Returns
    -------
    e_fused : np.ndarray
        Cleared and fused cardiac signal window of shape `(N,)`.
    details : dict, optional
        Returned only when `return_details=True` (includes `ppg1`, `ppg2`, `acc`, `accx`, `accy`, `accz`).
    """
    ppg1, ppg2, acc = preprocess_signal(window_5d, lo=lo, hi=hi, fs=fs)

    if return_details:
        e_fused, details = lms_bucket(
            ppg1,
            ppg2,
            acc,
            algorithm=lms_algorithm,
            M=M,
            return_details=True,
            **lms_params,
        )
        details["ppg1"] = ppg1
        details["ppg2"] = ppg2
        details["acc"] = acc
        details["accx"] = acc[0]
        details["accy"] = acc[1]
        details["accz"] = acc[2]
        details["d_fused"] = snr_weighted_late_fusion(ppg1, ppg2, fs=fs)
        return e_fused, details

    return lms_bucket(
        ppg1,
        ppg2,
        acc,
        algorithm=lms_algorithm,
        M=M,
        return_details=False,
        **lms_params,
    )


def predict_window_bpm(
    window_5d: np.ndarray,
    lms_algorithm: str = "rls",
    M: int | None = None,
    lms_params: dict | None = None,
    hr_algorithm: str = "periodogram",
    hr_params: dict | None = None,
    fs: float = FS,
) -> float:
    """Reconstruct the heart signal and estimate BPM for a single 5D window `(5, N)`."""
    lms_kwargs = dict(lms_params) if lms_params else {}
    hr_kwargs = dict(hr_params) if hr_params else {}

    ppg1, ppg2, acc = preprocess_signal(window_5d, fs=fs)
    e_fused = lms_bucket(
        ppg1,
        ppg2,
        acc,
        algorithm=lms_algorithm,
        M=M,
        return_details=False,
        **lms_kwargs,
    )
    return estimate_bpm_window(e_fused, hr_algorithm=hr_algorithm, fs=fs, **hr_kwargs)


class HeartRateEstimatorModel:
    """Configured end-to-end Heart Rate Estimation model encapsulating an Adaptive Filter + HR pipeline.

    Processes each windowed sequence `(5, N)` individually, making it directly applicable
    to validation folds and the test set (`IEEEPPG_TEST.ts`).
    """

    def __init__(
        self,
        lms_algorithm: str = "rls",
        M: int = 12,
        lms_params: dict | None = None,
        hr_algorithm: str = "periodogram",
        hr_params: dict | None = None,
        fs: float = FS,
    ) -> None:
        self.lms_algorithm = lms_algorithm
        self.M = int(M)
        self.lms_params = dict(lms_params) if lms_params else {}
        self.hr_algorithm = hr_algorithm
        self.hr_params = dict(hr_params) if hr_params else {}
        self.fs = fs

    @classmethod
    def from_project1_optimal(cls, algorithm: str, fs: float = FS) -> "HeartRateEstimatorModel":
        """Instantiate `nlms` or `vs_lms` with the optimal hyperparameters obtained in Project 1."""
        key = algorithm.strip().lower().replace("-", "_")
        if key == "nlms":
            params = dict(OPTIMAL_NLMS_PARAMS)
            M = int(params.pop("M"))
            return cls(
                lms_algorithm="nlms",
                M=M,
                lms_params=params,
                hr_algorithm=OPTIMAL_NLMS_HR_CONFIG["hr_algorithm"],
                hr_params=dict(OPTIMAL_NLMS_HR_CONFIG["hr_params"]),
                fs=fs,
            )
        if key in ("vs_lms", "vslms"):
            params = dict(OPTIMAL_VS_LMS_PARAMS)
            M = int(params.pop("M"))
            return cls(
                lms_algorithm="vs_lms",
                M=M,
                lms_params=params,
                hr_algorithm=OPTIMAL_VS_LMS_HR_CONFIG["hr_algorithm"],
                hr_params=dict(OPTIMAL_VS_LMS_HR_CONFIG["hr_params"]),
                fs=fs,
            )
        raise ValueError(f"Project 1 optimal model only available for 'nlms' or 'vs_lms', got '{algorithm}'.")

    def reconstruct_window(
        self, window_5d: np.ndarray, return_details: bool = False
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """Return the reconstructed cardiac signal `e_fused[n]` for a single `(5, N)` window."""
        return reconstruct_heart_signal(
            window_5d,
            lms_algorithm=self.lms_algorithm,
            M=self.M,
            fs=self.fs,
            return_details=return_details,
            **self.lms_params,
        )

    def predict_one(self, window_5d: np.ndarray) -> float:
        """Predict BPM for a single `(5, N)` window."""
        return predict_window_bpm(
            window_5d,
            lms_algorithm=self.lms_algorithm,
            M=self.M,
            lms_params=self.lms_params,
            hr_algorithm=self.hr_algorithm,
            hr_params=self.hr_params,
            fs=self.fs,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict BPM for a single window `(5, N)` or a dataset of windows `(n_windows, 5, N)`."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            return np.array([self.predict_one(X)])
        ppg1_all, ppg2_all, acc_all = preprocess_signal(X, fs=self.fs)
        n_win = len(X)
        bpm_preds = np.empty(n_win, dtype=float)
        for i in range(n_win):
            e_fused = lms_bucket(
                ppg1_all[i],
                ppg2_all[i],
                acc_all[i],
                algorithm=self.lms_algorithm,
                M=self.M,
                return_details=False,
                **self.lms_params,
            )
            bpm_preds[i] = estimate_bpm_window(
                e_fused,
                hr_algorithm=self.hr_algorithm,
                fs=self.fs,
                **self.hr_params,
            )
        return bpm_preds

    def predict_with_signals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict BPM and return the cleaned signals `e_fused` in a single pass over `X`."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        n_win, _, N = X.shape
        ppg1_all, ppg2_all, acc_all = preprocess_signal(X, fs=self.fs)
        bpm_preds = np.empty(n_win, dtype=float)
        cleaned_signals = np.empty((n_win, N), dtype=float)

        for i in range(n_win):
            e_fused = lms_bucket(
                ppg1_all[i],
                ppg2_all[i],
                acc_all[i],
                algorithm=self.lms_algorithm,
                M=self.M,
                return_details=False,
                **self.lms_params,
            )
            cleaned_signals[i] = e_fused
            bpm_preds[i] = estimate_bpm_window(
                e_fused,
                hr_algorithm=self.hr_algorithm,
                fs=self.fs,
                **self.hr_params,
            )
        return bpm_preds, cleaned_signals
