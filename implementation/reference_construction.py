"""
Reference signal construction module for MISO (Multiple-Input Single-Output) adaptive filtering.

Constructs the spatio-temporal regressor vectors from the 3 accelerometer axes
(`acc_x, acc_y, acc_z`) with configurable tap delay stride (dilation factor `delay_step = D`)
so that the motion artifact estimate at sample `n` is given by:

    m_hat[n] = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]

where:
    - a_k[n] = [a_k[n], a_k[n - D], a_k[n - 2D], ..., a_k[n - (M - 1)D]]^T  (M dilated taps for axis k)
    - w_k[n] = [w_{k,0}[n], w_{k,1}[n], ..., w_{k,M-1}[n]]^T                (M filter weights for axis k)
    - a[n]   = [a_1^T[n], a_2^T[n], a_3^T[n]]^T in R^{3M}
    - w[n]   = [w_1^T[n], w_2^T[n], w_3^T[n]]^T in R^{3M}

Because the preprocessed signals are bandlimited to [0.5, 4.0] Hz at fs = 125 Hz
(oversampled ~15.6x relative to f_max = 4.0 Hz), a dilation stride D > 1 breaks
adjacent-sample collinearity, reduces the condition number kappa(R_a) by orders of
magnitude, and expands the temporal aperture T_aperture = (M - 1) * D / fs without
increasing the number of adaptive parameters L = 3M.
"""

import numpy as np


def build_U(acc: np.ndarray, M: int, delay_step: int = 1) -> np.ndarray:
    """Build the MISO spatio-temporal regressor matrix U of shape (N, 3 * M).

    Given 3 accelerometer reference channels (`acc_x, acc_y, acc_z`) of shape `(3, N)`,
    filter tap order `M`, and tap delay stride `delay_step = D`, row `n` of `U` is the
    3M-dimensional stacked regressor vector:
        [acc_x(n), acc_x(n - D), ..., acc_x(n - (M - 1)D),
         acc_y(n), acc_y(n - D), ..., acc_y(n - (M - 1)D),
         acc_z(n), acc_z(n - D), ..., acc_z(n - (M - 1)D)]
    with zero-padding for samples before `n = 0`.

    Parameters
    ----------
    acc : np.ndarray
        Reference accelerometer channels of shape `(3, N)` (where rows are `x, y, z`
        and `N` is the number of samples in the window).
    M : int
        Filter order (number of FIR taps per accelerometer axis; total taps = 3 * M).
    delay_step : int, default=1
        Tap delay stride (dilation factor `D >= 1`) in samples between consecutive FIR taps.

    Returns
    -------
    U : np.ndarray
        Regressor matrix of shape `(N, 3 * M)` where row `n` is `a[n]^T`.
    """
    acc = np.asarray(acc, dtype=float)
    if acc.ndim == 1:
        acc = acc[np.newaxis, :]
    n_ref, N = acc.shape
    M = int(M)
    D = max(1, int(delay_step))
    max_lag = (M - 1) * D
    U = np.zeros((N, n_ref * M), dtype=float)
    for k in range(n_ref):
        padded = np.concatenate([np.zeros(max_lag, dtype=float), acc[k]])
        for i in range(M):
            offset = max_lag - i * D
            U[:, k * M + i] = padded[offset : offset + N]
    return U
