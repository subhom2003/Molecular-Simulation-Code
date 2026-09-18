"""
Autocorrelation + integrated autocorrelation time.

The autocorrelation is computed via FFT for efficiency::

    ACF(t) = sum_i (x[i] - mean) (x[i+t] - mean) / sum_i (x[i]-mean)^2

The integrated autocorrelation time IAT is::

    tau = 1 + 2 * sum_{t=1..} ACF(t)     (truncated at where ACF first crosses 0)

References
- A. Sokal, CERN Summer School 1996.
- N. Madras & A. D. Sokal, J. Stat. Phys. 50, 109 (1988).
"""

from __future__ import annotations
import numpy as np


def autocorrelation(data: np.ndarray, max_lag: int | None = None,
                     normalize: bool = True) -> np.ndarray:
    """Normalized autocorrelation function via FFT."""
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    if n == 0:
        return np.array([])
    x -= x.mean()
    # zero-pad to 2n for circular-convolution safe
    n2 = 1
    while n2 < 2 * n:
        n2 *= 2
    f = np.fft.rfft(x, n2)
    acf = np.fft.irfft(f * np.conjugate(f), n2)[:n]
    if normalize:
        acf /= acf[0]
    if max_lag is not None and max_lag < n:
        acf = acf[:max_lag]
    return acf


def integrated_autocorrelation_time(data: np.ndarray,
                                     truncate_at_negative: bool = True) -> float:
    """IAT computed by truncating ACF sum at the first negative crossing."""
    acf = autocorrelation(data, normalize=True)
    n = acf.size
    if n < 4:
        return float("nan")
    if truncate_at_negative:
        first_neg = np.where(acf < 0)[0]
        if first_neg.size:
            acf = acf[:first_neg[0]]
    # tau = 1 + 2 * sum_{t=1..} ACF(t)
    if acf.size <= 1:
        return 1.0
    return float(1.0 + 2.0 * np.sum(acf[1:]) / acf[0])
