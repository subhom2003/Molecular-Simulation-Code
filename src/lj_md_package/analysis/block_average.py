"""
=============================================================================
Block averaging -- Flyvbjerg & Petersen (1989) error analysis
=============================================================================

Given a correlated time series x[0..n-1], the standard error of its mean is
not std/sqrt(n) but std/sqrt(n_eff) where n_eff depends on the integrated
autocorrelation time.  The Flyvbjerg-Petersen (FP) algorithm successively
halves the data and tracks the apparent std error until the doubling
plateaus, which gives the asymptotic value.

:func:`block_average` returns ``(mean, error, std)`` for a chosen block size.
:func:`optimal_block_size` runs FP and returns ``(block_sz, errors, std)``.
=============================================================================
"""

from __future__ import annotations
import numpy as np


def block_average(data: np.ndarray, block_size: int | None = None
                  ) -> tuple[float, float, float]:
    """Compute (mean, std_error, std) of the data using FP blocking.

    If ``block_size`` is None, an FP plateau search is run to identify the
    optimal block size.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if block_size is None:
        block_size, _, _ = optimal_block_size(x)
    block_size = max(int(block_size), 1)
    nb = n // block_size
    if nb < 2:
        # too few blocks: just use full-sample std
        return float(x.mean()), float(x.std(ddof=0)) / max(np.sqrt(1), 1), float(x.std(ddof=0))
    blocks = x[:nb * block_size].reshape(nb, block_size).mean(axis=1)
    mean = float(blocks.mean())
    std = float(blocks.std(ddof=1)) if nb > 1 else float("nan")
    error = std / np.sqrt(nb) if not np.isnan(std) else float("nan")
    return mean, error, std


def optimal_block_size(data: np.ndarray, max_trials: int = 50
                        ) -> tuple[int, np.ndarray, np.ndarray]:
    """FP plateau search: returns (chosen_block_size, errors, stds)."""
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    if n < 4:
        return 1, np.array([float("nan")]), np.array([float("nan")])
    sizes = []
    errs = []
    stds = []
    b = 1
    while True:
        nb = n // b
        if nb < 2:
            break
        blocks = x[:nb * b].reshape(nb, b).mean(axis=1)
        if nb >= 2:
            std = float(blocks.std(ddof=1)) / np.sqrt(nb)
            errs.append(std)
            stds.append(float(blocks.std(ddof=1)))
            sizes.append(b)
        b *= 2
        if b > n // 2:
            break
    errs = np.array(errs)
    stds = np.array(stds)
    sizes = np.array(sizes)
    if errs.size == 0:
        return 1, errs, stds
    # Plateau detection: take the last index where the gradient (in absolute)
    # changes by < 5 % of max.
    grad = np.abs(np.gradient(errs))
    if errs.size >= 2 and not np.all(np.isnan(errs)):
        threshold = 0.05 * np.nanmax(errs)
        # last index where grad above threshold (we want the plateau start AFTER that)
        plateau_idx = np.where(grad < threshold)[0]
        if plateau_idx.size:
            chosen = sizes[plateau_idx[0] if plateau_idx[0] > 0 else 1
                              if errs.size > 1 else 0]
        else:
            chosen = int(sizes[-1])
    else:
        chosen = int(sizes[-1])
    return int(chosen), errs, stds
