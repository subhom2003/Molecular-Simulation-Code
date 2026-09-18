"""Analysis: block averaging, autocorrelation, validation report."""
from .block_average import block_average, optimal_block_size
from .autocorrelation import autocorrelation, integrated_autocorrelation_time
from .summary import build_validation_report

__all__ = [
    "block_average", "optimal_block_size",
    "autocorrelation", "integrated_autocorrelation_time",
    "build_validation_report",
]
