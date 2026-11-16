"""Shared numerical helpers for cell prediction."""

import numpy as np

from .. import guards

LOG2_MAX = 1000.0


def exp2(values, log2_max: float | None) -> np.ndarray:
    """Exponentiate on the log2 scale with overflow protection and optional model bounds."""
    array = np.asarray(values, np.float64)
    with np.errstate(invalid="ignore"):
        guards.fired("log2_overflow", int((np.abs(array) > LOG2_MAX).sum()), array.size)
    if log2_max is not None:
        array = guards.clip_abs(array, log2_max, "log2_max")
    return np.power(2.0, np.clip(array, -LOG2_MAX, LOG2_MAX))


def finite(values) -> np.ndarray:
    """Return float64 values with NaN and infinities replaced by zero."""
    return np.nan_to_num(np.asarray(values, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
