"""Shared parameters and local-volatility profile for Stage 2."""

from __future__ import annotations

import numpy as np


K = 100.0
R = 0.05
Q = 0.02
S_MAX = 2400.0


def local_volatility(S: np.ndarray, K_value: float = K) -> np.ndarray:
    """Return ``0.20 * (1 + 0.35 * tanh(log(S/K)/0.40))``."""
    S = np.asarray(S, dtype=float)
    S_safe = np.maximum(S, np.finfo(float).tiny)
    return 0.20 * (1.0 + 0.35 * np.tanh(np.log(S_safe / K_value) / 0.40))


SIGMA_INFINITY = 0.20 * (1.0 + 0.35)
