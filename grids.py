"""Validation grids that place an explicit node at the free boundary.

Uniform refinement of a free-boundary problem samples the kink at a
different phase on every mesh.  That phase effect can dominate the
full-grid maximum norm and obscure the order of the spatial stencil.  The
helpers here are used only in verification experiments: they put one node
exactly at the supplied boundary estimate and then refine a uniform or
clustered family of grids around it.
"""

from __future__ import annotations

import numpy as np


def boundary_aligned_grid(N: int, center: float, s_max: float) -> np.ndarray:
    """Return a piecewise-uniform grid with a node at ``center``.

    The left and right intervals are uniform separately.  Their numbers of
    cells are chosen so that ``len(grid) == N + 1`` and the node at
    ``center`` is retained exactly.
    """
    if N < 4:
        raise ValueError("N must be at least 4")
    if not 0.0 < center < s_max:
        raise ValueError("center must lie strictly inside (0, s_max)")

    n_left = int(round(N * center / s_max))
    n_left = min(max(n_left, 2), N - 2)
    left = np.linspace(0.0, center, n_left + 1)
    right = np.linspace(center, s_max, N - n_left + 1)
    return np.concatenate((left, right[1:]))


def boundary_clustered_grid(
    N: int,
    center: float,
    s_max: float,
    strength: float = 16.0,
    width: float = 15.0,
    fine_points: int = 40001,
) -> np.ndarray:
    """Return a clustered grid with a node at ``center``.

    The left side is uniform.  On the right, a Gaussian density
    ``1 + strength * exp(-((S-center)/width)**2)`` is inverted so that the
    grid remains fine around the free boundary and coarse in the far field.
    """
    if N < 4:
        raise ValueError("N must be at least 4")
    if not 0.0 < center < s_max:
        raise ValueError("center must lie strictly inside (0, s_max)")
    if strength < 0.0 or width <= 0.0:
        raise ValueError("strength must be non-negative and width positive")

    n_left = int(round(N * center / s_max))
    n_left = min(max(n_left, 2), N - 2)
    n_right = N - n_left

    left = np.linspace(0.0, center, n_left + 1)
    fine = np.linspace(center, s_max, fine_points)
    density = 1.0 + strength * np.exp(-((fine - center) / width) ** 2)
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(fine)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    targets = np.linspace(0.0, cumulative[-1], n_right + 1)
    right = np.interp(targets, cumulative, fine)
    right[0] = center
    right[-1] = s_max
    return np.concatenate((left, right[1:]))
