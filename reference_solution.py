"""Independent semi-analytic reference for the perpetual American put.

The reference does not use a finite-difference grid.  On the continuation
region it integrates the free-boundary ODE

    V'' = [-(r-q) S V' + r V] / (0.5 sigma(S)^2 S^2)

with ``V(S*) = K - S*`` and ``V'(S*) = -1``.  The unknown boundary is found
by shooting.  For the infinite-domain solution the far-field condition is
the decaying Robin tail ``S V' = lambda V`` at a very large price.  For the
finite-domain problem with the report's truncation condition, the shooting
condition is ``V(S_max) = 0``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from local_vol_model import K, Q, R, SIGMA_INFINITY, S_MAX, local_volatility


@dataclass(frozen=True)
class ShootingReference:
    """Semi-analytic reference values on ``[0, S_max]``."""

    S_star: float
    S_max: float
    S: np.ndarray
    V: np.ndarray
    domain: str
    _solution: object = field(repr=False, compare=False)

    def evaluate(self, S: np.ndarray) -> np.ndarray:
        """Evaluate the reference at arbitrary prices without interpolation error."""
        points = np.asarray(S, dtype=float)
        values = np.empty_like(points)
        exercise = points <= self.S_star
        values[exercise] = K - points[exercise]
        continuation = ~exercise
        if np.any(continuation):
            values[continuation] = self._solution.sol(points[continuation])[0]
        return values

    def value_at(self, target: float) -> float:
        """Evaluate the reference at one target price."""
        return float(self.evaluate(np.array([target]))[0])


def decaying_exponent(sigma: float, r: float = R, q: float = Q) -> float:
    """Return the decaying root for constant ``sigma`` and ``q``."""
    a = 0.5 * sigma * sigma
    b = (r - q) - a
    discriminant = b * b + 4.0 * a * r
    return (-b - math.sqrt(discriminant)) / (2.0 * a)


def _ode_rhs(
    S: float,
    state: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray],
    r: float,
    q: float,
) -> list[float]:
    V, dV = state
    sigma_value = float(np.asarray(sigma(np.array([S])))[0])
    diffusion = 0.5 * sigma_value * sigma_value * S * S
    drift = (r - q) * S
    return [dV, (-drift * dV + r * V) / diffusion]


def _integrate(
    start: float,
    end: float,
    initial: tuple[float, float],
    sigma: Callable[[np.ndarray], np.ndarray],
    r: float,
    q: float,
    dense_output: bool = False,
):
    return solve_ivp(
        lambda S, y: _ode_rhs(S, y, sigma, r, q),
        (start, end),
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        dense_output=dense_output,
    )


def _infinite_boundary(
    sigma: Callable[[np.ndarray], np.ndarray],
    r: float,
    q: float,
    K_value: float,
    s_inf: float,
) -> tuple[float, float]:
    sigma_inf = float(np.asarray(sigma(np.array([s_inf])))[0])
    lam = decaying_exponent(sigma_inf, r=r, q=q)

    def tail_metric(star: float) -> float:
        solution = _integrate(
            s_inf,
            star,
            (1.0, lam / s_inf),
            sigma,
            r,
            q,
        )
        v, dv = solution.y[:, -1]
        return float((K_value - star) * dv + v)

    return brentq(tail_metric, 0.30 * K_value, 0.95 * K_value,
                  xtol=1e-12, rtol=1e-14), lam


def _finite_boundary(
    sigma: Callable[[np.ndarray], np.ndarray],
    r: float,
    q: float,
    K_value: float,
    s_max: float,
) -> float:
    def endpoint_metric(star: float) -> float:
        solution = _integrate(
            star,
            s_max,
            (K_value - star, -1.0),
            sigma,
            r,
            q,
        )
        return float(solution.y[0, -1])

    return brentq(endpoint_metric, 0.30 * K_value, 0.95 * K_value,
                  xtol=1e-12, rtol=1e-14)


def build_reference(
    sigma: Callable[[np.ndarray], np.ndarray] | float = local_volatility,
    K_value: float = K,
    r: float = R,
    q: float = Q,
    s_max: float = S_MAX,
    domain: str = "infinite",
    s_inf: float = 1.0e5,
    n_values: int = 20001,
) -> ShootingReference:
    """Build the semi-analytic reference on ``[0, s_max]``.

    ``domain='infinite'`` uses the asymptotic decaying tail and therefore
    gives the true half-line value.  ``domain='finite'`` imposes
    ``V(s_max)=0`` and gives the exact solution of the finite-domain problem
    used by the production solver.
    """
    if domain not in ("infinite", "finite"):
        raise ValueError("domain must be 'infinite' or 'finite'")
    if isinstance(sigma, (float, int)):
        constant = float(sigma)
        sigma_fn = lambda S: np.full(np.asarray(S).shape, constant, dtype=float)
    else:
        sigma_fn = sigma

    if domain == "infinite":
        S_star, _ = _infinite_boundary(sigma_fn, r, q, K_value, s_inf)
    else:
        S_star = _finite_boundary(sigma_fn, r, q, K_value, s_max)

    solution = _integrate(
        S_star,
        s_max,
        (K_value - S_star, -1.0),
        sigma_fn,
        r,
        q,
        dense_output=True,
    )
    S = np.linspace(0.0, s_max, n_values)
    reference = ShootingReference(
        S_star=float(S_star),
        S_max=float(s_max),
        S=S,
        V=np.empty_like(S),
        domain=domain,
        _solution=solution,
    )
    reference.V[:] = reference.evaluate(S)
    return reference


def validate_constant_volatility_reference() -> dict[str, float]:
    """Self-check the shooting solver against the closed-form benchmark."""
    from stage1_validation import exact_free_boundary, exact_perpetual_put

    reference = build_reference(sigma=0.20, q=0.0, domain="infinite")
    exact_star = exact_free_boundary(sigma=0.20, q=0.0)
    exact_100 = float(exact_perpetual_put(np.array([100.0]), sigma=0.20, q=0.0)[0])
    return {
        "S_star_error": abs(reference.S_star - exact_star),
        "V100_error": abs(reference.value_at(100.0) - exact_100),
    }
