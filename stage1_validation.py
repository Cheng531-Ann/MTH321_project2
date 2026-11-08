"""Perpetual American put via an LCP and policy iteration.

The spatial operator is discretized with exponential fitting
(Scharfetter--Gummel).  The fitted coefficients are an M-matrix at every
cell Peclet number, so no limiter or start-up correction is required.  The
resulting linear complementarity problem is solved by policy iteration.

The production solvers use the truncation boundary condition V(S_max) = 0.
An optional ``V_right`` argument is retained so the same code can also be
run with an exact far-field value for validation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


SLACK_TOL_FACTOR = 1e-10
POLICY_TOL = 1e-10


@dataclass
class GridOperator:
    """Fitted discrete operator A = -L_h and boundary data on a uniform grid."""

    N: int
    S: np.ndarray
    psi: np.ndarray
    lower: np.ndarray
    diag: np.ndarray
    upper: np.ndarray
    rhs: np.ndarray
    V_left: float
    V_right: float
    h: float


@dataclass
class LCPResult:
    """Numerical solution and diagnostics for one grid."""

    N: int
    S: np.ndarray
    V: np.ndarray
    S_free: float
    max_error: float
    max_error_interior: float
    iterations: int
    residual: float
    S_lo: float = float("nan")
    S_hi: float = float("nan")
    min_a: float = float("nan")
    min_b: float = float("nan")
    complementarity: float = float("nan")
    value_bounds_ok: bool = False
    monotone: bool = False
    convex: bool = False
    single_contact_set: bool = False


def exact_free_boundary(
    K: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
) -> float:
    """Return the exact perpetual American put exercise boundary.

    The decaying root of ``0.5 sigma^2 S^2 V'' + (r-q) S V' - r V = 0`` is
    ``lambda = (-b - sqrt(b^2 + 4 a r)) / (2 a)`` with ``a = 0.5 sigma^2`` and
    ``b = (r - q) - a``.  Smooth pasting gives ``S* = lambda K / (lambda - 1)``.
    """
    a = 0.5 * sigma * sigma
    b = (r - q) - a
    discriminant = b * b + 4.0 * a * r
    if discriminant < 0.0:
        raise ValueError("invalid parameters: negative discriminant")
    lam = (-b - math.sqrt(discriminant)) / (2.0 * a)
    if lam >= -1.0:
        raise ValueError("no finite exercise boundary for these parameters")
    return lam * K / (lam - 1.0)


def exact_perpetual_put(
    S: np.ndarray,
    K: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    S_star: float | None = None,
) -> np.ndarray:
    """Exact perpetual American put value for the supplied parameters."""
    S = np.asarray(S, dtype=float)
    if S_star is None:
        S_star = exact_free_boundary(K=K, r=r, q=q, sigma=sigma)

    a = 0.5 * sigma * sigma
    b = (r - q) - a
    lam = (-b - math.sqrt(b * b + 4.0 * a * r)) / (2.0 * a)

    V = np.empty_like(S)
    exercise = S <= S_star
    V[exercise] = K - S[exercise]
    V[~exercise] = (K - S_star) * (S[~exercise] / S_star) ** lam
    return V


def _bernstein(z: np.ndarray) -> np.ndarray:
    """Evaluate B(z) = z / (exp(z) - 1) without cancellation or overflow."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    small = np.abs(z) < 1e-8
    out[small] = 1.0 - 0.5 * z[small] + z[small] ** 2 / 12.0

    positive = (~small) & (z > 0.0)
    negative = (~small) & (z < 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        out[positive] = z[positive] / np.expm1(z[positive])
        w = -z[negative]
        out[negative] = w / (-np.expm1(-w))
    return out


def build_fitted_operator(
    N: int,
    K: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float | np.ndarray | Callable[[np.ndarray], np.ndarray] = 0.2,
    S_max: float = 2400.0,
    V_right: float = 0.0,
) -> GridOperator:
    """Build A = -L_h with exponential fitting on a uniform grid.

    The fitted coefficients are

        Pe_i = b_i h / a_i,
        lower_i = -(a_i/h^2) B(Pe_i),
        upper_i = -(a_i/h^2) B(-Pe_i),
        diag_i  =  (a_i/h^2) [B(Pe_i) + B(-Pe_i)] + r,

    with ``a_i = 0.5 sigma(S_i)^2 S_i^2`` and ``b_i = (r-q) S_i``.  Since
    ``B(z) > 0``, both off-diagonals are non-positive and the diagonal is
    positive for every Peclet number.  The right boundary is applied as
    ``V(S_max) = V_right``; production runs use ``V_right = 0``.
    """
    if N < 2:
        raise ValueError("N must be at least 2")
    if S_max <= 0.0 or K <= 0.0:
        raise ValueError("S_max and K must be positive")

    h = S_max / N
    S = np.linspace(0.0, S_max, N + 1)
    S_int = S[1:-1]
    n_int = N - 1
    psi = np.maximum(K - S, 0.0)

    if callable(sigma):
        sigma_int = np.asarray(sigma(S_int), dtype=float)
    elif np.isscalar(sigma):
        sigma_int = np.full(n_int, float(sigma), dtype=float)
    else:
        sigma_int = np.asarray(sigma, dtype=float)
    if sigma_int.shape != S_int.shape:
        raise ValueError("sigma must be a scalar or have shape (N-1,)")
    if np.any(sigma_int <= 0.0):
        raise ValueError("sigma must be positive")

    a_int = 0.5 * sigma_int**2 * S_int**2
    b_int = (r - q) * S_int
    peclet = b_int * h / a_int

    bp = _bernstein(peclet)
    bm = _bernstein(-peclet)
    scale = a_int / h**2

    lower = -scale * bp
    upper = -scale * bm
    diag = scale * (bp + bm) + r

    rhs = np.zeros(n_int)
    rhs[0] = -lower[0] * K
    rhs[-1] += -upper[-1] * V_right

    return GridOperator(
        N=N,
        S=S,
        psi=psi,
        lower=lower,
        diag=diag,
        upper=upper,
        rhs=rhs,
        V_left=K,
        V_right=float(V_right),
        h=h,
    )


def _assert_m_matrix(op: GridOperator) -> None:
    """Raise if the fitted operator fails the M-matrix property."""
    off_diagonal = np.concatenate((op.lower, op.upper))
    scale = max(1.0, float(np.max(np.abs(op.diag))))
    if np.any(op.diag <= 0.0) or np.any(off_diagonal > 1e-14 * scale):
        raise ArithmeticError("fitted operator failed the M-matrix check")


def _system_matrix(op: GridOperator) -> sp.csr_matrix:
    n_int = op.N - 1
    return sp.diags(
        (op.lower[1:], op.diag, op.upper[:-1]),
        offsets=(-1, 0, 1),
        shape=(n_int, n_int),
        format="csr",
    )


def solve_lcp(
    op: GridOperator,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, int, float]:
    """Solve the discrete LCP ``min(A V - rhs, V - psi) = 0``.

    The iteration starts from ``V = psi``.  Active rows are those with
    ``V <= psi + 1e-10``; a row is released when its full-operator residual is
    negative.  The stopping rule is ``max|dV| < tol`` together with a stable
    active set.
    """
    _assert_m_matrix(op)

    N = op.N
    n_int = N - 1
    psi = op.psi
    psi_int = psi[1:-1]
    A_full = _system_matrix(op)

    V = psi.copy()
    V[0] = op.V_left
    V[-1] = op.V_right

    residual = math.inf
    iterations = 0
    previous_active: np.ndarray | None = None

    for iterations in range(1, max_iter + 1):
        V_old = V.copy()

        lcp_residual = A_full @ V[1:-1] - op.rhs
        active = V <= psi + POLICY_TOL
        active_int = active[1:-1].copy()
        active_int[active_int & (lcp_residual < -POLICY_TOL)] = False
        policy_is_stable = (
            previous_active is not None
            and np.array_equal(active_int, previous_active)
        )
        previous_active = active_int.copy()

        diag = op.diag.copy()
        diag[active_int] = 1.0

        if n_int > 1:
            lower = op.lower[1:].copy()
            upper = op.upper[:-1].copy()
            lower[active_int[1:]] = 0.0
            upper[active_int[:-1]] = 0.0
        else:
            lower = np.empty(0, dtype=float)
            upper = np.empty(0, dtype=float)

        A = sp.diags(
            (lower, diag, upper),
            offsets=(-1, 0, 1),
            shape=(n_int, n_int),
            format="csr",
        )

        b = op.rhs.copy()
        b[active_int] = psi_int[active_int]
        V_new_int = spsolve(A, b)
        V[1:-1] = np.maximum(V_new_int, psi_int)
        V[0] = op.V_left
        V[-1] = op.V_right

        residual = float(np.max(np.abs(V - V_old)))
        if residual < tol and policy_is_stable:
            break
    else:
        raise RuntimeError(
            f"policy iteration did not converge in {max_iter} iterations "
            f"(last residual={residual:.3e})"
        )

    return V, iterations, residual


def lcp_residuals(
    op: GridOperator,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interior ``a_h = A V - rhs`` and ``b_h = V - psi``."""
    V_int = V[1:-1]
    a_h = op.diag * V_int - op.rhs
    if V_int.size > 1:
        a_h[1:] += op.lower[1:] * V_int[:-1]
        a_h[:-1] += op.upper[:-1] * V_int[1:]
    b_h = V_int - op.psi[1:-1]
    return a_h, b_h


def extract_free_boundary(
    S: np.ndarray,
    b_h: np.ndarray,
    tol_slack: float,
) -> tuple[float, float, float]:
    """Bracket and interpolate the free boundary.

    The last contact node is the last index with ``b_h <= tol_slack``; the
    next node is the first continuation node.  Since ``b_h`` behaves like
    ``c (S - S*)^2`` in the continuation region, ``sqrt(b_h)`` is locally
    linear.  A linear extrapolation of ``sqrt(b_h)`` through the first two
    continuation nodes gives the crossing, which is clamped to the bracket
    ``[S_lo, S_hi]``.  If that extrapolation is unavailable, the midpoint of
    the bracket is used, so ``|S_h - S*| <= h/2``.
    """
    contact = b_h <= tol_slack
    first_continuation = np.flatnonzero(~contact)
    if first_continuation.size == 0:
        return float("nan"), float("nan"), float("nan")

    i1 = int(first_continuation[0])
    if i1 == 0:
        return float("nan"), float("nan"), float("nan")

    i0 = i1 - 1
    S_lo = float(S[i0])
    S_hi = float(S[i1])

    if i1 + 1 < S.size and b_h[i1] > 0.0 and b_h[i1 + 1] > 0.0:
        s1 = math.sqrt(float(b_h[i1]))
        s2 = math.sqrt(float(b_h[i1 + 1]))
        if s2 > s1:
            step = float(S[i1 + 1] - S[i1])
            estimate = float(S[i1]) + step * (-s1) / (s2 - s1)
            estimate = min(max(estimate, S_lo), S_hi)
            return S_lo, S_hi, estimate

    return S_lo, S_hi, 0.5 * (S_lo + S_hi)


def compute_lcp_diagnostics(
    op: GridOperator,
    V: np.ndarray,
    tol_slack: float | None = None,
) -> dict[str, float | bool]:
    """Compute LCP residuals, bounds, shape checks and the free-boundary bracket."""
    if tol_slack is None:
        tol_slack = SLACK_TOL_FACTOR * max(1.0, abs(op.V_left))

    a_h, b_h = lcp_residuals(op, V)
    complementarity = np.minimum(a_h, b_h)
    psi = op.psi
    K = op.V_left
    shape_tol = 1e-10 * max(1.0, K)

    contact = b_h <= tol_slack
    first_continuation = np.flatnonzero(~contact)
    if contact.size == 0:
        single_contact_set = False
    elif first_continuation.size == 0:
        single_contact_set = bool(contact[0])
    else:
        first = int(first_continuation[0])
        single_contact_set = bool(
            contact[0]
            and np.all(contact[:first])
            and np.all(~contact[first:])
        )

    S_lo, S_hi, S_free = extract_free_boundary(op.S[1:-1], b_h, tol_slack)

    return {
        "min_a": float(np.min(a_h)),
        "min_b": float(np.min(b_h)),
        "complementarity": float(np.max(np.abs(complementarity))),
        "value_bounds_ok": bool(
            np.all(V >= psi - tol_slack) and np.all(V <= K + tol_slack)
        ),
        "monotone": bool(np.all(np.diff(V) <= shape_tol)),
        "convex": bool(np.all(np.diff(V, n=2) >= -shape_tol)),
        "single_contact_set": single_contact_set,
        "S_lo": S_lo,
        "S_hi": S_hi,
        "S_free": S_free,
        "tol_slack": float(tol_slack),
    }


def solve_perpetual_american_put(
    N: int,
    K: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    S_max: float = 2400.0,
    V_right: float = 0.0,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> LCPResult:
    """Solve the perpetual American put LCP with exponential fitting.

    ``V_right`` is the Dirichlet value at ``S_max``.  Production runs use the
    truncation condition ``V(S_max) = 0`` (the default).
    """
    op = build_fitted_operator(
        N=N,
        K=K,
        r=r,
        q=q,
        sigma=sigma,
        S_max=S_max,
        V_right=V_right,
    )
    V, iterations, residual = solve_lcp(op, tol=tol, max_iter=max_iter)
    diagnostics = compute_lcp_diagnostics(op, V)

    V_exact = exact_perpetual_put(op.S, K=K, r=r, q=q, sigma=sigma)
    max_error = float(np.max(np.abs(V - V_exact)))
    max_error_interior = float(np.max(np.abs(V[1:-1] - V_exact[1:-1])))

    return LCPResult(
        N=N,
        S=op.S,
        V=V,
        S_free=float(diagnostics["S_free"]),
        max_error=max_error,
        max_error_interior=max_error_interior,
        iterations=iterations,
        residual=residual,
        S_lo=float(diagnostics["S_lo"]),
        S_hi=float(diagnostics["S_hi"]),
        min_a=float(diagnostics["min_a"]),
        min_b=float(diagnostics["min_b"]),
        complementarity=float(diagnostics["complementarity"]),
        value_bounds_ok=bool(diagnostics["value_bounds_ok"]),
        monotone=bool(diagnostics["monotone"]),
        convex=bool(diagnostics["convex"]),
        single_contact_set=bool(diagnostics["single_contact_set"]),
    )


def observed_order(previous_error: float, current_error: float) -> float:
    """Observed order under a grid-refinement ratio of two."""
    if previous_error == 0.0 or current_error == 0.0:
        return float("nan")
    return math.log2(previous_error / current_error)


def main() -> dict[int, LCPResult]:
    """Run the constant-volatility convergence study."""
    K = 100.0
    r = 0.05
    q = 0.0
    sigma = 0.2
    S_max = 2400.0
    S_star = exact_free_boundary(K=K, r=r, q=q, sigma=sigma)
    grid_sizes = (600, 1200, 2400, 4800)

    results: dict[int, LCPResult] = {}
    previous_error: float | None = None

    print()
    print("Perpetual American Put -- Fitted LCP, V(S_max)=0")
    print(
        f"K={K:.1f}, r={r:.4f}, q={q:.4f}, sigma={sigma:.4f}, "
        f"S_max={S_max:.1f}"
    )
    print("Exponential fitting (Scharfetter-Gummel); no limiter")
    print("Initial iterate V=psi; stop when max|dV|<1e-12 and policy is stable")
    print(
        "Free boundary: bracket where b_h=V_h-psi_h exceeds "
        f"{SLACK_TOL_FACTOR:.0e}*K, sqrt(b_h) interpolation"
    )
    print()
    header = (
        f"{'N':>6}  {'S_lo':>9}  {'S_hi':>9}  {'S_h*':>9}  "
        f"{'Exact S*':>10}  {'|dS*|':>9}  {'Max Error':>12}  "
        f"{'p':>7}  {'Iter':>5}"
    )
    rule = "-" * len(header)
    print(header)
    print(rule)

    for N in grid_sizes:
        result = solve_perpetual_american_put(
            N=N,
            K=K,
            r=r,
            q=q,
            sigma=sigma,
            S_max=S_max,
            V_right=0.0,
        )
        results[N] = result

        error = result.max_error
        order = (
            "      --"
            if previous_error is None
            else f"{observed_order(previous_error, error):7.3f}"
        )
        print(
            f"{N:6d}  {result.S_lo:9.3f}  {result.S_hi:9.3f}  "
            f"{result.S_free:9.3f}  {S_star:10.6f}  "
            f"{abs(result.S_free - S_star):9.3f}  {error:12.4e}  "
            f"{order:>7}  {result.iterations:5d}"
        )
        previous_error = error

    print(rule)
    print()
    print("Max Error is the full-grid infinity norm on all N+1 nodes.")
    print(
        "The right boundary uses V(S_max)=0, so the S_max study measures "
        "truncation rather than imposing the exact answer."
    )
    print(
        "The fitted scheme is second-order away from the free-boundary kink; "
        "|S_h*-S*| is O(h) and the bracket width is h."
    )
    print(
        "The uniform-grid errors above mix the kink phase with the nonzero "
        "right-boundary tail; they are production diagnostics, not the order "
        "table. Run experiment_convergence.py for the aligned second-order "
        "verification table used in Section 4.1."
    )
    print("Each result stores S and V, e.g. results[800].S and results[800].V.")
    print()

    return results


if __name__ == "__main__":
    results = main()

    plot_result = results[2400]
    S_plot = plot_result.S
    V_plot = plot_result.V
    payoff_plot = np.maximum(100.0 - S_plot, 0.0)
    S_star = exact_free_boundary()

    plt.figure(figsize=(8, 5))
    plt.plot(S_plot, V_plot, color="blue", linestyle="-", label="LCP solution")
    plt.plot(
        S_plot,
        payoff_plot,
        color="black",
        linestyle="--",
        label="Payoff",
    )
    plt.axvline(
        S_star,
        color="red",
        linestyle="-",
        label=rf"Exact free boundary $S^*={S_star:.4f}$",
    )
    plt.axvspan(
        plot_result.S_lo,
        plot_result.S_hi,
        color="red",
        alpha=0.12,
        label=(
            rf"Bracket $[{plot_result.S_lo:.2f},"
            rf"{plot_result.S_hi:.2f}]$"
        ),
    )
    plt.xlim(50.0, 100.0)
    plt.xlabel("Stock price S")
    plt.ylabel("Option value V(S)")
    plt.title("Perpetual American Put: Free Boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig("free_boundary_plot.png", dpi=150)
    plt.show()
