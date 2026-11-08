"""Controlled 2D extension: a perpetual American spread-put LCP.

The experiment is deliberately restricted to constant volatilities and zero
correlation.  Under those assumptions the two-dimensional operator separates,
and exponential fitting in each coordinate gives a five-point M-matrix.  The
correlated case is not approximated here: a non-zero correlation would require
a cross-derivative stencil and a separate monotonicity analysis.

The script assembles the sparse system independently of the production 1D
solver.  The only imported numerical routine supplies the exact-boundary
analogue on S2=0, namely the fitted perpetual American put value.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve

from stage1_validation import exact_perpetual_put, solve_perpetual_american_put


K = 100.0
R = 0.05
Q1 = 0.00
Q2 = 0.02
SIGMA1 = 0.20
SIGMA2 = 0.20
RHO = 0.00
L = 300.0
GRID_SIZES = (80, 160, 320)
DOMAIN_GRID = 80
DOMAIN_L_VALUES = (240.0, 300.0, 400.0)
DOMAIN_POINT = (100.0, 50.0)
PRICING_WINDOW = 200.0
POLICY_TOL = 1e-10
SLACK_TOL_FACTOR = 1e-10


def _bernstein(z: np.ndarray) -> np.ndarray:
    """Evaluate B(z)=z/(exp(z)-1) without cancellation or overflow."""
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


@dataclass
class Spread2DOperator:
    """Five-point fitted operator and boundary data."""

    N: int
    L: float
    S: np.ndarray
    psi_int: np.ndarray
    A: sp.csr_matrix
    rhs: np.ndarray
    diag_flat: np.ndarray
    iminus_flat: np.ndarray
    jminus_flat: np.ndarray
    jplus_flat: np.ndarray
    iplus_flat: np.ndarray
    boundary: dict[str, np.ndarray]
    min_diagonal: float
    max_off_diagonal: float
    min_row_sum: float

    @property
    def unknowns(self) -> int:
        return (self.N - 1) ** 2

    def policy_matrix(self, active: np.ndarray) -> sp.csr_matrix:
        """Replace active rows by identity rows for one policy iteration."""
        active_flat = active.ravel()
        keep = ~active_flat
        diagonal = np.where(active_flat, 1.0, self.diag_flat)
        size = self.unknowns
        n = self.N - 1

        return sp.diags(
            (
                self.iminus_flat[n:] * keep[n:],
                self.jminus_flat[1:] * keep[1:],
                diagonal,
                self.jplus_flat[:-1] * keep[:-1],
                self.iplus_flat[:-n] * keep[:-n],
            ),
            offsets=(-n, -1, 0, 1, n),
            shape=(size, size),
            format="csr",
        )


@dataclass
class Spread2DResult:
    """Converged solution and diagnostics on one mesh."""

    N: int
    S: np.ndarray
    V: np.ndarray
    iterations: int
    delta: float
    complementarity: float
    min_slack: float
    obstacle_ok: bool
    contact_fraction: float
    min_diagonal: float
    max_off_diagonal: float
    min_row_sum: float
    elapsed: float


def _fitted_direction(
    S: np.ndarray,
    h: float,
    r: float,
    q: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower, upper, and diffusion part of the diagonal."""
    a = 0.5 * sigma**2 * S**2
    b = (r - q) * S
    peclet = b * h / a
    bp = _bernstein(peclet)
    bm = _bernstein(-peclet)
    scale = a / h**2
    return -scale * bp, -scale * bm, scale * (bp + bm)


def build_spread_operator(
    N: int,
    K: float = K,
    r: float = R,
    q1: float = Q1,
    q2: float = Q2,
    sigma1: float = SIGMA1,
    sigma2: float = SIGMA2,
    L: float = L,
) -> Spread2DOperator:
    """Assemble the five-point fitted operator for a spread-put LCP.

    The payoff is ``psi(S1,S2)=max(K-S1+S2,0)``.  The S2=0 boundary is the
    fitted perpetual-put value with parameters ``(r,q1,sigma1)``.  The S1=0
    boundary is the payoff.  On S1=L the boundary is the larger of the 1D
    far-field put tail and immediate payoff; on S2=L the payoff is used as a
    local exercise/truncation closure.
    """
    if N < 4:
        raise ValueError("N must be at least 4")
    if L <= K:
        raise ValueError("the computational domain must contain the strike")

    h = L / N
    S = np.linspace(0.0, L, N + 1)
    S_int = S[1:-1]
    n = N - 1

    lower1, upper1, diffusion1 = _fitted_direction(
        S_int, h, r, q1, sigma1
    )
    lower2, upper2, diffusion2 = _fitted_direction(
        S_int, h, r, q2, sigma2
    )

    psi_int = np.maximum(K - S_int[:, None] + S_int[None, :], 0.0)
    diagonal = diffusion1[:, None] + diffusion2[None, :] + r

    iminus = np.broadcast_to(lower1[:, None], (n, n)).copy()
    iplus = np.broadcast_to(upper1[:, None], (n, n)).copy()
    jminus = np.broadcast_to(lower2[None, :], (n, n)).copy()
    jplus = np.broadcast_to(upper2[None, :], (n, n)).copy()

    # Remove wrap-around couplings at the first/last row or column.
    iminus[0, :] = 0.0
    iplus[-1, :] = 0.0
    jminus[:, 0] = 0.0
    jplus[:, -1] = 0.0

    iminus_flat = iminus.ravel()
    iplus_flat = iplus.ravel()
    jminus_flat = jminus.ravel()
    jplus_flat = jplus.ravel()
    diag_flat = diagonal.ravel()

    A = sp.diags(
        (
            iminus_flat[n:],
            jminus_flat[1:],
            diag_flat,
            jplus_flat[:-1],
            iplus_flat[:-n],
        ),
        offsets=(-n, -1, 0, 1, n),
        shape=(n * n, n * n),
        format="csr",
    )

    put_tail = float(
        exact_perpetual_put(
            np.asarray([L]), K=K, r=r, q=q1, sigma=sigma1
        )[0]
    )
    put_1d = solve_perpetual_american_put(
        N=N,
        K=K,
        r=r,
        q=q1,
        sigma=sigma1,
        S_max=L,
        V_right=put_tail,
    )

    left = K + S_int
    right = np.maximum(K - L + S_int, put_tail)
    bottom = put_1d.V[1:-1]
    top = np.maximum(K - S_int + L, 0.0)

    rhs_grid = np.zeros((n, n))
    rhs_grid[0, :] += -lower1[0] * left
    rhs_grid[-1, :] += -upper1[-1] * right
    rhs_grid[:, 0] += -lower2[0] * bottom
    rhs_grid[:, -1] += -upper2[-1] * top

    off_diagonal = np.concatenate(
        (iminus_flat[n:], jminus_flat[1:], jplus_flat[:-1], iplus_flat[:-n])
    )
    row_sum = np.asarray(A.sum(axis=1)).ravel()

    return Spread2DOperator(
        N=N,
        L=L,
        S=S,
        psi_int=psi_int,
        A=A,
        rhs=rhs_grid.ravel(),
        diag_flat=diag_flat,
        iminus_flat=iminus_flat,
        jminus_flat=jminus_flat,
        jplus_flat=jplus_flat,
        iplus_flat=iplus_flat,
        boundary={
            "left": left,
            "right": right,
            "bottom": bottom,
            "top": top,
        },
        min_diagonal=float(np.min(diag_flat)),
        max_off_diagonal=float(np.max(off_diagonal)),
        min_row_sum=float(np.min(row_sum)),
    )


def solve_spread_lcp(
    N: int,
    tol: float = 1e-12,
    max_iter: int = 300,
    **kwargs: float,
) -> Spread2DResult:
    """Solve the discrete LCP by sparse policy iteration."""
    start = time.perf_counter()
    op = build_spread_operator(N=N, **kwargs)

    scale = max(1.0, abs(op.max_off_diagonal), abs(op.min_diagonal))
    if op.min_diagonal <= 0.0 or op.max_off_diagonal > 1e-13 * scale:
        raise ArithmeticError("the 2D fitted operator is not an M-matrix")
    if op.min_row_sum <= 0.0:
        raise ArithmeticError("the 2D fitted operator lacks strict row dominance")

    V_int = op.psi_int.ravel().copy()
    previous_active: np.ndarray | None = None
    delta = math.inf

    for iterations in range(1, max_iter + 1):
        V_old = V_int.copy()
        residual = op.A @ V_int - op.rhs

        active = (V_int <= op.psi_int.ravel() + POLICY_TOL).copy()
        release = active & (residual < -POLICY_TOL)
        active[release] = False
        policy_stable = (
            previous_active is not None and np.array_equal(active, previous_active)
        )
        previous_active = active.copy()

        policy = op.policy_matrix(active)
        rhs = op.rhs.copy()
        rhs[active] = op.psi_int.ravel()[active]
        V_new = spsolve(policy, rhs)
        V_int = np.maximum(V_new, op.psi_int.ravel())

        delta = float(np.max(np.abs(V_int - V_old)))
        if delta < tol and policy_stable:
            break
    else:
        raise RuntimeError(
            f"2D policy iteration did not converge in {max_iter} iterations "
            f"(last change={delta:.3e})"
        )

    residual = op.A @ V_int - op.rhs
    slack = V_int - op.psi_int.ravel()
    complementarity = float(np.max(np.abs(np.minimum(residual, slack))))
    slack_tol = SLACK_TOL_FACTOR * max(1.0, K)

    V = np.empty((N + 1, N + 1), dtype=float)
    V[0, 1:-1] = op.boundary["left"]
    V[-1, 1:-1] = op.boundary["right"]
    V[1:-1, 0] = op.boundary["bottom"]
    V[1:-1, -1] = op.boundary["top"]
    V[0, 0] = K
    V[0, -1] = K + op.L
    V[-1, 0] = op.boundary["right"][0]
    V[-1, -1] = K
    V[1:-1, 1:-1] = V_int.reshape(N - 1, N - 1)

    elapsed = time.perf_counter() - start
    return Spread2DResult(
        N=N,
        S=op.S,
        V=V,
        iterations=iterations,
        delta=delta,
        complementarity=complementarity,
        min_slack=float(np.min(slack)),
        obstacle_ok=bool(np.all(slack >= -slack_tol)),
        contact_fraction=float(np.mean(slack <= slack_tol)),
        min_diagonal=op.min_diagonal,
        max_off_diagonal=op.max_off_diagonal,
        min_row_sum=op.min_row_sum,
        elapsed=elapsed,
    )


def _window_error(
    result: Spread2DResult, reference: Spread2DResult, window: float
) -> float:
    stride = reference.N // result.N
    coarse = result.S <= window
    difference = np.abs(
        result.V[np.ix_(coarse, coarse)]
        - reference.V[::stride, ::stride][np.ix_(coarse, coarse)]
    )
    return float(np.max(difference))


def _write_csv(results: dict[int, Spread2DResult], path: Path) -> None:
    reference = results[max(results)]
    fields = (
        "N",
        "unknowns",
        "iterations",
        "wall_time_s",
        "max_diff_vs_fine",
        "lcp_residual",
        "min_slack",
        "min_diagonal",
        "max_off_diagonal",
        "min_row_sum",
        "contact_fraction",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for N, result in sorted(results.items()):
            if N == reference.N:
                error = 0.0
            else:
                error = _window_error(result, reference, PRICING_WINDOW)
            writer.writerow(
                {
                    "N": N,
                    "unknowns": (N - 1) ** 2,
                    "iterations": result.iterations,
                    "wall_time_s": f"{result.elapsed:.6e}",
                    "max_diff_vs_fine": f"{error:.6e}",
                    "lcp_residual": f"{result.complementarity:.6e}",
                    "min_slack": f"{result.min_slack:.6e}",
                    "min_diagonal": f"{result.min_diagonal:.6e}",
                    "max_off_diagonal": f"{result.max_off_diagonal:.6e}",
                    "min_row_sum": f"{result.min_row_sum:.6e}",
                    "contact_fraction": f"{result.contact_fraction:.6e}",
                }
            )


def domain_truncation_study() -> list[dict[str, float | int | bool]]:
    """Check the high-price boundary closure at a fixed mesh.

    The value at (100,50), where the option is in the continuation region, is
    interpolated bilinearly on the same N=80 grid for three domain lengths.
    """
    rows: list[dict[str, float | int | bool]] = []
    for domain_length in DOMAIN_L_VALUES:
        result = solve_spread_lcp(DOMAIN_GRID, L=domain_length)
        interpolator = RegularGridInterpolator(
            (result.S, result.S), result.V, bounds_error=True
        )
        value = float(interpolator(np.asarray([DOMAIN_POINT]))[0])
        rows.append(
            {
                "L": domain_length,
                "N": DOMAIN_GRID,
                "V_100_50": value,
                "iterations": result.iterations,
                "lcp_residual": result.complementarity,
                "contact_fraction": result.contact_fraction,
                "obstacle_ok": result.obstacle_ok,
            }
        )
    longest_value = float(rows[-1]["V_100_50"])
    for row in rows:
        row["diff_vs_L400"] = float(row["V_100_50"]) - longest_value
    return rows


def _write_domain_csv(
    rows: list[dict[str, float | int | bool]], path: Path
) -> None:
    fields = (
        "L",
        "N",
        "V_100_50",
        "iterations",
        "lcp_residual",
        "contact_fraction",
        "diff_vs_L400",
        "obstacle_ok",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["V_100_50"] = f"{float(row['V_100_50']):.10e}"
            formatted["lcp_residual"] = f"{float(row['lcp_residual']):.6e}"
            formatted["contact_fraction"] = f"{float(row['contact_fraction']):.6e}"
            formatted["diff_vs_L400"] = f"{float(row['diff_vs_L400']):.6e}"
            writer.writerow(formatted)


def _plot_result(result: Spread2DResult, path: Path) -> None:
    S = result.S
    view = S <= 200.0
    S_view = S[view]
    X, Y = np.meshgrid(S_view, S_view, indexing="ij")
    Z = result.V[np.ix_(view, view)]
    payoff = np.maximum(K - X + Y, 0.0)
    contact = Z - payoff <= SLACK_TOL_FACTOR * max(1.0, K)

    fig = plt.figure(figsize=(13.2, 5.2))
    ax_surface = fig.add_subplot(1, 2, 1, projection="3d")
    ax_surface.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        rstride=2,
        cstride=2,
    )
    ax_surface.set_xlabel(r"$S_1$")
    ax_surface.set_ylabel(r"$S_2$")
    ax_surface.set_zlabel(r"$V_h(S_1,S_2)$")
    ax_surface.set_title("Fitted LCP value surface")

    ax_contact = fig.add_subplot(1, 2, 2)
    colors = ListedColormap(["#1b9e77", "#d95f02"])
    ax_contact.pcolormesh(
        X,
        Y,
        contact.astype(float),
        shading="auto",
        cmap=colors,
        vmin=0.0,
        vmax=1.0,
    )
    ax_contact.contour(X, Y, Z - payoff, levels=[0.0], colors="black", linewidths=0.8)
    ax_contact.set_xlabel(r"$S_1$")
    ax_contact.set_ylabel(r"$S_2$")
    ax_contact.set_title("Exercise/contact set (orange)")
    ax_contact.set_aspect("equal")

    fig.suptitle(
        rf"American spread put: five-point fitted LCP on a "
        rf"${result.N+1}\times{result.N+1}$ grid",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    plt.close(fig)


def main() -> dict[int, Spread2DResult]:
    print()
    print("2D exploratory extension -- American spread-put LCP")
    print(
        f"K={K:.1f}, r={R:.4f}, q1={Q1:.4f}, q2={Q2:.4f}, "
        f"sigma1={SIGMA1:.2f}, sigma2={SIGMA2:.2f}, rho={RHO:.2f}, L={L:.1f}"
    )
    print("Payoff psi=max(K-S1+S2,0); five-point exponential fitting")

    results = {N: solve_spread_lcp(N) for N in GRID_SIZES}
    reference = results[max(results)]

    header = (
        f"{'N':>5}  {'unknowns':>9}  {'iter':>5}  {'time/s':>9}  "
        f"{'E_[0,200]^2':>12}  {'LCP res':>11}  {'contact':>9}"
    )
    print(header)
    print("-" * len(header))
    for N, result in sorted(results.items()):
        if N == reference.N:
            error = 0.0
        else:
            error = _window_error(result, reference, PRICING_WINDOW)
        print(
            f"{N:5d}  {(N-1)**2:9d}  {result.iterations:5d}  "
            f"{result.elapsed:9.3f}  {error:12.3e}  "
            f"{result.complementarity:11.3e}  "
            f"{result.contact_fraction:9.3f}"
        )

    print()
    for N, result in sorted(results.items()):
        print(
            f"N={N}: min diag={result.min_diagonal:.3e}, "
            f"max off-diag={result.max_off_diagonal:.3e}, "
            f"min row sum={result.min_row_sum:.3e}, "
            f"obstacle={result.obstacle_ok}"
        )

    domain_rows = domain_truncation_study()
    print()
    print("Fixed-mesh domain-truncation check at (S1,S2)=(100,50), N=80")
    print(
        f"{'L':>6}  {'iter':>5}  {'V(100,50)':>16}  "
        f"{'diff vs L=400':>15}  {'LCP res':>11}"
    )
    print("-" * 61)
    for row in domain_rows:
        print(
            f"{float(row['L']):6.1f}  {int(row['iterations']):5d}  "
            f"{float(row['V_100_50']):16.10f}  "
            f"{float(row['diff_vs_L400']):15.6e}  "
            f"{float(row['lcp_residual']):11.3e}"
        )

    root = Path(__file__).resolve().parent
    _write_csv(results, root / "spread2d_results.csv")
    _write_domain_csv(domain_rows, root / "spread2d_domain.csv")
    _plot_result(reference, root / "spread2d_plot.png")
    print()
    print(
        "Saved spread2d_results.csv, spread2d_domain.csv, and "
        "spread2d_plot.png"
    )
    return results


if __name__ == "__main__":
    main()
