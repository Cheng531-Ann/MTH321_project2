"""Compare policy iteration and PSOR on the Stage 1 perpetual-put LCP."""

from __future__ import annotations

import csv
import math
import time

import numpy as np

from stage1_validation import (
    build_fitted_operator,
    compute_lcp_diagnostics,
    solve_perpetual_american_put,
)


N = 400
K = 100.0
R = 0.05
Q = 0.0
SIGMA = 0.2
S_MAX = 400.0
TOL = 1e-10
OMEGA = 1.2
RESIDUAL_CHECK_INTERVAL = 100
TOLERANCES = (1e-4, 1e-6, 1e-8, 1e-10)


def lcp_residual_inf(
    x: np.ndarray,
    psi: np.ndarray,
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> float:
    a_h = diagonal * x - rhs
    if x.size > 1:
        a_h[1:] += lower[1:] * x[:-1]
        a_h[:-1] += upper[:-1] * x[1:]
    b_h = x - psi
    return float(np.max(np.abs(np.minimum(a_h, b_h))))


def lcp_scaled_residual(
    x: np.ndarray,
    psi: np.ndarray,
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Return a scale-free backward-style LCP residual.

    The raw complementarity residual is divided by the natural data scale
    ``||A||_inf ||x||_inf + ||rhs||_inf + ||psi||_inf``.  This makes values
    from different meshes and tolerances directly comparable.
    """
    raw = lcp_residual_inf(x, psi, lower, diagonal, upper, rhs)
    row_abs = np.abs(diagonal).copy()
    if x.size > 1:
        row_abs[1:] += np.abs(lower[1:])
        row_abs[:-1] += np.abs(upper[:-1])
    matrix_inf = float(np.max(row_abs))
    scale = (
        matrix_inf * float(np.max(np.abs(x)))
        + float(np.max(np.abs(rhs)))
        + float(np.max(np.abs(psi)))
    )
    return raw / max(1.0, scale)


def psor(
    psi_int: np.ndarray,
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    omega: float = OMEGA,
    tol: float = TOL,
    max_iter: int = 2_000_000,
) -> tuple[np.ndarray, int, float, float]:
    """Solve the LCP by PSOR using the complementarity residual as the test."""
    n = psi_int.size
    x = psi_int.copy()
    n_iter = 0
    max_change = math.inf
    lcp_residual = math.inf

    for n_iter in range(1, max_iter + 1):
        max_change = 0.0
        for j in range(n):
            residual = rhs[j]
            if j >= 1:
                residual -= lower[j] * x[j - 1]
            if j + 1 < n:
                residual -= upper[j] * x[j + 1]

            unconstrained = residual / diagonal[j]
            candidate = x[j] + omega * (unconstrained - x[j])
            if candidate < psi_int[j]:
                candidate = psi_int[j]

            change = abs(candidate - x[j])
            if change > max_change:
                max_change = change
            x[j] = candidate

        if n_iter % RESIDUAL_CHECK_INTERVAL == 0 or n_iter == max_iter:
            lcp_residual = lcp_residual_inf(
                x,
                psi_int,
                lower,
                diagonal,
                upper,
                rhs,
            )
            if lcp_residual < tol:
                break
    else:
        raise RuntimeError(
            f"PSOR did not converge in {max_iter} iterations "
            f"(last change={max_change:.3e}, "
            f"LCP residual={lcp_residual:.3e})"
        )

    return x, n_iter, max_change, lcp_residual


def benchmark(
    N: int = N,
    K: float = K,
    r: float = R,
    q: float = Q,
    sigma: float = SIGMA,
    S_max: float = S_MAX,
    tol: float = TOL,
    omega: float = OMEGA,
) -> dict[str, float | int]:
    """Run both solvers on the same fitted LCP and return timing data."""
    start = time.time()
    policy_result = solve_perpetual_american_put(
        N=N,
        K=K,
        r=r,
        q=q,
        sigma=sigma,
        S_max=S_max,
        tol=tol,
    )
    policy_time = time.time() - start

    op = build_fitted_operator(
        N=N,
        K=K,
        r=r,
        q=q,
        sigma=sigma,
        S_max=S_max,
        V_right=0.0,
    )
    psi_int = op.psi[1:-1]

    start = time.time()
    V_psor_int, psor_iterations, psor_change, psor_residual = psor(
        psi_int,
        op.lower,
        op.diag,
        op.upper,
        op.rhs,
        omega=omega,
        tol=tol,
    )
    psor_time = time.time() - start

    V_psor = np.concatenate(([op.V_left], V_psor_int, [op.V_right]))
    psor_diagnostics = compute_lcp_diagnostics(op, V_psor)
    solution_difference = float(np.max(np.abs(policy_result.V - V_psor)))

    return {
        "N": N,
        "policy_iterations": policy_result.iterations,
        "policy_time": policy_time,
        "policy_residual": policy_result.residual,
        "policy_lcp_residual": policy_result.complementarity,
        "psor_iterations": psor_iterations,
        "psor_time": psor_time,
        "psor_change": psor_change,
        "psor_residual": psor_residual,
        "psor_lcp_residual": float(psor_diagnostics["complementarity"]),
        "solution_difference": solution_difference,
    }


def tolerance_sweep(
    N: int = N,
    K: float = K,
    r: float = R,
    q: float = Q,
    sigma: float = SIGMA,
    S_max: float = S_MAX,
    omega: float = OMEGA,
) -> list[dict[str, float | int]]:
    """Vary the PSOR residual tolerance on one representative mesh.

    The direct policy-iteration solution is held fixed as the reference.  The
    sweep therefore isolates the algebraic error of an iterative solve from
    the spatial and truncation errors of the discrete problem.
    """
    policy_result = solve_perpetual_american_put(
        N=N,
        K=K,
        r=r,
        q=q,
        sigma=sigma,
        S_max=S_max,
    )
    op = build_fitted_operator(
        N=N,
        K=K,
        r=r,
        q=q,
        sigma=sigma,
        S_max=S_max,
        V_right=0.0,
    )
    psi_int = op.psi[1:-1]
    rows: list[dict[str, float | int]] = []

    for tol in TOLERANCES:
        start = time.time()
        x, iterations, _change, raw_residual = psor(
            psi_int,
            op.lower,
            op.diag,
            op.upper,
            op.rhs,
            omega=omega,
            tol=tol,
        )
        elapsed = time.time() - start
        V = np.concatenate(([op.V_left], x, [op.V_right]))
        diagnostics = compute_lcp_diagnostics(op, V)
        scaled = lcp_scaled_residual(
            x,
            psi_int,
            op.lower,
            op.diag,
            op.upper,
            op.rhs,
        )
        i100 = int(np.argmin(np.abs(op.S - 100.0)))
        rows.append(
            {
                "tolerance": tol,
                "iterations": iterations,
                "time": elapsed,
                "V100": float(V[i100]),
                "V100_reference": float(policy_result.V[i100]),
                "V100_difference": float(abs(V[i100] - policy_result.V[i100])),
                "max_difference": float(np.max(np.abs(V - policy_result.V))),
                "S_lo": float(diagnostics["S_lo"]),
                "S_hi": float(diagnostics["S_hi"]),
                "S_free": float(diagnostics["S_free"]),
                "raw_residual": raw_residual,
                "scaled_residual": scaled,
            }
        )

    return rows


def write_tolerance_sweep(
    rows: list[dict[str, float | int]],
    path: str = "tolerance_sweep.csv",
) -> None:
    """Write the tolerance sweep in a form that can be archived and audited."""
    fields = (
        "tolerance",
        "iterations",
        "time",
        "V100",
        "V100_reference",
        "V100_difference",
        "max_difference",
        "S_lo",
        "S_hi",
        "S_free",
        "raw_residual",
        "scaled_residual",
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> dict[str, float | int]:
    data = benchmark()
    sweep = tolerance_sweep()
    write_tolerance_sweep(sweep)

    print()
    print("Solver comparison: Stage 1 fitted LCP, V(S_max)=0")
    print(
        f"N={N}, K={K:.1f}, r={R:.4f}, q={Q:.4f}, sigma={SIGMA:.2f}, "
        f"S_max={S_MAX:.1f}, tol={TOL:.1e}"
    )
    print("Initial iterate V=psi for both solvers; PSOR relaxation omega=1.2")
    print()

    header = (
        f"{'Solver':<20}  {'Iterations':>12}  {'CPU time (s)':>14}  "
        f"{'Final change':>14}  {'LCP residual':>14}"
    )
    rule = "-" * len(header)
    print(header)
    print(rule)
    print(
        f"{'Policy iteration':<20}  {data['policy_iterations']:12d}  "
        f"{data['policy_time']:14.6f}  "
        f"{data['policy_residual']:14.3e}  "
        f"{data['policy_lcp_residual']:14.3e}"
    )
    print(
        f"{'PSOR (omega=1.2)':<20}  {data['psor_iterations']:12d}  "
        f"{data['psor_time']:14.6f}  {data['psor_change']:14.3e}  "
        f"{data['psor_lcp_residual']:14.3e}"
    )
    print(rule)
    print(f"Max |V_policy - V_PSOR| = {data['solution_difference']:.6e}")
    print()

    print("PSOR tolerance sweep at the same N=400 mesh")
    sweep_header = (
        f"{'Tolerance':>10}  {'Iterations':>12}  {'V(100)':>14}  "
        f"{'S_h*':>9}  {'Raw residual':>14}  {'Scaled residual':>16}  "
        f"{'Max |dV|':>12}"
    )
    print(sweep_header)
    print("-" * len(sweep_header))
    for row in sweep:
        print(
            f"{row['tolerance']:10.1e}  {row['iterations']:12d}  "
            f"{row['V100']:14.10f}  {row['S_free']:9.4f}  "
            f"{row['raw_residual']:14.3e}  {row['scaled_residual']:16.3e}  "
            f"{row['max_difference']:12.3e}"
        )
    print("Wrote tolerance_sweep.csv")
    print()
    return data


if __name__ == "__main__":
    main()
