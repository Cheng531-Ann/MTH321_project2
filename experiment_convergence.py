"""Convergence and production-boundary study for the constant-volatility benchmark.

A node is placed at the known analytic free boundary.  This removes the
free-boundary phase effect from the maximum norm and produces a clean
second-order refinement sequence.  The asymptotic Robin closure is used so
that the table measures spatial discretisation, not the O(1) value at the
artificial zero-Dirichlet boundary.

The same run also records the production uniform-grid contact bracket.  Those
values are diagnostics, not order evidence: a coarse active set can place the
contact transition on the wrong side of the analytic boundary.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from advanced_methods import build_nonuniform_operator, solve_operator
from grids import boundary_aligned_grid
from stage1_validation import (
    exact_free_boundary,
    exact_perpetual_put,
    observed_order,
    solve_perpetual_american_put,
)


GRID_SIZES = (600, 1200, 2400, 4800)
S_MAX = 2400.0
OUTPUT_FILE = Path(__file__).resolve().with_name("stage1_convergence.csv")


def aligned_study() -> list[dict[str, float]]:
    """Return the constant-volatility convergence table."""
    center = exact_free_boundary()
    rows: list[dict[str, float]] = []
    previous_error: float | None = None

    for N in GRID_SIZES:
        S = boundary_aligned_grid(N, center, S_MAX)
        op = build_nonuniform_operator(S, boundary="robin")
        V, diagnostics, iterations, _ = solve_operator(op)
        exact_values = exact_perpetual_put(S)
        error = float(np.max(np.abs(V - exact_values)))
        production = solve_perpetual_american_put(
            N=N,
            q=0.0,
            sigma=0.2,
            S_max=S_MAX,
            V_right=0.0,
        )
        ratio = (
            math.nan
            if previous_error is None
            else previous_error / error
        )
        order = (
            math.nan
            if previous_error is None
            else observed_order(previous_error, error)
        )
        rows.append(
            {
                "N": float(N),
                "h": S_MAX / N,
                "max_error": error,
                "error_ratio": ratio,
                "observed_order": order,
                "S_lo": float(diagnostics["S_lo"]),
                "S_hi": float(diagnostics["S_hi"]),
                "S_free": float(diagnostics["S_free"]),
                "boundary_error": abs(float(diagnostics["S_free"]) - center),
                "complementarity": float(diagnostics["complementarity"]),
                "iterations": float(iterations),
                "production_error": production.max_error,
                "production_S_lo": production.S_lo,
                "production_S_hi": production.S_hi,
                "production_S_free": production.S_free,
                "production_boundary_error": abs(production.S_free - center),
                "production_complementarity": production.complementarity,
            }
        )
        previous_error = error
    return rows


def main() -> None:
    rows = aligned_study()
    print("Aligned-grid convergence for the constant-volatility benchmark")
    print(
        f"grid=boundary-aligned piecewise uniform, centre S*={exact_free_boundary():.10f}, "
        "Robin closure"
    )
    print(
        "Production columns use the uniform V(S_max)=0 grid; they are "
        "diagnostics and are not used to estimate the order."
    )
    header = (
        f"{'N':>6}  {'h':>6}  {'max error':>12}  {'ratio':>8}  "
        f"{'p':>7}  {'bracket':>18}  {'S_h*':>10}  {'|dS*|':>10}  "
        f"{'LCP res.':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ratio = row["error_ratio"]
        order = row["observed_order"]
        ratio_text = "--" if math.isnan(ratio) else f"{ratio:8.3f}"
        order_text = "--" if math.isnan(order) else f"{order:7.3f}"
        print(
            f"{row['N']:6.0f}  {row['h']:6.1f}  {row['max_error']:12.4e}  "
            f"{ratio_text:>8}  {order_text:>7}  "
            f"[{row['S_lo']:7.3f},{row['S_hi']:7.3f}]  "
            f"{row['S_free']:10.3f}  {row['boundary_error']:10.3e}  "
            f"{row['complementarity']:11.3e}"
        )

    print()
    print("Production uniform-grid diagnostics")
    header = (
        f"{'N':>6}  {'bracket':>18}  {'S_h*':>10}  {'|dS*|':>10}  "
        f"{'max error':>12}  {'contains S*':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        contains = (
            row["production_S_lo"]
            <= exact_free_boundary()
            <= row["production_S_hi"]
        )
        print(
            f"{row['N']:6.0f}  "
            f"[{row['production_S_lo']:7.3f},{row['production_S_hi']:7.3f}]  "
            f"{row['production_S_free']:10.3f}  "
            f"{row['production_boundary_error']:10.3e}  "
            f"{row['production_error']:12.4e}  {str(contains):>11}"
        )

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
