"""Local-volatility convergence against an independent ODE reference.

The reference is obtained by shooting the free-boundary ODE, so it is not a
finer instance of the finite-difference solver.  Two comparisons are made:

* the production uniform grids are compared with the exact solution of the
  same finite-domain problem (``V(S_max)=0``);
* a boundary-aligned validation grid is used to measure the spatial order
  without the free-boundary phase effect.

The difference between the finite-domain and infinite-domain references is
reported separately as the truncation error.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from advanced_methods import build_nonuniform_operator, solve_operator
from grids import boundary_aligned_grid
from local_vol_model import K, Q, R, S_MAX, local_volatility
from reference_solution import build_reference
from stage1_validation import observed_order
from stage2_local_volatility import GRID_SIZES, solve_local_volatility_put


OUTPUT_FILE = Path(__file__).resolve().with_name("stage2_convergence.csv")


def main() -> None:
    finite_reference = build_reference(domain="finite", s_max=S_MAX)
    true_reference = build_reference(domain="infinite", s_max=S_MAX)

    print("Stage 2 -- local volatility against an independent ODE shooting reference")
    print(
        f"infinite-domain reference: S*={true_reference.S_star:.10f}, "
        f"V(100)={true_reference.value_at(100.0):.10f}, "
        f"V(2400)={true_reference.value_at(S_MAX):.6e}"
    )
    print(
        f"finite-domain reference:   S*={finite_reference.S_star:.10f}, "
        f"V(100)={finite_reference.value_at(100.0):.10f}, "
        f"V(2400)={finite_reference.value_at(S_MAX):.6e}"
    )
    print(
        "truncation error at S=100: "
        f"{abs(true_reference.value_at(100.0) - finite_reference.value_at(100.0)):.6e}"
    )
    print(
        "truncation error at S=2400: "
        f"{abs(true_reference.value_at(S_MAX) - finite_reference.value_at(S_MAX)):.6e}"
    )
    print()

    rows: list[dict[str, float | int]] = []
    previous_aligned_error: float | None = None
    header = (
        f"{'N':>6}  {'h':>5}  {'uniform E':>11}  {'aligned E':>11}  "
        f"{'ratio':>7}  {'p':>6}  {'uniform bracket':>18}  "
        f"{'uniform S_h*':>13}  {'aligned S_h*':>13}"
    )
    print(header)
    print("-" * len(header))

    for N in GRID_SIZES:
        production = solve_local_volatility_put(N)
        uniform_error = float(
            np.max(np.abs(production.V - finite_reference.evaluate(production.S)))
        )

        aligned_grid = boundary_aligned_grid(N, finite_reference.S_star, S_MAX)
        op = build_nonuniform_operator(
            aligned_grid,
            K_value=K,
            r=R,
            q=Q,
            sigma=local_volatility,
            boundary="dirichlet",
        )
        V, diagnostics, iterations, _ = solve_operator(op)
        aligned_error = float(
            np.max(np.abs(V - finite_reference.evaluate(aligned_grid)))
        )
        ratio = (
            math.nan
            if previous_aligned_error is None
            else previous_aligned_error / aligned_error
        )
        order = (
            math.nan
            if previous_aligned_error is None
            else observed_order(previous_aligned_error, aligned_error)
        )
        rows.append(
            {
                "N": N,
                "h": S_MAX / N,
                "uniform_error_finite_reference": uniform_error,
                "aligned_error_finite_reference": aligned_error,
                "aligned_error_ratio": ratio,
                "aligned_observed_order": order,
                "uniform_S_lo": production.S_lo,
                "uniform_S_hi": production.S_hi,
                "uniform_S_free": production.S_free,
                "aligned_S_free": float(diagnostics["S_free"]),
                "iterations": int(iterations),
                "true_S_star": true_reference.S_star,
                "finite_S_star": finite_reference.S_star,
                "truncation_error_V100": abs(
                    true_reference.value_at(100.0) - finite_reference.value_at(100.0)
                ),
                "truncation_error_V2400": abs(
                    true_reference.value_at(S_MAX) - finite_reference.value_at(S_MAX)
                ),
            }
        )
        print(
            f"{N:6d}  {S_MAX / N:5.1f}  {uniform_error:11.4e}  "
            f"{aligned_error:11.4e}  {ratio:7.3f}  {order:6.3f}  "
            f"[{production.S_lo:7.3f},{production.S_hi:7.3f}]  "
            f"{production.S_free:13.3f}  {float(diagnostics['S_free']):13.3f}"
        )
        if not (production.S_lo <= finite_reference.S_star <= production.S_hi):
            print(
                f"  note: N={N} contact bracket does not contain the "
                f"finite-domain reference S*={finite_reference.S_star:.10f}; "
                "this is disclosed as coarse-grid boundary localisation."
            )
        previous_aligned_error = aligned_error
    print("-" * len(header))

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
