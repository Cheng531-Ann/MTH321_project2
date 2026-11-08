"""Domain-truncation study for the Stage 2 local-volatility LCP.

The mesh spacing is held fixed while ``S_max`` is varied, so the table
isolates truncation error from mesh-refinement error.
"""

from __future__ import annotations

import numpy as np

from stage2_local_volatility import solve_local_volatility_put


H = 1.0
S_MAX_VALUES = (300.0, 400.0, 600.0, 1200.0, 2400.0)
COMPARISON_POINTS = (50.0, 100.0)


def value_at(S: np.ndarray, V: np.ndarray, target: float) -> float:
    """Linearly interpolate the numerical solution at a target stock price."""
    return float(np.interp(target, S, V))


def sweep() -> list[dict[str, float]]:
    """Run the fixed-h domain sweep and return one row per S_max."""
    rows: list[dict[str, float]] = []
    for S_max in S_MAX_VALUES:
        N = int(round(S_max / H))
        result = solve_local_volatility_put(N=N, S_max=S_max)
        rows.append(
            {
                "S_max": S_max,
                "N": float(N),
                "h": H,
                "S_lo": result.S_lo,
                "S_hi": result.S_hi,
                "S_free": result.S_free,
                "V_50": value_at(result.S, result.V, COMPARISON_POINTS[0]),
                "V_100": value_at(result.S, result.V, COMPARISON_POINTS[1]),
            }
        )
    return rows


def main() -> list[dict[str, float]]:
    rows = sweep()

    print()
    print("Stage 2 local volatility -- domain truncation study at fixed h")
    print(f"h={H:g}, sigma(S)=0.20*(1+0.35*tanh(log(S/K)/0.40)), q=0.02")
    print()
    header = (
        f"{'S_max':>8}  {'N':>7}  {'S_lo':>9}  {'S_hi':>9}  "
        f"{'S_h*':>9}  {'V(50)':>14}  {'V(100)':>14}"
    )
    rule = "-" * len(header)
    print(header)
    print(rule)

    for row in rows:
        print(
            f"{row['S_max']:8.0f}  {row['N']:7.0f}  "
            f"{row['S_lo']:9.3f}  {row['S_hi']:9.3f}  "
            f"{row['S_free']:9.3f}  {row['V_50']:14.8f}  "
            f"{row['V_100']:14.8f}"
        )

    print(rule)
    print()
    print(
        "V(50) and V(100) are linearly interpolated when the target price "
        "is not a grid node.  The mesh spacing is identical in every row."
    )
    print()
    return rows


if __name__ == "__main__":
    main()
