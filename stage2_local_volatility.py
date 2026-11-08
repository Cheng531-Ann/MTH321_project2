"""Stage 2: perpetual American put with local volatility.

The production solver uses the same exponential-fitting
(Scharfetter--Gummel) operator as Stage 1, with ``sigma = sigma(S)`` and the
truncation boundary condition ``V(S_max) = 0``.  The independent validation
reference is now a semi-analytic ODE shooting solution, not a finer finite
difference grid.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from local_vol_model import K, Q, R, S_MAX, local_volatility
from reference_solution import build_reference
from stage1_validation import (
    LCPResult,
    build_fitted_operator,
    compute_lcp_diagnostics,
    solve_lcp,
    solve_perpetual_american_put,
)


GRID_SIZES = (600, 1200, 2400, 4800)


def solve_local_volatility_put(
    N: int,
    K: float = K,
    r: float = R,
    q: float = Q,
    S_max: float = S_MAX,
    V_right: float = 0.0,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> LCPResult:
    """Solve the local-volatility LCP with the fitted operator.

    The right boundary uses ``V(S_max) = 0``.  The finite-domain shooting
    reference in ``reference_solution.py`` uses the same boundary condition.
    """
    op = build_fitted_operator(
        N=N,
        K=K,
        r=r,
        q=q,
        sigma=lambda S: local_volatility(S, K_value=K),
        S_max=S_max,
        V_right=V_right,
    )
    V, iterations, residual = solve_lcp(op, tol=tol, max_iter=max_iter)
    diagnostics = compute_lcp_diagnostics(op, V)

    return LCPResult(
        N=N,
        S=op.S,
        V=V,
        S_free=float(diagnostics["S_free"]),
        max_error=float("nan"),
        max_error_interior=float("nan"),
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


def main() -> dict[int, LCPResult]:
    """Report production errors against an independent shooting reference."""
    finite_reference = build_reference(domain="finite", s_max=S_MAX)
    true_reference = build_reference(domain="infinite", s_max=S_MAX)

    print()
    print("Stage 2 -- Perpetual American Put with Local Volatility")
    print(
        "sigma(S) = 0.20 * (1 + 0.35 * tanh(log(S/K) / 0.40)), "
        f"K={K:.1f}, r={R:.4f}, q={Q:.4f}, S_max={S_MAX:.1f}"
    )
    print("Reference: independent DOP853 shooting solution of the free-boundary ODE")
    print(
        f"  infinite-domain: S*={true_reference.S_star:.8f}, "
        f"V(100)={true_reference.value_at(100.0):.8f}, "
        f"V(2400)={true_reference.value_at(S_MAX):.8f}"
    )
    print(
        f"  finite-domain:   S*={finite_reference.S_star:.8f}, "
        f"V(100)={finite_reference.value_at(100.0):.8f}, "
        f"V(2400)={finite_reference.value_at(S_MAX):.3e}"
    )
    print(
        "  truncation at S=100: "
        f"{abs(true_reference.value_at(100.0) - finite_reference.value_at(100.0)):.3e}; "
        f"at S=2400: {abs(true_reference.value_at(S_MAX) - finite_reference.value_at(S_MAX)):.3e}"
    )
    print()

    results: dict[int, LCPResult] = {}
    header = (
        f"{'N':>6}  {'S_lo':>9}  {'S_hi':>9}  {'S_h*':>9}  "
        f"{'Err finite':>12}  {'Err true':>12}  {'Iter':>5}"
    )
    rule = "-" * len(header)
    print(header)
    print(rule)
    for N in GRID_SIZES:
        result = solve_local_volatility_put(N)
        results[N] = result
        finite_error = float(np.max(np.abs(result.V - finite_reference.evaluate(result.S))))
        true_error = float(np.max(np.abs(result.V - true_reference.evaluate(result.S))))
        print(
            f"{N:6d}  {result.S_lo:9.3f}  {result.S_hi:9.3f}  "
            f"{result.S_free:9.3f}  {finite_error:12.4e}  "
            f"{true_error:12.4e}  {result.iterations:5d}"
        )
    print(rule)
    print()

    print("LCP residual and solution-shape checks")
    header = (
        f"{'N':>6}  {'min a_h':>12}  {'min b_h':>12}  "
        f"{'||min||inf':>12}  {'psi<=V<=K':>10}  {'monotone':>9}  "
        f"{'convex':>7}  {'1 contact':>10}"
    )
    rule = "-" * len(header)
    print(header)
    print(rule)
    for N in GRID_SIZES:
        result = results[N]
        print(
            f"{N:6d}  {result.min_a:12.3e}  {result.min_b:12.3e}  "
            f"{result.complementarity:12.3e}  "
            f"{str(result.value_bounds_ok):>10}  "
            f"{str(result.monotone):>9}  {str(result.convex):>7}  "
            f"{str(result.single_contact_set):>10}"
        )
    print(rule)
    print()

    stage1 = solve_perpetual_american_put(
        N=2400,
        K=K,
        r=R,
        q=Q,
        S_max=S_MAX,
    )
    local = results[2400]

    fig, (ax_values, ax_diff) = plt.subplots(1, 2, figsize=(13, 5))

    ax_values.plot(
        stage1.S,
        stage1.V,
        color="black",
        linestyle="--",
        label=r"Stage 1: constant $\sigma=0.20$",
    )
    ax_values.plot(
        local.S,
        local.V,
        color="blue",
        linestyle="-",
        label="Stage 2: local sigma(S)",
    )
    zoom = np.linspace(50.0, 120.0, 600)
    ax_values.plot(
        zoom,
        true_reference.evaluate(zoom),
        color="tab:green",
        linestyle=":",
        linewidth=2.0,
        label="ODE shooting reference",
    )
    ax_values.axvspan(
        local.S_lo,
        local.S_hi,
        color="red",
        alpha=0.12,
        label=rf"Bracket $[{local.S_lo:.2f}, {local.S_hi:.2f}]$",
    )
    ax_values.set_xlim(50.0, 120.0)
    ax_values.set_xlabel("Stock price S")
    ax_values.set_ylabel("Option value V(S)")
    ax_values.set_title("Values (zoom)")
    ax_values.grid(True, linestyle=":", alpha=0.5)
    ax_values.legend()

    difference = local.V - stage1.V
    ax_diff.plot(
        local.S,
        difference,
        color="purple",
        linewidth=1.8,
        label=r"$V_{\mathrm{Stage\,2}}-V_{\mathrm{Stage\,1}}$",
    )
    ax_diff.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    ax_diff.axvspan(
        local.S_lo,
        local.S_hi,
        color="red",
        alpha=0.12,
        label=rf"Bracket $[{local.S_lo:.2f}, {local.S_hi:.2f}]$",
    )
    ax_diff.set_xlim(50.0, 120.0)
    ax_diff.set_xlabel("Stock price S")
    ax_diff.set_ylabel(r"$V_{\mathrm{Stage\,2}}-V_{\mathrm{Stage\,1}}$")
    ax_diff.set_title("Impact of local volatility")
    ax_diff.grid(True, linestyle=":", alpha=0.5)
    ax_diff.legend()

    fig.suptitle(
        "Perpetual American Put: Constant vs Local Volatility (N=2400)",
        fontsize=13,
    )
    fig.tight_layout()
    plt.savefig("stage2_plot.png", dpi=150)
    plt.show()

    return results


if __name__ == "__main__":
    stage2_results = main()
