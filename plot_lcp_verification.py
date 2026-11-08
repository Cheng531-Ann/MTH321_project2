"""Plot obstacle slackness and the LCP residual for Stage 1, N=400."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from stage1_validation import (
    build_fitted_operator,
    compute_lcp_diagnostics,
    lcp_residuals,
    solve_perpetual_american_put,
)


N = 400
K = 100.0
R = 0.05
Q = 0.0
SIGMA = 0.2
S_MAX = 400.0
TOL = 1e-10


def main() -> None:
    op = build_fitted_operator(
        N=N,
        K=K,
        r=R,
        q=Q,
        sigma=SIGMA,
        S_max=S_MAX,
        V_right=0.0,
    )
    result = solve_perpetual_american_put(
        N=N,
        K=K,
        r=R,
        q=Q,
        sigma=SIGMA,
        S_max=S_MAX,
        tol=TOL,
    )
    diagnostics = compute_lcp_diagnostics(op, result.V)
    a_h, b_h = lcp_residuals(op, result.V)
    S_int = op.S[1:-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(S_int, b_h, color="blue", linewidth=1.6)
    axes[0].axvline(
        result.S_free,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=rf"$S_h^*={result.S_free:.3f}$",
    )
    axes[0].axvspan(
        result.S_lo,
        result.S_hi,
        color="red",
        alpha=0.12,
        label=rf"bracket $[{result.S_lo:.3f}, {result.S_hi:.3f}]$",
    )
    axes[0].set_xlim(0.0, 160.0)
    axes[0].set_xlabel("Stock price S")
    axes[0].set_ylabel(r"$b_h=V_h-\psi_h$")
    axes[0].set_title("Obstacle Slackness")
    axes[0].legend()

    axes[1].plot(S_int, a_h, color="darkgreen", linewidth=1.6)
    axes[1].axvline(
        result.S_free,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=rf"$S_h^*={result.S_free:.3f}$",
    )
    axes[1].set_xlim(0.0, 160.0)
    axes[1].set_xlabel("Stock price S")
    axes[1].set_ylabel(r"$a_h=-L_hV_h$")
    axes[1].set_title("LCP Residual")
    axes[1].legend()

    fig.suptitle("Stage 1 LCP Verification (N=400, fitted scheme)", fontsize=13)
    fig.tight_layout()
    print("Saved lcp_verification.png")
    print(
        f"Free boundary S_h*={result.S_free:.3f} in "
        f"[{result.S_lo:.3f}, {result.S_hi:.3f}]; "
        f"min(b_h)={diagnostics['min_b']:.6e}, "
        f"min(a_h)={diagnostics['min_a']:.6e}"
    )
    plt.savefig("lcp_verification.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
