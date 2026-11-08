"""Verification checks for the Stage 1 constant-volatility LCP solution."""

from __future__ import annotations

import numpy as np

from stage1_validation import (
    build_fitted_operator,
    compute_lcp_diagnostics,
    solve_lcp,
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
    V, iterations, residual = solve_lcp(op, tol=TOL)
    diagnostics = compute_lcp_diagnostics(op, V)

    off_diagonal = np.concatenate((op.lower, op.upper))
    max_off_diagonal = float(np.max(off_diagonal))

    print()
    print("Stage 1 verification, N=400, fitted operator, V(S_max)=0")
    print(f"tol={TOL:.1e}, iterations={iterations}, final update={residual:.3e}")
    print()
    print("LCP residuals on interior nodes")
    print(f"min(a_h)                    = {diagnostics['min_a']:.17g}")
    print(f"min(b_h)                    = {diagnostics['min_b']:.17g}")
    print(
        "||min(a_h, b_h)||_inf       = "
        f"{diagnostics['complementarity']:.17g}"
    )
    print()

    print("M-matrix check")
    print(f"max off-diagonal            = {max_off_diagonal:.3e}")
    print(f"min diagonal                = {np.min(op.diag):.6e}")
    print()

    print("Solution-shape checks")
    print(f"monotonically decreasing    = {diagnostics['monotone']}")
    print(f"convex                      = {diagnostics['convex']}")
    print(
        "single contiguous contact set from S=0 = "
        f"{diagnostics['single_contact_set']}"
    )
    print()

    print("Value bounds")
    print(f"min(V)                      = {np.min(V):.17g}")
    print(f"max(V)                      = {np.max(V):.17g}")
    print(f"psi <= V <= K               = {diagnostics['value_bounds_ok']}")
    print()
    print(
        f"Free-boundary bracket       = "
        f"[{diagnostics['S_lo']:.6f}, {diagnostics['S_hi']:.6f}]"
    )
    print(f"S_h*                        = {diagnostics['S_free']:.6f}")
    print()


if __name__ == "__main__":
    main()
