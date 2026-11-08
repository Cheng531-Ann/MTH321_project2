'''Synthetic zero-flux boundary-row experiment for the Stage 1 put.

The fitted operator and policy-iteration structure are inherited from the
Stage 1 solver. Only the left boundary changes from V(0)=K to dV/dS(0)=0.
A ghost node at -h enforces zero flux as V_{-1}=V_1, hence V_0=V_1 after
boundary interpolation.

This is an algebraic stress test, not a financially admissible left boundary
for the American put.  At S=0 the obstacle requires V(0)>=K while the payoff
slope is -1, so a zero-flux condition is incompatible with the exercise region.
The script therefore prints the obstacle gap at S=0; it does not use the result
for pricing or convergence claims.
'''

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from stage1_validation import (
    SLACK_TOL_FACTOR,
    GridOperator,
    build_fitted_operator,
    extract_free_boundary,
    lcp_residuals,
    solve_lcp as solve_dirichlet_lcp,
)


N = 400
K = 100.0
S_MAX = 400.0
TOL = 1e-10
FittedOperator = GridOperator


def solve_dirichlet(
    op: FittedOperator,
) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray, float]:
    '''Solve the Dirichlet LCP and return residuals and boundary data.'''
    V, iterations, residual = solve_dirichlet_lcp(op, tol=TOL)
    a_h, b_h = lcp_residuals(op, V)
    return V, iterations, residual, a_h, b_h, float(V[0])


def free_boundary(
    op: FittedOperator,
    b_h: np.ndarray,
    tol_slack: float,
) -> tuple[float, float, float]:
    '''Wrap the Stage 1 free-boundary interpolation.'''
    return extract_free_boundary(op.S[1:-1], b_h, tol_slack)


def m_matrix_check(op: FittedOperator, left_bc: str) -> tuple[float, float]:
    '''Return maximum off-diagonal and minimum diagonal after closure.'''
    diagonal = op.diag.copy()
    if left_bc == 'neumann':
        diagonal[0] += op.lower[0]
    off_diagonal = np.concatenate((op.lower[1:], op.upper[:-1]))
    return float(np.max(off_diagonal)), float(np.min(diagonal))


def solve_zero_flux(
    op: FittedOperator,
    tol: float = TOL,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray, float]:
    '''Solve the LCP with the ghost-point zero-flux closure V_0=V_1.'''
    n_int = op.N - 1
    psi_int = op.psi[1:-1]
    diag = op.diag.copy()
    rhs = op.rhs.copy()

    # Substitute V_0=V_1 in l_1 V_0 + d_1 V_1 + u_1 V_2 = F_1.
    diag[0] += op.lower[0]
    lower = op.lower[1:].copy()
    upper = op.upper[:-1].copy()
    A_full = sp.diags(
        (lower, diag, upper),
        offsets=(-1, 0, 1),
        shape=(n_int, n_int),
        format='csr',
    )

    V_int = psi_int.copy()
    previous_active: np.ndarray | None = None
    residual = math.inf

    for iterations in range(1, max_iter + 1):
        old = V_int.copy()
        lcp_residual = A_full @ V_int - rhs
        active = V_int <= psi_int + TOL
        active[active & (lcp_residual < -TOL)] = False
        policy_stable = (
            previous_active is not None
            and np.array_equal(active, previous_active)
        )
        previous_active = active.copy()

        solve_diag = diag.copy()
        solve_diag[active] = 1.0
        solve_lower = lower.copy()
        solve_upper = upper.copy()
        if n_int > 1:
            solve_lower[active[1:]] = 0.0
            solve_upper[active[:-1]] = 0.0

        system = sp.diags(
            (solve_lower, solve_diag, solve_upper),
            offsets=(-1, 0, 1),
            shape=(n_int, n_int),
            format='csr',
        )
        solve_rhs = rhs.copy()
        solve_rhs[active] = psi_int[active]
        V_int = np.maximum(spsolve(system, solve_rhs), psi_int)

        residual = float(np.max(np.abs(V_int - old)))
        if residual < tol and policy_stable:
            break
    else:
        raise RuntimeError('zero-flux policy iteration did not converge')

    V0 = float(V_int[0])
    V = np.concatenate(([V0], V_int, [0.0]))

    a_h = diag * V_int - rhs
    if n_int > 1:
        a_h[1:] += lower * V_int[:-1]
        a_h[:-1] += upper * V_int[1:]
    b_h = V_int - psi_int
    return V, iterations, residual, a_h, b_h, V0


def main() -> None:
    op = build_fitted_operator(
        N=N,
        K=K,
        r=0.05,
        q=0.0,
        sigma=0.20,
        S_max=S_MAX,
        V_right=0.0,
    )
    V_dir, iter_dir, res_dir, a_dir, b_dir, _ = solve_dirichlet(op)
    V_zero, iter_zero, res_zero, a_zero, b_zero, V0_zero = solve_zero_flux(op)

    tol_slack = SLACK_TOL_FACTOR * K
    lo_dir, hi_dir, free_dir = free_boundary(op, b_dir, tol_slack)
    lo_zero, hi_zero, free_zero = free_boundary(op, b_zero, tol_slack)
    max_off_dir, min_diag_dir = m_matrix_check(op, 'dirichlet')
    max_off_zero, min_diag_zero = m_matrix_check(op, 'neumann')
    max_difference = float(np.max(np.abs(V_zero - V_dir)))
    ghost_value = float(V_zero[1])
    ghost_derivative = float(
        (V_zero[1] - ghost_value) / (2.0 * op.h)
    )
    boundary_obstacle_gap = float(V_zero[0] - K)

    print()
    print('Perpetual American put: zero-flux Neumann versus Dirichlet')
    print(
        f'N={N}, S_max={S_MAX:.0f}, h={op.h:.6f}, '
        f'tol={TOL:.1e}'
    )
    print()
    print(
        f'Dirichlet: S_h*={free_dir:.9f}, bracket='
        f'[{lo_dir:.6f}, {hi_dir:.6f}], iterations={iter_dir}'
    )
    print(
        f'Zero flux: S_h*={free_zero:.9f}, bracket='
        f'[{lo_zero:.6f}, {hi_zero:.6f}], iterations={iter_zero}'
    )
    print(f'Difference in free boundary: {abs(free_zero - free_dir):.6e}')
    print(f'Zero-flux boundary value V(0)={V0_zero:.12f}')
    print(f'Ghost value V(-h)={ghost_value:.12f}')
    print(
        f'Ghost closure dV/dS(0)={ghost_derivative:.12f} '
        '(enforced read-back, not an independent check)'
    )
    print(
        f'Boundary obstacle gap V(0)-K={boundary_obstacle_gap:.12f} '
        '(negative means the synthetic closure is not an admissible price)'
    )
    print(f'Maximum pointwise difference: {max_difference:.12f}')
    print(
        'M-matrix Dirichlet: max off-diagonal='
        f'{max_off_dir:.3e}, min diagonal={min_diag_dir:.6e}'
    )
    print(
        'M-matrix zero flux: max off-diagonal='
        f'{max_off_zero:.3e}, min diagonal={min_diag_zero:.6e}'
    )
    print(
        'LCP residuals: '
        f'Dirichlet min(a_h,b_h)={min(a_dir.min(), b_dir.min()):.3e}, '
        f'zero flux min(a_h,b_h)={min(a_zero.min(), b_zero.min()):.3e}'
    )
    print(
        f'Final policy updates: Dirichlet={res_dir:.3e}, '
        f'zero flux={res_zero:.3e}'
    )
    print()

    fig, (ax_origin, ax_boundary) = plt.subplots(
        1, 2, figsize=(11, 4.8)
    )
    payoff = np.maximum(K - op.S, 0.0)
    for ax in (ax_origin, ax_boundary):
        ax.plot(op.S, payoff, color='black', linestyle='--', label='Payoff')
        ax.plot(
            op.S,
            V_dir,
            color='tab:blue',
            linewidth=2.0,
            label=rf'Dirichlet $S_h^*={free_dir:.3f}$',
        )
        ax.plot(
            op.S,
            V_zero,
            color='tab:orange',
            linewidth=1.4,
            linestyle=':',
            label=rf'Zero flux $S_h^*={free_zero:.3f}$',
        )
        ax.set_xlabel('Stock price S')
        ax.set_ylabel('Option value V(S)')

    ax_origin.set_xlim(0.0, 5.0)
    ax_origin.set_ylim(94.0, 100.5)
    ax_origin.set_title('Near the reflecting origin')
    ax_origin.legend(loc='lower left')

    ax_boundary.axvline(free_dir, color='tab:blue', linestyle='-', alpha=0.4)
    ax_boundary.axvline(
        free_zero, color='tab:orange', linestyle=':', alpha=0.9
    )
    ax_boundary.set_xlim(50.0, 100.0)
    ax_boundary.set_ylim(0.0, 52.0)
    ax_boundary.set_title('Free-boundary region')
    ax_boundary.legend(loc='upper right')

    fig.suptitle('Zero-Flux Neumann versus Dirichlet Left Boundary (N=400)')
    fig.tight_layout()
    fig.savefig('neumann_true_comparison.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
