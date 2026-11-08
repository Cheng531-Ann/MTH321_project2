'''Advanced extensions for the perpetual American put LCP.

The module implements nonuniform exponential fitting, an asymptotic Robin
far-field closure, multi-point free-boundary extraction, and discrete
adjoint sensitivities for value sensitivities.
'''

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from stage1_validation import (
    compute_lcp_diagnostics,
    exact_free_boundary,
    exact_perpetual_put,
    extract_free_boundary,
    solve_lcp,
)
from grids import boundary_aligned_grid, boundary_clustered_grid


K = 100.0
R = 0.05
Q = 0.0
SIGMA = 0.20
S_MAX = 2400.0
TOL_SLACK = 1e-10 * K


@dataclass
class NonuniformOperator:
    '''Fitted operator A = -L_h on a possibly nonuniform grid.'''

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
    boundary: str
    tail_ratio: float


def bernoulli(z: np.ndarray) -> np.ndarray:
    '''Evaluate B(z) = z / (exp(z) - 1) without cancellation.'''
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    small = np.abs(z) < 1e-8
    out[small] = 1.0 - 0.5 * z[small] + z[small] ** 2 / 12.0
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        out[~small] = z[~small] / np.expm1(z[~small])
    return out


def asymptotic_lambda(r: float = R, q: float = Q, sigma: float = SIGMA) -> float:
    '''Return the decaying root of the constant-coefficient far-field ODE.'''
    a = 0.5 * sigma * sigma
    b = (r - q) - a
    discriminant = b * b + 4.0 * a * r
    return (-b - math.sqrt(discriminant)) / (2.0 * a)


def uniform_grid(N: int, s_max: float = S_MAX) -> np.ndarray:
    return np.linspace(0.0, s_max, N + 1)


def clustered_grid(
    N: int,
    center: float,
    s_max: float = S_MAX,
    strength: float = 8.0,
    width: float = 15.0,
) -> np.ndarray:
    '''Build a clustered grid with an explicit node at ``center``.'''
    return boundary_clustered_grid(
        N,
        center,
        s_max,
        strength=strength,
        width=width,
    )


def _sigma_values(
    S: np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    if callable(sigma):
        values = np.asarray(sigma(S), dtype=float)
    elif np.ndim(sigma) == 0:
        values = np.full(S.shape, float(sigma), dtype=float)
    else:
        values = np.asarray(sigma, dtype=float)
    if values.shape != S.shape:
        raise ValueError('sigma must be scalar, callable, or match S')
    if np.any(values <= 0.0):
        raise ValueError('sigma must be positive')
    return values


def build_nonuniform_operator(
    S: np.ndarray,
    K_value: float = K,
    r: float = R,
    q: float = Q,
    sigma: float | np.ndarray = SIGMA,
    boundary: str = 'dirichlet',
) -> NonuniformOperator:
    '''Build a fitted operator using left and right mesh spacings.'''
    S = np.asarray(S, dtype=float)
    if S.ndim != 1 or S.size < 3:
        raise ValueError('S must be a one-dimensional grid with at least 3 nodes')
    if np.any(np.diff(S) <= 0.0):
        raise ValueError('S must be strictly increasing')
    if boundary not in ('dirichlet', 'robin'):
        raise ValueError('invalid boundary type')

    n_int = S.size - 2
    S_int = S[1:-1]
    h_left = S_int - S[:-2]
    h_right = S[2:] - S_int
    width = 0.5 * (h_left + h_right)
    sigma_int = _sigma_values(S_int, sigma)
    a = 0.5 * sigma_int**2 * S_int**2
    b = (r - q) * S_int
    peclet_left = b * h_left / a
    peclet_right = b * h_right / a

    lower = -(a / (h_left * width)) * bernoulli(peclet_left)
    upper = -(a / (h_right * width)) * bernoulli(-peclet_right)
    diagonal = -lower - upper + r
    rhs = np.zeros(n_int)
    rhs[0] = -lower[0] * K_value

    tail_ratio = 0.0
    if boundary == 'robin':
        lam = asymptotic_lambda(r=r, q=q, sigma=sigma)
        h_last = float(S[-1] - S[-2])
        tail_ratio = float(S[-1] / (S[-1] - lam * h_last))
        diagonal[-1] += upper[-1] * tail_ratio

    return NonuniformOperator(
        N=S.size - 1,
        S=S,
        psi=np.maximum(K_value - S, 0.0),
        lower=lower,
        diag=diagonal,
        upper=upper,
        rhs=rhs,
        V_left=K_value,
        V_right=0.0,
        h=float(np.median(np.diff(S))),
        boundary=boundary,
        tail_ratio=tail_ratio,
    )


def interior_matrix(op: NonuniformOperator) -> sp.csr_matrix:
    '''Return the tridiagonal interior matrix associated with ``op``.'''
    n_int = op.N - 1
    return sp.diags(
        (op.lower[1:], op.diag, op.upper[:-1]),
        offsets=(-1, 0, 1),
        shape=(n_int, n_int),
        format='csr',
    )


def solve_operator(
    op: NonuniformOperator,
    tol: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float | bool], int, float]:
    '''Solve one LCP and reconstruct the Robin boundary value.'''
    V, iterations, residual = solve_lcp(op, tol=tol)
    if op.boundary == 'robin':
        V[-1] = op.tail_ratio * V[-2]
    diagnostics = compute_lcp_diagnostics(op, V)
    return V, diagnostics, iterations, residual


def extract_boundary_high_order(
    S_int: np.ndarray,
    slack: np.ndarray,
    tol_slack: float = TOL_SLACK,
    n_points: int = 4,
) -> tuple[float, float, float]:
    '''Extrapolate the boundary from a multi-point fit of sqrt(slack).'''
    contact = slack <= tol_slack
    first_continuation = np.flatnonzero(~contact)
    if first_continuation.size == 0:
        return float('nan'), float('nan'), float('nan')
    i1 = int(first_continuation[0])
    if i1 == 0:
        return float('nan'), float('nan'), float('nan')

    S_lo = float(S_int[i1 - 1])
    S_hi = float(S_int[i1])
    available = min(n_points, S_int.size - i1)
    if available < 3:
        return extract_free_boundary(S_int, slack, tol_slack)

    x = S_int[i1:i1 + available] - S_hi
    y = np.sqrt(np.maximum(slack[i1:i1 + available], 0.0))
    coefficients = np.polyfit(x, y, 1)
    roots = np.roots(coefficients)
    candidates = [
        S_hi + float(root.real)
        for root in roots
        if abs(root.imag) < 1e-10
        and S_lo - 1e-12 <= S_hi + float(root.real) <= S_hi + 1e-12
    ]
    if not candidates:
        return extract_free_boundary(S_int, slack, tol_slack)
    estimate = min(candidates, key=lambda value: abs(value - S_lo))
    return S_lo, S_hi, float(estimate)


def adaptive_study() -> list[dict[str, float | str]]:
    '''Compare aligned uniform and clustered grids at equal ``N``.

    Both grids contain a node at the known analytic boundary.  This removes
    the free-boundary phase effect from the maximum norm, so the error ratio
    under mesh halving measures the spatial order rather than the changing
    distance from the nearest node to the kink.
    '''
    exact_boundary = exact_free_boundary()
    center = exact_boundary
    rows: list[dict[str, float | str]] = []
    previous_error: dict[str, float] = {}

    for N in (600, 1200, 2400, 4800):
        grids = (
            ('uniform', boundary_aligned_grid(N, center, S_MAX)),
            ('clustered', clustered_grid(N, center, S_MAX, strength=16.0, width=15.0)),
        )
        grid_errors: dict[str, float] = {}
        for name, S in grids:
            op = build_nonuniform_operator(S, boundary='robin')
            V, diagnostics, iterations, _ = solve_operator(op)
            slack = V[1:-1] - op.psi[1:-1]
            boundary_high = extract_boundary_high_order(S[1:-1], slack)[2]
            exact_values = exact_perpetual_put(S)
            spacing = np.diff(S)
            max_error = float(np.max(np.abs(V - exact_values)))
            grid_errors[name] = max_error
            ratio = (
                previous_error[name] / max_error
                if name in previous_error and max_error > 0.0
                else float('nan')
            )
            rows.append(
                {
                    'N': float(N),
                    'grid': name,
                    'max_error': max_error,
                    'error_ratio': ratio,
                    'observed_order': (
                        math.log2(ratio) if ratio > 0.0 else float('nan')
                    ),
                    'boundary_error': abs(float(diagnostics['S_free']) - exact_boundary),
                    'boundary_high_error': abs(float(boundary_high) - exact_boundary),
                    'min_h': float(np.min(spacing)),
                    'max_h': float(np.max(spacing)),
                    'iterations': float(iterations),
                }
            )
        rows[-2]['gain'] = grid_errors['uniform'] / grid_errors['clustered']
        rows[-1]['gain'] = rows[-2]['gain']
        previous_error.update(grid_errors)
    return rows


def robin_study() -> list[dict[str, float]]:
    '''Compare zero Dirichlet and asymptotic Robin far-field closures.'''
    target = 80.0
    h = 0.5
    exact_value = float(exact_perpetual_put(np.array([target]))[0])
    rows: list[dict[str, float]] = []
    for s_max in (100.0, 150.0, 200.0, 400.0, 800.0):
        N = int(round(s_max / h))
        S = uniform_grid(N, s_max)
        row = {'s_max': s_max, 'N': float(N), 'exact': exact_value}
        for boundary in ('dirichlet', 'robin'):
            op = build_nonuniform_operator(S, boundary=boundary)
            V, _, _, _ = solve_operator(op)
            value = float(np.interp(target, S, V))
            row[f'{boundary}_error'] = abs(value - exact_value)
        rows.append(row)
    return rows


def _benchmark_solution(
    S: np.ndarray,
    parameter: str,
    value: float,
) -> tuple[np.ndarray, NonuniformOperator]:
    params = {'r': R, 'q': Q, 'sigma': SIGMA}
    params[parameter] = value
    op = build_nonuniform_operator(S, **params)
    V, _, _, _ = solve_operator(op)
    return V, op


def _interpolation_weights(S: np.ndarray, target: float) -> np.ndarray:
    n_int = S.size - 2
    weights = np.zeros(n_int)
    upper = int(np.searchsorted(S, target))
    upper = min(max(upper, 1), S.size - 1)
    lower = upper - 1
    fraction = (target - S[lower]) / (S[upper] - S[lower])
    if lower >= 1:
        weights[lower - 1] += 1.0 - fraction
    if upper <= S.size - 2:
        weights[upper - 1] += fraction
    return weights


def adjoint_sensitivity(
    parameter: str,
    target: float = 100.0,
    N: int = 1200,
    eps: float = 1e-5,
) -> dict[str, float | str]:
    '''Return adjoint, direct finite-difference, and exact sensitivities.'''
    S = uniform_grid(N, S_MAX)
    params = {'r': R, 'q': Q, 'sigma': SIGMA}
    base = float(params[parameter])
    plus = dict(params)
    minus = dict(params)
    plus[parameter] = base + eps
    minus[parameter] = base - eps

    op0 = build_nonuniform_operator(S, **params)
    V0, _, _, _ = solve_operator(op0)
    op_plus = build_nonuniform_operator(S, **plus)
    op_minus = build_nonuniform_operator(S, **minus)

    A0 = interior_matrix(op0)
    dA = (
        interior_matrix(op_plus)
        - interior_matrix(op_minus)
    ) / (2.0 * eps)
    drhs = (op_plus.rhs - op_minus.rhs) / (2.0 * eps)

    slack = V0[1:-1] - op0.psi[1:-1]
    free = np.flatnonzero(slack > TOL_SLACK)
    weights = _interpolation_weights(S, target)
    forcing = drhs[free] - dA[free, :] @ V0[1:-1]
    adjoint = spsolve(A0[free, :][:, free].T.tocsc(), weights[free])
    adjoint_value = float(adjoint @ forcing)

    V_plus, _ = _benchmark_solution(S, parameter, base + eps)
    V_minus, _ = _benchmark_solution(S, parameter, base - eps)
    direct = (
        float(np.interp(target, S, V_plus))
        - float(np.interp(target, S, V_minus))
    ) / (2.0 * eps)

    exact_eps = 1e-6
    exact_plus = dict(params)
    exact_minus = dict(params)
    exact_plus[parameter] = base + exact_eps
    exact_minus[parameter] = base - exact_eps
    exact_value = (
        float(exact_perpetual_put(np.array([target]), **exact_plus)[0])
        - float(exact_perpetual_put(np.array([target]), **exact_minus)[0])
    ) / (2.0 * exact_eps)

    return {
        'parameter': parameter,
        'adjoint': adjoint_value,
        'direct': direct,
        'exact': exact_value,
        'adjoint_error': abs(adjoint_value - exact_value),
        'direct_error': abs(direct - exact_value),
    }


def plot_advanced_results(
    adaptive: list[dict[str, float | str]],
    robin: list[dict[str, float]],
    adjoint: list[dict[str, float | str]],
) -> Path:
    '''Create the four-panel summary figure.'''
    output = Path(__file__).resolve().parent / 'advanced_methods.png'
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

    N_plot = 2400
    center = exact_free_boundary()
    S_uniform = boundary_aligned_grid(N_plot, center, S_MAX)
    S_clustered = clustered_grid(N_plot, center, S_MAX, strength=16.0, width=15.0)
    axes[0, 0].plot(S_clustered[:-1], np.diff(S_clustered), color='tab:red', label='clustered')
    axes[0, 0].plot(S_uniform[:-1], np.diff(S_uniform), color='tab:blue', label='uniform')
    axes[0, 0].axvline(center, color='black', linestyle='--', linewidth=1.0)
    axes[0, 0].set_xlim(0.0, 180.0)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('Stock price S')
    axes[0, 0].set_ylabel('Local mesh spacing')
    axes[0, 0].set_title('(a) Grid clustering')
    axes[0, 0].legend()

    for grid_name, color in (('uniform', 'tab:blue'), ('clustered', 'tab:red')):
        rows = [row for row in adaptive if row['grid'] == grid_name]
        axes[0, 1].loglog(
            [row['N'] for row in rows],
            [row['max_error'] for row in rows],
            marker='o',
            color=color,
            label=grid_name,
        )
    axes[0, 1].set_xlabel('N')
    axes[0, 1].set_ylabel(r'Maximum value error')
    axes[0, 1].set_title('(b) Adaptive value error')
    axes[0, 1].legend()

    axes[1, 0].semilogy(
        [row['s_max'] for row in robin],
        [row['dirichlet_error'] for row in robin],
        marker='o',
        color='tab:blue',
        label='V(S_max)=0',
    )
    axes[1, 0].semilogy(
        [row['s_max'] for row in robin],
        [row['robin_error'] for row in robin],
        marker='s',
        color='tab:red',
        label='asymptotic Robin',
    )
    axes[1, 0].set_xlabel(r'$S_{\max}$')
    axes[1, 0].set_ylabel(r'Error in V(80)')
    axes[1, 0].set_title('(c) Far-field closure')
    axes[1, 0].legend()

    labels = [str(row['parameter']) for row in adjoint]
    relative = [
        100.0 * float(row['adjoint_error']) / abs(float(row['exact']))
        for row in adjoint
    ]
    axes[1, 1].bar(labels, relative, color='tab:green')
    axes[1, 1].set_ylabel('Relative error (%)')
    axes[1, 1].set_title('(d) Adjoint sensitivity accuracy')

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output


def main() -> None:
    adaptive = adaptive_study()
    robin = robin_study()
    adjoint = [adjoint_sensitivity(name, N=4800) for name in ('sigma', 'r', 'q')]

    print('Adaptive-grid study')
    for row in adaptive:
        print(row)
    adaptive_file = Path(__file__).resolve().with_name('adaptive_convergence.csv')
    with adaptive_file.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(adaptive[0]),
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(adaptive)
    print()
    print('Robin far-field study')
    for row in robin:
        print(row)
    print()
    print('Adjoint sensitivities at S=100, N=4800')
    for row in adjoint:
        print(row)
    print()
    print(f'Saved {plot_advanced_results(adaptive, robin, adjoint)}')
    print(f'Saved {adaptive_file.name}')


if __name__ == '__main__':
    main()
