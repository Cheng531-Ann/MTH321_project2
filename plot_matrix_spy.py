'''Plot sparsity patterns of the fitted and central-difference matrices.'''

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


MATRIX_SIZE = 100
BENCHMARK_N = 400
R = 0.05
Q = 0.0
SIGMA = 0.1
BENCHMARK_SIGMA = 0.20
S_MAX = 400.0


def bernoulli(z: np.ndarray) -> np.ndarray:
    '''Evaluate B(z) = z / (exp(z) - 1) without cancellation.'''
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)

    small = np.abs(z) < 1e-8
    out[small] = 1.0 - 0.5 * z[small] + z[small] ** 2 / 12.0

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        out[~small] = z[~small] / np.expm1(z[~small])
    return out


def _interior_grid() -> tuple[np.ndarray, float]:
    h = S_MAX / (MATRIX_SIZE + 1)
    S = np.arange(1, MATRIX_SIZE + 1, dtype=float) * h
    return S, h


def build_fitted_matrix() -> sp.csr_matrix:
    '''Build the exponentially fitted operator A = -L_h.'''
    S, h = _interior_grid()
    a = 0.5 * SIGMA**2 * S**2
    b = (R - Q) * S
    peclet = b * h / a
    scale = a / h**2

    lower = -scale * bernoulli(peclet)
    upper = -scale * bernoulli(-peclet)
    diagonal = scale * (bernoulli(peclet) + bernoulli(-peclet)) + R

    return sp.diags(
        (lower[1:], diagonal, upper[:-1]),
        offsets=(-1, 0, 1),
        shape=(MATRIX_SIZE, MATRIX_SIZE),
        format='csr',
    )


def build_fitted_matrix_n400() -> sp.csr_matrix:
    '''Build the fitted system matrix for the report benchmark N=400.'''
    h = S_MAX / BENCHMARK_N
    S = np.arange(1, BENCHMARK_N, dtype=float) * h
    n_interior = BENCHMARK_N - 1
    a = 0.5 * BENCHMARK_SIGMA**2 * S**2
    b = (R - Q) * S
    peclet = b * h / a
    scale = a / h**2

    lower = -scale * bernoulli(peclet)
    upper = -scale * bernoulli(-peclet)
    diagonal = scale * (bernoulli(peclet) + bernoulli(-peclet)) + R

    return sp.diags(
        (lower[1:], diagonal, upper[:-1]),
        offsets=(-1, 0, 1),
        shape=(n_interior, n_interior),
        format='csr',
    )


def build_central_matrix() -> sp.csr_matrix:
    '''Build the central-difference operator A = -L_h.'''
    S, h = _interior_grid()
    diffusion = 0.5 * SIGMA**2 * S**2 / h**2
    drift = 0.5 * (R - Q) * S / h

    lower = -diffusion + drift
    upper = -diffusion - drift
    diagonal = 2.0 * diffusion + R

    return sp.diags(
        (lower[1:], diagonal, upper[:-1]),
        offsets=(-1, 0, 1),
        shape=(MATRIX_SIZE, MATRIX_SIZE),
        format='csr',
    )


def positive_off_diagonal(matrix: sp.spmatrix) -> sp.csr_matrix:
    '''Return only the positive off-diagonal entries of a sparse matrix.'''
    coo = matrix.tocoo()
    keep = (coo.row != coo.col) & (coo.data > 0.0)
    return sp.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=matrix.shape,
    )


def plot_spy(
    matrix: sp.spmatrix,
    output_path: Path,
    title: str,
    highlight: sp.spmatrix | None = None,
    markersize: float = 4.0,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.spy(
        matrix,
        marker='s',
        markersize=markersize,
        color='tab:blue',
        label='nonzero entry',
    )
    if highlight is not None and highlight.nnz:
        ax.spy(
            highlight,
            marker='s',
            markersize=2.0 * markersize,
            markerfacecolor='red',
            markeredgecolor='red',
            label='positive off-diagonal',
        )

    ax.set_title(title)
    ax.set_xlabel('column index')
    ax.set_ylabel('row index')
    ax.legend(loc='upper right', framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    fitted_path = output_dir / 'matrix_spy_fitted.png'
    central_path = output_dir / 'matrix_spy_central.png'
    fitted_n400_path = output_dir / 'matrix_spy_fitted_n400.png'
    fitted = build_fitted_matrix()
    central = build_central_matrix()
    fitted_n400 = build_fitted_matrix_n400()

    fitted_positive = positive_off_diagonal(fitted)
    central_positive = positive_off_diagonal(central)
    fitted_n400_positive = positive_off_diagonal(fitted_n400)
    if fitted_positive.nnz:
        raise RuntimeError('the fitted matrix has a positive off-diagonal entry')
    if not central_positive.nnz:
        raise RuntimeError('the central matrix has no positive off-diagonal entry')
    if fitted_n400_positive.nnz:
        raise RuntimeError('the N=400 fitted matrix has a positive off-diagonal entry')

    plot_spy(
        fitted,
        fitted_path,
        'Exponential-Fitting Matrix (100 x 100)',
    )
    plot_spy(
        central,
        central_path,
        'Central-Difference Matrix (100 x 100)',
        highlight=central_positive,
    )
    plot_spy(
        fitted_n400,
        fitted_n400_path,
        'Exponential-Fitting Matrix (N=400)',
        markersize=0.3,
    )

    print(f'Saved {fitted_path}')
    print(f'Saved {central_path}')
    print(f'Saved {fitted_n400_path}')
    print(f'Central-difference positive off-diagonals: {central_positive.nnz}')


if __name__ == '__main__':
    main()
