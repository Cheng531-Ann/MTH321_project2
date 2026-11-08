'''Reproduce the figures and numerical tables used in the final report.'''

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


SCRIPTS = (
    'stage1_validation.py',
    'stage2_local_volatility.py',
    'stage2_convergence_report.py',
    'experiment_convergence.py',
    'stage2d_spread.py',
    'experiment_central_diff.py',
    'experiment_smax.py',
    'experiment_solvers.py',
    'experiment_neumann_true.py',
    'experiment_verification.py',
    'plot_matrix_spy.py',
    'advanced_methods.py',
    'plot_lcp_verification.py',
    'plot_solver_comparison.py',
    'plot_smax_study.py',
    'plot_greeks.py',
)

EXPECTED_FIGURES = (
    'matrix_spy_fitted.png',
    'matrix_spy_central.png',
    'matrix_spy_fitted_n400.png',
    'central_diff_oscillations.png',
    'lcp_verification.png',
    'free_boundary_plot.png',
    'solver_comparison.png',
    'stage2_plot.png',
    'smax_study.png',
    'greeks.png',
    'neumann_true_comparison.png',
    'advanced_methods.png',
    'spread2d_plot.png',
)

EXPECTED_DATA = (
    'stage1_convergence.csv',
    'stage2_convergence.csv',
    'adaptive_convergence.csv',
    'tolerance_sweep.csv',
    'spread2d_results.csv',
    'spread2d_domain.csv',
)


def main() -> None:
    root = Path(__file__).resolve().parent
    env = os.environ.copy()

    # A noninteractive backend prevents plt.show() from blocking the batch run.
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONUNBUFFERED'] = '1'

    for filename in SCRIPTS:
        script_path = root / filename
        if not script_path.exists():
            raise FileNotFoundError(f'missing required script: {script_path}')

        print(f'Running {filename}...')
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=root,
            env=env,
            check=True,
        )

    missing = [name for name in EXPECTED_FIGURES if not (root / name).exists()]
    if missing:
        raise RuntimeError(f'missing generated figures: {missing}')

    missing_data = [name for name in EXPECTED_DATA if not (root / name).exists()]
    if missing_data:
        raise RuntimeError(f'missing generated data files: {missing_data}')

    print()
    print('Generated figures:')
    for filename in EXPECTED_FIGURES:
        print(f'  - {filename}')
    print('Generated numerical tables:')
    for filename in EXPECTED_DATA:
        print(f'  - {filename}')
    print('All report figures and numerical tables generated successfully!')


if __name__ == '__main__':
    main()
