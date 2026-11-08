# Numerical Valuation of Perpetual American Put Options via LCP

## Overview

This project solves the perpetual American put option problem as a linear
complementarity problem (LCP). The option value `V(S)` must satisfy

```text
V(S) >= psi(S) = max(K - S, 0),
-L V(S) >= 0,
(-L V(S))(V(S) - psi(S)) = 0,
```

where

```text
L V = 0.5 sigma(S)^2 S^2 V'' + (r - q) S V' - r V.
```

The spatial operator uses exponential fitting (the Scharfetter--Gummel
scheme). Its off-diagonals are non-positive and its diagonal is positive at
every cell Peclet number, so the discrete operator is an M-matrix without a
limiter or start-up correction. The LCP is solved by policy iteration with
sparse direct linear solves, starting from `V = psi` and stopping when
`max|dV| < 1e-12` with a stable active set.

Both production stages use the truncation boundary condition
`V(S_max) = 0`. The constant-volatility convergence study uses
`S_max = 2400` (24K), with the mesh spacing refined from `h = 4` to
`h = 0.5`. Its order study is performed on a grid with a node exactly at the
analytic free boundary, which removes the kink-phase effect. This is a
stencil-verification experiment, not a blind free-boundary recovery test.
The production contact brackets are reported separately: at `N = 600`,
`N = 1200`, and `N = 4800` they lie to the right of the analytic boundary,
while at `N = 2400` they contain it. The production
local-volatility stage uses `q = 0.02` and an independent DOP853 shooting
solution of the free-boundary ODE; no `N = 9600` fine-grid reference is used.
At `N = 1200`, the local-volatility contact bracket `[70, 72]` misses the
finite-domain reference `69.1701793435`; the CSV and report expose this rather
than treating it as order evidence.

`advanced_methods.py` adds four solver-independent extensions: a nonuniform
exponential-fitting operator, an asymptotic Robin far-field closure,
multi-point free-boundary extraction, and sparse discrete adjoint
sensitivities. The adaptive and Robin studies are compared against the
uniform Dirichlet baseline, while the adjoint derivatives are checked against
direct finite differences and the exact constant-volatility formulas.

`stage2d_spread.py` adds one controlled two-dimensional extension: a
perpetual American spread put with payoff `max(K - S1 + S2, 0)`. The experiment
is deliberately restricted to constant volatilities and zero correlation, so
the five-point fitted matrix remains an M-matrix. The high-price edges use a
local payoff/tail closure and are not claimed to be exact infinite-domain
boundaries. A fixed-mesh `L = 240, 300, 400` study quantifies this truncation.

For the local-volatility case, the pure finite-domain truncation in `V(100)`
at `S_max = 2400` is `5.578e-3`. The production `N = 2400` total error is
`4.055e-3` because spatial and truncation errors partially cancel. These
quantities are kept separate throughout the report.

`experiment_neumann_true.py` is a synthetic boundary-row stress test, not an
admissible American-put left boundary. Its zero-flux closure produces
`V(0) = 99 < K`, so it violates the obstacle at `S = 0` and is not used as a
price or a convergence benchmark.

Free boundaries are reported as a discrete contact bracket
`[S_lo, S_hi]`: `S_lo` is the last active node and `S_hi` the first
continuation node, with a documented slackness tolerance of `1e-10*K`. The
estimate `S_h*` uses local linear interpolation of `sqrt(V_h - psi_h)`,
clamped to the bracket. A bracket is not a confidence interval and can miss
the continuous boundary on a coarse mesh; this is reported explicitly.

## File Structure

### Entry Point

| File | Description |
| --- | --- |
| `run_all.py` | Sequentially runs all experiments and plotting scripts, then verifies that every expected figure and numerical table was generated. |

### Core Solvers

| File | Description |
| --- | --- |
| `stage1_validation.py` | Defines the shared fitted operator, the LCP solver, parameter-aware closed-form perpetual-put values, LCP diagnostics, and free-boundary extraction. Validates the constant-volatility case. |
| `stage2_local_volatility.py` | Extends the fitted solver to `sigma(S) = 0.20(1 + 0.35 tanh(log(S/K)/0.40))`, reports the full LCP residual/shape diagnostics, and compares against the independent ODE shooting reference. |
| `reference_solution.py` | DOP853 shooting solver for the free-boundary ODE. It provides both the infinite-domain asymptotic reference and the finite-domain `V(S_max)=0` reference, and self-checks against the constant-volatility closed form. |
| `grids.py` | Boundary-aligned uniform and clustered grids used only in validation experiments. |
| `local_vol_model.py` | Shared local-volatility profile and Stage 2 parameters. |

### Experiments

| File | Description |
| --- | --- |
| `experiment_central_diff.py` | Demonstrates that central drift differencing breaks the M-matrix property and introduces spurious near-origin behavior. |
| `experiment_smax.py` | Studies domain truncation at fixed mesh spacing for `S_max/K` from 3 upward. |
| `experiment_solvers.py` | Benchmarks policy iteration against PSOR in iteration count and CPU time on the same fitted LCP. PSOR stops when the componentwise LCP residual is below `1e-10`, not on an iterate-difference test. |
| `experiment_verification.py` | Checks LCP residuals, complementarity, monotonicity, convexity, option bounds, and contact-set contiguity. |
| `experiment_convergence.py` | Recomputes the aligned-grid constant-volatility convergence table plus production contact brackets and writes `stage1_convergence.csv`. |
| `stage2_convergence_report.py` | Recomputes the four-mesh local-volatility study against the independent ODE reference and writes `stage2_convergence.csv`. |
| `stage2d_spread.py` | Solves a controlled `rho = 0` American spread-put LCP on three two-dimensional meshes, checks the five-point M-matrix and complementarity residual, performs a domain-truncation check, and writes the 2D plot and CSV tables. |
| `advanced_methods.py` | Runs adaptive-grid, Robin-closure, multi-point boundary-extraction, and adjoint-sensitivity experiments and writes `advanced_methods.png` and `adaptive_convergence.csv`. |

### Plotting Scripts

| File | Description |
| --- | --- |
| `plot_lcp_verification.py` | Plots obstacle slackness and the LCP residual across the exercise and continuation regions. |
| `plot_solver_comparison.py` | Produces log-scale iteration-count and CPU-time comparisons from `experiment_solvers.benchmark()`. |
| `plot_smax_study.py` | Plots `V(100)` against `S_max` using the fixed-`h` sweep from `experiment_smax.sweep()`. |
| `plot_greeks.py` | Computes Delta and Gamma for the local-volatility solution and marks the computed free-boundary bracket. |
| `plot_matrix_spy.py` | Builds the fitted and central-difference sparse matrices and saves their spy plots, highlighting positive central-difference off-diagonals in red. |

## Generated Figures

| Figure | Caption |
| --- | --- |
| `free_boundary_plot.png` | Constant-volatility solution, payoff, exact free boundary, and the computed bracket. |
| `stage2_plot.png` | Local-volatility versus constant-volatility option values at `N = 2400`. |
| `central_diff_oscillations.png` | Spurious near-origin behavior produced by central drift differencing. |
| `lcp_verification.png` | Obstacle slackness and LCP residual across the exercise and continuation regions. |
| `solver_comparison.png` | Iteration-count and CPU-time comparison between policy iteration and PSOR. |
| `smax_study.png` | Fixed-`h` domain-truncation study for `V(100)`. |
| `greeks.png` | Delta and Gamma for the local-volatility solution around the free boundary. |
| `advanced_methods.png` | Adaptive-grid refinement, Robin closure, boundary extraction, and adjoint sensitivities. |
| `matrix_spy_fitted.png` | Sparsity pattern of the fitted operator on a `100 x 100` illustrative grid. |
| `matrix_spy_central.png` | Sparsity pattern of the central-difference operator with positive off-diagonals highlighted in red. |
| `matrix_spy_fitted_n400.png` | Tridiagonal sparsity pattern of the assembled fitted matrix at `N = 400`. |
| `spread2d_plot.png` | Fitted value surface and exercise/contact set for the controlled American spread-put LCP on a `321 x 321` grid. |

## Generated Numerical Tables

| File | Contents |
| --- | --- |
| `stage1_convergence.csv` | Aligned-grid constant-volatility errors, ratios, observed orders, LCP residuals, and production contact brackets. |
| `stage2_convergence.csv` | Production and aligned local-volatility errors, observed orders, production brackets, free-boundary locations, and ODE-reference metadata. |
| `adaptive_convergence.csv` | Uniform versus boundary-aligned clustered errors, ratios, orders, and clustering gains. |
| `spread2d_results.csv` | Three-mesh 2D spread-put iteration counts, wall times, pricing-window errors, LCP residuals, and M-matrix diagnostics. |
| `spread2d_domain.csv` | Fixed-mesh `L = 240, 300, 400` domain-truncation check at `(S1,S2) = (100,50)`. |

## How to Run

From the project directory, reproduce every experiment and figure with:

```bash
python run_all.py
```

The runner uses Matplotlib's noninteractive `Agg` backend so figures are saved
without blocking the sequence. To display an individual figure window, run its
plotting script directly, for example:

```bash
python plot_greeks.py
```

## Requirements

- Python 3.10 or newer
- `numpy`
- `scipy`
- `matplotlib`

Install the dependencies with:

```bash
python -m pip install numpy scipy matplotlib
```
