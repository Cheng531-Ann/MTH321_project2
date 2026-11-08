"""Experiment: perpetual American put with central drift differencing.

This script intentionally uses central differencing for V' instead of the
exponential-fitting stencil used in the validated solvers.  The resulting discrete
operator is not an M-matrix and can produce spurious oscillations near S=0.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def solve_central_diff_put(
    N: int,
    K: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    S_max: float = 400.0,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Solve the LCP using central differencing for the drift term."""
    h = S_max / N
    S = np.linspace(0.0, S_max, N + 1)
    S_int = S[1:-1]
    n_int = N - 1
    psi = np.maximum(K - S, 0.0)
    psi_int = psi[1:-1]

    diffusion = 0.5 * sigma**2 * S_int**2 / h**2
    central_drift = 0.5 * (r - q) * S_int / h

    # A = -L_h with central drift differencing.
    lower_full = -diffusion + central_drift
    diag_full = 2.0 * diffusion + r
    upper_full = -diffusion - central_drift

    V_right = 0.0
    b_boundary = np.zeros(n_int)
    b_boundary[0] = -lower_full[0] * K
    b_boundary[-1] += -upper_full[-1] * V_right

    A_full = sp.diags(
        (lower_full[1:], diag_full, upper_full[:-1]),
        offsets=(-1, 0, 1),
        shape=(n_int, n_int),
        format="csr",
    )

    V = psi.copy()
    V[0] = K
    V[-1] = V_right

    residual = math.inf
    iterations = 0
    previous_active: np.ndarray | None = None

    for iterations in range(1, max_iter + 1):
        V_old = V.copy()

        lcp_residual = A_full @ V[1:-1] - b_boundary
        active = V <= psi + 1e-10
        active_int = active[1:-1].copy()
        active_int[active_int & (lcp_residual < -1e-10)] = False
        policy_is_stable = (
            previous_active is not None
            and np.array_equal(active_int, previous_active)
        )
        previous_active = active_int.copy()

        diag = diag_full.copy()
        lower = lower_full[1:].copy()
        upper = upper_full[:-1].copy()
        diag[active_int] = 1.0
        if n_int > 1:
            lower[active_int[1:]] = 0.0
            upper[active_int[:-1]] = 0.0

        A = sp.diags(
            (lower, diag, upper),
            offsets=(-1, 0, 1),
            shape=(n_int, n_int),
            format="csr",
        )

        b = b_boundary.copy()
        b[active_int] = psi_int[active_int]
        V_new_int = spsolve(A, b)
        V[1:-1] = np.maximum(V_new_int, psi_int)
        V[0] = K
        V[-1] = V_right

        residual = float(np.max(np.abs(V - V_old)))
        if residual < tol and policy_is_stable:
            break
    else:
        raise RuntimeError(
            f"policy iteration did not converge in {max_iter} iterations "
            f"(last residual={residual:.3e})"
        )

    return S, V, iterations, residual


def solve_central_diff_linear(
    N: int,
    K: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    S_max: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the linear central-difference system without obstacle projection.

    This exposes the unstable mode that the LCP active-set projection masks in
    the exercise region.
    """
    h = S_max / N
    S = np.linspace(0.0, S_max, N + 1)
    S_int = S[1:-1]
    n_int = N - 1

    diffusion = 0.5 * sigma**2 * S_int**2 / h**2
    central_drift = 0.5 * (r - q) * S_int / h
    lower_full = -diffusion + central_drift
    diag_full = 2.0 * diffusion + r
    upper_full = -diffusion - central_drift

    V_right = 0.0
    b_boundary = np.zeros(n_int)
    b_boundary[0] = -lower_full[0] * K
    b_boundary[-1] += -upper_full[-1] * V_right

    A = sp.diags(
        (lower_full[1:], diag_full, upper_full[:-1]),
        offsets=(-1, 0, 1),
        shape=(n_int, n_int),
        format="csr",
    )
    V_int = spsolve(A, b_boundary)
    return S, np.concatenate(([K], V_int, [V_right]))


def main() -> tuple[np.ndarray, np.ndarray]:
    """Run N=400, print the M-matrix warning, and plot the oscillations."""
    N = 400
    S, V, iterations, residual = solve_central_diff_put(N=N)
    S_linear, V_linear = solve_central_diff_linear(N=N)

    h = 400.0 / N
    S_int = S[1:-1]
    diffusion = 0.5 * 0.2**2 * S_int**2 / h**2
    central_drift = 0.5 * 0.05 * S_int / h
    lower = -diffusion + central_drift
    upper = -diffusion - central_drift
    diagonal = 2.0 * diffusion + 0.05

    positive_off_diagonal = np.count_nonzero(lower > 1e-14) + np.count_nonzero(
        upper > 1e-14
    )

    print()
    print("Central-difference drift experiment, N=400")
    print(
        f"Policy iterations={iterations}, final update={residual:.3e}, "
        f"minimum V={V.min():.6e}"
    )
    print(
        "WARNING: the M-matrix property is broken.  Central differencing "
        "produces positive off-diagonal entries."
    )
    print(
        f"         Positive off-diagonal entries in A=-L_h: "
        f"{positive_off_diagonal}; minimum diagonal={diagonal.min():.6e}"
    )
    print(
        f"         Unprojected central solve minimum V={V_linear.min():.6e} "
        f"at S={S_linear[np.argmin(V_linear)]:.3f}."
    )
    print(
        "         The LCP obstacle projection masks the unstable mode in the "
        "exercise region; the unprojected solve exposes it."
    )
    print()

    plt.figure(figsize=(8, 5))
    plt.plot(
        S,
        V,
        color="blue",
        linestyle="-",
        label="Central-difference LCP (projected)",
    )
    plt.plot(
        S_linear,
        V_linear,
        color="darkorange",
        linestyle="--",
        label="Unprojected central-difference solve",
    )
    plt.plot(
        S,
        np.maximum(100.0 - S, 0.0),
        color="black",
        linestyle="--",
        label="Payoff",
    )
    plt.axhline(0.0, color="red", linestyle=":", linewidth=1.0)
    plt.xlim(0.0, 20.0)
    plt.xlabel("Stock price S")
    plt.ylabel("Option value V(S)")
    plt.title("Central Drift Differencing: Unstable Near-Origin Mode (N=400)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("central_diff_oscillations.png", dpi=150)
    plt.show()

    return S, V


if __name__ == "__main__":
    central_S, central_V = main()
