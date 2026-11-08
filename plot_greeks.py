"""Plot Delta and Gamma for the Stage 2 local-volatility solution."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stage2_local_volatility import solve_local_volatility_put


def main() -> None:
    result = solve_local_volatility_put(N=2400)
    S = result.S
    V = result.V

    delta = np.gradient(V, S)
    gamma = np.gradient(delta, S)
    S_star = result.S_free
    delta_at_100 = float(np.interp(100.0, S, delta))
    max_gamma = float(np.max(gamma))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(S, delta, color="blue", linewidth=1.8, label=r"$\Delta$")
    axes[0].axhline(-1.0, color="black", linestyle=":", linewidth=1.1)
    axes[0].axvline(
        S_star,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=rf"$S_h^*={S_star:.2f}$",
    )
    axes[0].axvspan(
        result.S_lo,
        result.S_hi,
        color="red",
        alpha=0.12,
        label=rf"bracket $[{result.S_lo:.2f}, {result.S_hi:.2f}]$",
    )
    axes[0].set_xlim(50.0, 100.0)
    axes[0].set_xlabel("Stock price S")
    axes[0].set_ylabel(r"$\Delta=\partial V/\partial S$")
    axes[0].set_title("Delta")
    axes[0].legend(loc="lower right")

    axes[1].plot(S, gamma, color="darkgreen", linewidth=1.8, label=r"$\Gamma$")
    axes[1].axvline(
        S_star,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=rf"$S_h^*={S_star:.2f}$",
    )
    axes[1].axvspan(
        result.S_lo,
        result.S_hi,
        color="red",
        alpha=0.12,
        label=rf"bracket $[{result.S_lo:.2f}, {result.S_hi:.2f}]$",
    )
    axes[1].set_xlim(50.0, 100.0)
    axes[1].set_xlabel("Stock price S")
    axes[1].set_ylabel(r"$\Gamma=\partial^2 V/\partial S^2$")
    axes[1].set_title("Gamma")
    axes[1].legend(loc="upper right")

    fig.suptitle("Stage 2 Local-Volatility Greeks (N=2400)", fontsize=13)
    fig.tight_layout()
    print(f"Delta(100) = {delta_at_100:.6f}")
    print(f"max Gamma = {max_gamma:.6e}")
    print("Saved greeks.png")
    plt.savefig(Path(__file__).with_name("greeks.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
