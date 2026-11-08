"""Plot the fixed-h local-volatility domain-truncation study."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiment_smax import sweep
from reference_solution import build_reference


def main() -> None:
    rows = sweep()
    S_max = np.array([row["S_max"] for row in rows])
    V_100 = np.array([row["V_100"] for row in rows])
    reference = build_reference(domain="infinite").value_at(100.0)

    plt.figure(figsize=(8, 5))
    plt.plot(
        S_max,
        V_100,
        color="blue",
        marker="o",
        linewidth=1.8,
        label=r"Numerical $V(100)$ at fixed $h=1$",
    )
    plt.axhline(
        reference,
        color="red",
        linestyle="--",
        linewidth=1.4,
        label=rf"ODE reference: ${reference:.6f}$",
    )

    for s_value, v_value in zip(S_max, V_100):
        plt.annotate(
            f"{v_value:.4f}",
            (s_value, v_value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
        )

    plt.xlim(250.0, 2500.0)
    plt.xlabel(r"Computational domain boundary $S_{max}$")
    plt.ylabel(r"Option value at $S=100$")
    plt.title(r"Effect of Domain Truncation on $V(100)$ (fixed $h=1$)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("smax_study.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
