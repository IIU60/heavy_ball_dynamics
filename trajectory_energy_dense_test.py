"""
Short test: the three (A, γ, Ω) cases from trajectory_visualization.py __main__,
with a brief integration window.

Reference IC plus two small perturbations (same as trajectory_visualization /
poincare_map). Energy E(t) = ½(v_x² + v_y²) + f(x, y) is evaluated on a shared
dense uniform time grid so all three curves use the same abscissa.

Left: three (x, y) trajectories (tab:blue / tab:orange / tab:green).
Right: E(t) for each trajectory in the matching color + legend.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from heavy_ball_system import potential, rhs
from trajectory_visualization import (
    ATOL,
    DEFAULT_INITIAL_STATE,
    DEFAULT_PERTURBATIONS,
    METHOD,
    RTOL,
    TRAJECTORY_SCATTER_ALPHA,
    TRAJECTORY_SCATTER_SIZE,
    draw_potential_contour_background,
)

TRAJECTORY_COLORS = ("tab:blue", "tab:orange", "tab:green")
TRAJECTORY_LABELS = ("reference", "perturbation 1", "perturbation 2")

# Legible legend text and larger color swatches (scatter + line handles).
LEGEND_KWARGS_XY = {
    "loc": "best",
    "fontsize": 11,
    "markerscale": 2.8,
    "framealpha": 0.95,
    "borderpad": 0.6,
    "labelspacing": 0.6,
}
LEGEND_KWARGS_E = {
    "loc": "best",
    "fontsize": 11,
    "handlelength": 3.2,
    "handletextpad": 0.9,
    "framealpha": 0.95,
    "borderpad": 0.6,
    "labelspacing": 0.6,
}

# Same three cases as trajectory_visualization.py __main__ (order preserved).
PARAMETER_CASES = (
    # (amplitude, gamma, omega, short label)
    (0.1, 0.5, 1.0, "convergence"),
    (0.3, 0.1, 1.0, "periodic locking"),
    (0.5, 0.5, 1.0, "chaotic behavior"),
)

# Much shorter than N_CYCLES_DEFAULT=600 (~5 forcing periods)
N_CYCLES = 20

# Shared output times for all ICs (adaptive sol.t differs per trajectory).
N_DENSE_POINTS = 4000


def _initial_states():
    base = np.asarray(DEFAULT_INITIAL_STATE, dtype=float)
    d1, d2 = DEFAULT_PERTURBATIONS
    return [base, base + np.asarray(d1), base + np.asarray(d2)]


def _integrate_uniform(
    initial_state,
    amplitude: float,
    gamma: float,
    omega: float,
    t_end: float,
    n_points: int,
):
    t_eval = np.linspace(0.0, t_end, n_points)
    return solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        (0.0, t_end),
        initial_state,
        t_eval=t_eval,
        method=METHOD,
        rtol=RTOL,
        atol=ATOL,
    )


def plot_one_case(amplitude: float, gamma: float, omega: float, case_label: str):
    T = 2 * np.pi / omega
    t_end = N_CYCLES * T
    states = _initial_states()

    fig, (ax_xy, ax_e) = plt.subplots(1, 2, figsize=(14, 6))
    draw_potential_contour_background(ax_xy)

    for y0, color, label in zip(states, TRAJECTORY_COLORS, TRAJECTORY_LABELS):
        sol = _integrate_uniform(
            y0, amplitude, gamma, omega, t_end, N_DENSE_POINTS
        )
        x, y, vx, vy = sol.y
        v_mag2 = vx**2 + vy**2
        energy = 0.5 * v_mag2 + potential(x, y)
        t = sol.t

        ax_xy.scatter(
            x,
            y,
            s=TRAJECTORY_SCATTER_SIZE,
            c=color,
            alpha=TRAJECTORY_SCATTER_ALPHA,
            linewidths=0,
            label=label,
            zorder=5,
        )
        ax_e.plot(
            t,
            energy,
            color=color,
            linewidth=1.4,
            label=label,
        )

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel(r"$x$")
    ax_xy.set_ylabel(r"$y$")
    ax_xy.set_title(r"Three trajectories on $f(x,y)$")
    ax_xy.legend(**LEGEND_KWARGS_XY)

    ax_e.set_xlabel(r"$t$")
    ax_e.set_ylabel(r"$E = \frac{1}{2}(v_x^2+v_y^2) + f(x,y)$")
    ax_e.set_title(r"Energy (shared dense $t$ grid)")
    ax_e.legend(**LEGEND_KWARGS_E)

    fig.suptitle(
        rf"Dense energy test ({case_label}): $A={amplitude},\ \gamma={gamma},\ "
        rf"\Omega={omega}$, {N_CYCLES} periods "
        rf"($t_{{\mathrm{{end}}}}={t_end:.3f}$)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


def main():
    for amplitude, gamma, omega, case_label in PARAMETER_CASES:
        plot_one_case(amplitude, gamma, omega, case_label)


if __name__ == "__main__":
    main()
