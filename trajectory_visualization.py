import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from heavy_ball_system import DEFAULT_OMEGA, potential, rhs
from visualizations import plot_equilibrium_contours

WARMUP_FRACTION = 0.3
METHOD = "RK45"
RTOL = 1e-8

DEFAULT_INITIAL_STATE = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
DEFAULT_PERTURBATIONS = (
    np.array([1e-3, 0.0, 0.0, 0.0]),
    np.array([0.0, 1e-3, 0.0, 0.0]),
)
TRAJECTORY_ZORDER = 6


def integrate_dense(
    initial_state,
    amplitude: float,
    gamma: float,
    omega: float,
    t_end: float,
    n_points: int = 4000,
):
    """Integrate from t=0 to t_end with dense output for smooth (x, y) plots."""
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        (0.0, t_end),
        initial_state,
        t_eval=t_eval,
        method=METHOD,
        rtol=RTOL,
    )
    return sol


def strobed_energy(
    initial_state,
    amplitude: float,
    gamma: float,
    omega: float,
    n_cycles: int,
):
    """
    Energy at strobe times k*T (same sampling as poincare_map.get_poincare_data).
    Returns energy array after warm-up trim, indexed as cycle number after trim.
    """
    T = 2 * np.pi / omega
    t_span = (0.0, n_cycles * T)
    t_eval = np.arange(0.0, n_cycles * T, T)
    sol = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        t_span,
        initial_state,
        t_eval=t_eval,
        method=METHOD,
        rtol=RTOL,
    )
    v_mag2 = sol.y[2] ** 2 + sol.y[3] ** 2
    pot = potential(sol.y[0], sol.y[1])
    energy = 0.5 * v_mag2 + pot
    warm_up = int(WARMUP_FRACTION * len(energy))
    return energy[warm_up:]


def plot_trajectory_dynamics(
    amplitude: float,
    gamma: float,
    *,
    omega: float = DEFAULT_OMEGA,
    initial_state=None,
    perturbations=None,
    n_cycles: int = 600,
    n_dense_points: int = 4000,
    overlay_equilibria: bool = False,
):
    """
    For one (amplitude, gamma) pair: plot three slightly perturbed (x, y) trajectories
    and strobed energy for the reference trajectory. All other settings default to
    shared values (same omega, ICs, integration length, …).

    If overlay_equilibria is True, draw the trajectory panel on top of the
    equilibrium contour / classification plot from visualizations.plot_equilibrium_contours.
    """
    if initial_state is None:
        initial_state = DEFAULT_INITIAL_STATE
    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS

    base = np.asarray(initial_state, dtype=float)
    d1, d2 = perturbations
    states = [base, base + np.asarray(d1), base + np.asarray(d2)]

    T = 2 * np.pi / omega
    t_end = n_cycles * T

    colors = ("tab:blue", "tab:orange", "tab:green")
    labels = ("reference", "perturbation 1", "perturbation 2")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6) if overlay_equilibria else (12, 5))
    ax_xy, ax_e = axes[0], axes[1]

    if overlay_equilibria:
        plot_equilibrium_contours(gamma=gamma, ax=ax_xy)

    for state0, color, label in zip(states, colors, labels):
        out = integrate_dense(
            state0, amplitude, gamma, omega, t_end, n_points=n_dense_points
        )
        ax_xy.plot(
            out.y[0],
            out.y[1],
            color=color,
            linewidth=1.0,
            label=label,
            zorder=TRAJECTORY_ZORDER,
        )

    energy_trimmed = strobed_energy(
        base, amplitude, gamma, omega, n_cycles=n_cycles
    )
    ax_e.plot(energy_trimmed, color="red", linewidth=0.8)
    ax_e.set_title("Energy at sampled periods")
    ax_e.set_xlabel("Cycle (n)")
    ax_e.set_ylabel("Energy V")

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("$x$")
    ax_xy.set_ylabel("$y$")
    if overlay_equilibria:
        ax_xy.set_title(r"Trajectories on $\partial_x f=0$, $\partial_y f=0$, equilibria")
    else:
        ax_xy.set_title(r"$(x,y)$ trajectories")

    fig.suptitle(
        rf"Trajectories and energy: $A={amplitude},\ \gamma={gamma},\ \Omega={omega}$",
        fontsize=14,
    )
    ax_xy.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Same shared defaults (omega, ICs, n_cycles, …); only amplitude and gamma change.
    # plot_trajectory_dynamics(amplitude=1.0, gamma=10.0)
    plot_trajectory_dynamics(amplitude=0.5, gamma=0.1, omega=1, overlay_equilibria=True)
    plot_trajectory_dynamics(amplitude=1, gamma=3.0, omega=1)
    # plot_trajectory_dynamics(amplitude=0.0, gamma=8.0)

    # Example with trajectories overlaid on equilibrium contours (uncomment to run):
    # plot_trajectory_dynamics(1.0, 10.0, overlay_equilibria=True)
