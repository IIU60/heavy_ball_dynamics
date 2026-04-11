import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from heavy_ball_system import potential, rhs

WARMUP_FRACTION = 0.3
METHOD = "RK45"
RTOL = 1e-8


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


def plot_all_scenarios(scenarios):
    n = len(scenarios)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = np.array([axes])

    colors = ("tab:blue", "tab:orange", "tab:green")
    labels = ("reference", "perturbation 1", "perturbation 2")

    for row, scenario in enumerate(scenarios):
        ax_xy, ax_e = axes[row]
        base = np.asarray(scenario["initial_state"], dtype=float)
        d1, d2 = scenario["perturbations"]
        states = [base, base + np.asarray(d1), base + np.asarray(d2)]

        amplitude = scenario["amplitude"]
        gamma = scenario["gamma"]
        omega = scenario["omega"]
        t_end = scenario["t_end"]
        n_dense = scenario.get("n_dense_points", 4000)
        n_cycles = scenario["n_cycles"]

        for state0, color, label in zip(states, colors, labels):
            out = integrate_dense(
                state0, amplitude, gamma, omega, t_end, n_points=n_dense
            )
            ax_xy.plot(out.y[0], out.y[1], color=color, linewidth=0.9, label=label)

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
        ax_xy.set_title(
            f"{scenario['label']}: "
            rf"$A={amplitude},\ \gamma={gamma},\ \Omega={omega}$"
        )
        ax_xy.legend(loc="best", fontsize=8)

    fig.suptitle("Trajectory comparison: three perturbed paths per regime", fontsize=14)
    plt.tight_layout()
    plt.show()


# Base perturbations (small offsets in x and y)
_DELTA_XY = (
    np.array([1e-3, 0.0, 0.0, 0.0]),
    np.array([0.0, 1e-3, 0.0, 0.0]),
)

T_PERIODIC = 2 * np.pi / 0.5
T_CHAOTIC = 2 * np.pi / 1.0
T_STABLE = 2 * np.pi / 1.0

SCENARIOS = [
    {
        "label": "Periodic",
        "amplitude": 1.0,
        "gamma": 10.0,
        "omega": 0.5,
        "initial_state": [0.0, 0.1, 0.0, 0.1],
        "perturbations": _DELTA_XY,
        "n_cycles": 600,
        "t_end": 600 * T_PERIODIC,
        "n_dense_points": 4000,
    },
    {
        "label": "Chaotic",
        "amplitude": 0.1,
        "gamma": 3.0,
        "omega": 1.0,
        "initial_state": [0.0, 0.1, 0.0, 0.1],
        "perturbations": _DELTA_XY,
        "n_cycles": 600,
        "t_end": 600 * T_CHAOTIC,
        "n_dense_points": 4000,
    },
    {
        "label": "Stable (damped)",
        "amplitude": 0.0,
        "gamma": 8.0,
        "omega": 1.0,
        "initial_state": [0.95, 0.0, 0.0, 0.0],
        "perturbations": _DELTA_XY,
        "n_cycles": 200,
        "t_end": 200 * T_STABLE,
        "n_dense_points": 2500,
    },
]


if __name__ == "__main__":
    plot_all_scenarios(SCENARIOS)
