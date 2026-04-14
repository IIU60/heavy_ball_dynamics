import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import label, minimum_filter

from heavy_ball_system import (
    DEFAULT_OMEGA,
    XMAX,
    XMIN,
    YMAX,
    YMIN,
    hessian,
    potential,
    rhs,
)

# Match poincare_map.get_poincare_data (integrator tolerances, IC, n_cycles default).
METHOD = "RK45"
RTOL = 1e-10
ATOL = 1e-12
N_CYCLES_DEFAULT = 600

# [x0, y0, v_x0, v_y0] — same as poincare_map.py
DEFAULT_INITIAL_STATE = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
DEFAULT_PERTURBATIONS = (
    np.array([1e-3, 0.0, 0.0, 0.0]),
    np.array([-1e-3, 0.0 , 0.0, 0.0]),
)
TRAJECTORY_ZORDER = 6
TRAJECTORY_SCATTER_SIZE = 4
TRAJECTORY_SCATTER_ALPHA = 0.65
POTENTIAL_CMAP = "viridis"
MINIMUM_LABEL_ZORDER = 7


def _find_labeled_minima(
    x_1d,
    y_1d,
    Z,
    rtol=1e-5,
    atol=1e-10,
    hessian_eig_tol: float = 1e-8,
):
    """
    Local minima of f on the grid: 3x3 neighborhood minima, merged by connectivity,
    then filtered to points where the analytic Hessian is positive definite.
    """
    min_nb = minimum_filter(Z, size=3, mode="nearest")
    is_candidate = np.isclose(Z, min_nb, rtol=rtol, atol=atol)
    is_candidate[0, :] = is_candidate[-1, :] = False
    is_candidate[:, 0] = is_candidate[:, -1] = False

    structure = np.ones((3, 3), dtype=int)
    labeled, nfeat = label(is_candidate, structure=structure)

    Xg, Yg = np.meshgrid(x_1d, y_1d, indexing="ij")
    minima_xy = []
    for k in range(1, nfeat + 1):
        mask = labeled == k
        xc = float(Xg[mask].mean())
        yc = float(Yg[mask].mean())
        H = hessian(xc, yc)
        ev = np.linalg.eigvalsh(H)
        if ev[0] > hessian_eig_tol:
            minima_xy.append((xc, yc))
    return minima_xy


def draw_potential_contour_background(
    ax,
    xmin: float = XMIN,
    xmax: float = XMAX,
    ymin: float = YMIN,
    ymax: float = YMAX,
    grid_points: int = 301,
    contour_levels: int = 50,
):
    """Smooth filled contours of f(x,y) with numbered labels at local minima."""
    x_1d = np.linspace(xmin, xmax, grid_points)
    y_1d = np.linspace(ymin, ymax, grid_points)
    Xg, Yg = np.meshgrid(x_1d, y_1d, indexing="ij")
    Z = potential(Xg, Yg)

    levels = np.linspace(Z.min(), Z.max(), contour_levels)
    cf = ax.contourf(
        Xg,
        Yg,
        Z,
        levels=levels,
        cmap=POTENTIAL_CMAP,
        alpha=0.9,
        zorder=0,
    )
    plt.colorbar(cf, ax=ax, shrink=0.72, label=r"$f(x,y)$")

    minima = _find_labeled_minima(x_1d, y_1d, Z)
    for i, (xm, ym) in enumerate(minima, start=1):
        ax.plot(
            xm,
            ym,
            marker="o",
            color="white",
            markeredgecolor="black",
            markersize=7,
            linestyle="none",
            zorder=MINIMUM_LABEL_ZORDER,
        )
        ax.annotate(
            f"min {i}",
            (xm, ym),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color="white",
            fontweight="bold",
            path_effects=[
                patheffects.withStroke(linewidth=2, foreground="black")
            ],
            zorder=MINIMUM_LABEL_ZORDER + 1,
        )


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
        atol=ATOL,
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
        atol=ATOL,
    )
    v_mag2 = sol.y[2] ** 2 + sol.y[3] ** 2
    pot = potential(sol.y[0], sol.y[1])
    energy = 0.5 * v_mag2 + pot
    return energy


def plot_trajectory_dynamics(
    amplitude: float,
    gamma: float,
    *,
    omega: float = DEFAULT_OMEGA,
    initial_state=None,
    perturbations=None,
    n_cycles: int = N_CYCLES_DEFAULT,
    n_dense_points: int = 20000,
    overlay_potential: bool = False,
):
    """
    For one (amplitude, gamma) pair: plot three slightly perturbed (x, y) trajectories
    and strobed energy for the reference trajectory. Defaults match poincare_map.py
    (IC [1,1,0,0], n_cycles=600, RK45 rtol/atol).

    If overlay_potential is True, draw a smooth filled contour map of f(x, y) behind
    the paths and label detected local minima (no equilibrium / zero-gradient plots).
    """
    if initial_state is None:
        initial_state = DEFAULT_INITIAL_STATE
    if perturbations is None:
        perturbations = DEFAULT_PERTURBATIONS

    base = np.asarray(initial_state, dtype=float)
    d1, d2 = perturbations
    states = [base]#, base + np.asarray(d1), base + np.asarray(d2)]

    T = 2 * np.pi / omega
    t_end = n_cycles * T

    colors = ("tab:blue", "tab:orange", "tab:green")
    labels = ("reference", "perturbation 1", "perturbation 2")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6) if overlay_potential else (12, 5))
    ax_xy, ax_e = axes[0], axes[1]

    if overlay_potential:
        draw_potential_contour_background(ax_xy)

    for state0, color, label in zip(states, colors, labels):
        out = integrate_dense(
            state0, amplitude, gamma, omega, t_end, n_points=n_dense_points
        )
        ax_xy.scatter(
            out.y[0],
            out.y[1],
            s=TRAJECTORY_SCATTER_SIZE,
            c=color,
            alpha=TRAJECTORY_SCATTER_ALPHA,
            linewidths=0,
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
    if overlay_potential:
        ax_xy.set_title(r"Trajectories on $f(x,y)$")
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
    # (amplitude, gamma, omega) sets from poincare_map.py — shared IC and n_cycles via defaults.
    # Convergence
    plot_trajectory_dynamics(amplitude=0.1, gamma=0.5, omega=1.0, overlay_potential=True)
    # Periodic locking
    plot_trajectory_dynamics(amplitude=0.3, gamma=0.1, omega=1.0, overlay_potential=True)
    # Chaotic behavior
    plot_trajectory_dynamics(amplitude=0.5, gamma=0.5, omega=1.0, overlay_potential=True)

    # Optional: trajectories on smooth f(x,y) with minima labeled
    # plot_trajectory_dynamics(0.3, 0.1, omega=1.0, overlay_potential=True)
