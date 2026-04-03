import matplotlib.pyplot as plt
import numpy as np

from classify_equilibria import classify_equilibria
from find_equilibria import find_equilibria_pipeline
from heavy_ball_system import (
    CONTOUR_NUM_POINTS,
    DEFAULT_AMPLITUDE,
    DEFAULT_GAMMA,
    DEFAULT_OMEGA,
    DEFAULT_T,
    DEFAULT_VX,
    DEFAULT_VY,
    SCAN_NUM_POINTS,
    SCAN_TOL,
    CLUSTER_DISTANCE,
    SURFACE_NUM_POINTS,
    XMAX,
    XMIN,
    YMAX,
    YMIN,
    dfdx,
    dfdy,
    potential,
    rhs,
    rhs_components_on_xy_grid,
)


def make_xy_grid(
    xmin: float = XMIN,
    xmax: float = XMAX,
    ymin: float = YMIN,
    ymax: float = YMAX,
    num_points: int = SURFACE_NUM_POINTS,
):
    """Create a uniform 2D grid in the x-y plane."""
    x_values = np.linspace(xmin, xmax, num_points)
    y_values = np.linspace(ymin, ymax, num_points)
    return np.meshgrid(x_values, y_values, indexing="ij")


def plot_equilibrium_contours(
    xmin: float = XMIN,
    xmax: float = XMAX,
    ymin: float = YMIN,
    ymax: float = YMAX,
    num_points: int = CONTOUR_NUM_POINTS,
    scan_num_points: int = SCAN_NUM_POINTS,
    scan_tol: float = SCAN_TOL,
    cluster_distance: float = CLUSTER_DISTANCE,
    show_candidates: bool = False,
    gamma: float = DEFAULT_GAMMA,
):
    """
    Plot the zero contours of dfdx and dfdy and overlay classified equilibria.
    """
    x_grid, y_grid = make_xy_grid(xmin, xmax, ymin, ymax, num_points=num_points)
    fx_grid = dfdx(x_grid, y_grid)
    fy_grid = dfdy(x_grid, y_grid)

    candidates, guesses, roots = find_equilibria_pipeline(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        num_points=scan_num_points,
        scan_tol=scan_tol,
        cluster_distance=cluster_distance,
    )
    records, _ = classify_equilibria(
        roots=roots,
        gamma=gamma,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contour(x_grid, y_grid, fx_grid, levels=[0.0], colors="tab:blue", linewidths=2)
    ax.contour(x_grid, y_grid, fy_grid, levels=[0.0], colors="tab:orange", linewidths=2)

    if show_candidates and len(candidates) > 0:
        ax.scatter(
            candidates[:, 0],
            candidates[:, 1],
            s=10,
            c="0.75",
            alpha=0.6,
            label="grid candidates",
        )

    if len(guesses) > 0:
        ax.scatter(
            guesses[:, 0],
            guesses[:, 1],
            s=50,
            c="tab:green",
            marker="x",
            label="clustered guesses",
        )

    if len(roots) > 0:
        style_map = {
            ("minimum", "stable"): ("tab:green", "o"),
            ("minimum", "unstable"): ("yellowgreen", "o"),
            ("minimum", "marginal"): ("olive", "o"),
            ("maximum", "stable"): ("tab:purple", "^"),
            ("maximum", "unstable"): ("mediumpurple", "^"),
            ("maximum", "marginal"): ("indigo", "^"),
            ("saddle", "stable"): ("tab:red", "s"),
            ("saddle", "unstable"): ("tomato", "s"),
            ("saddle", "marginal"): ("firebrick", "s"),
            ("degenerate", "stable"): ("tab:gray", "D"),
            ("degenerate", "unstable"): ("dimgray", "D"),
            ("degenerate", "marginal"): ("black", "D"),
        }

        plotted_labels = set()
        for record in records:
            color, marker = style_map.get(
                (record["hessian_type"], record["stability"]),
                ("black", "o"),
            )
            label = f"{record['hessian_type']}, {record['stability']}"
            ax.scatter(
                record["x"],
                record["y"],
                s=90,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.6,
                label=label if label not in plotted_labels else None,
                zorder=5,
            )
            plotted_labels.add(label)
            text_label = record["hessian_type"]
            if record["hessian_type"] == "minimum":
                text_label = f"{text_label} ({record['minimum_scope']})"
            ax.annotate(
                text_label,
                (record["x"], record["y"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75, "ec": "none"},
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Zero contours of $\partial_x f$, $\partial_y f$, and classified equilibria")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


def plot_surface(x_grid, y_grid, z_grid, title: str, zlabel: str):
    """Plot a single 3D surface over the x-y plane."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.1)
    return fig, ax


def plot_potential_surface(
    xmin: float = XMIN,
    xmax: float = XMAX,
    ymin: float = YMIN,
    ymax: float = YMAX,
    num_points: int = SURFACE_NUM_POINTS,
):
    """Plot the potential f(x, y)."""
    x_grid, y_grid = make_xy_grid(xmin, xmax, ymin, ymax, num_points=num_points)
    z_grid = potential(x_grid, y_grid)
    return plot_surface(x_grid, y_grid, z_grid, "Potential surface f(x, y)", "f")


def plot_gradient_surfaces(
    xmin: float = XMIN,
    xmax: float = XMAX,
    ymin: float = YMIN,
    ymax: float = YMAX,
    num_points: int = SURFACE_NUM_POINTS,
):
    """Plot 3D surfaces for dfdx and dfdy."""
    x_grid, y_grid = make_xy_grid(xmin, xmax, ymin, ymax, num_points=num_points)
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    z1 = dfdx(x_grid, y_grid)
    surf1 = ax1.plot_surface(x_grid, y_grid, z1, cmap="coolwarm", linewidth=0)
    ax1.set_title(r"$\partial_x f(x,y)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("dfdx")
    fig.colorbar(surf1, ax=ax1, shrink=0.65, pad=0.08)

    ax2 = fig.add_subplot(122, projection="3d")
    z2 = dfdy(x_grid, y_grid)
    surf2 = ax2.plot_surface(x_grid, y_grid, z2, cmap="plasma", linewidth=0)
    ax2.set_title(r"$\partial_y f(x,y)$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("dfdy")
    fig.colorbar(surf2, ax=ax2, shrink=0.65, pad=0.08)

    return fig, (ax1, ax2)


def compute_rhs_surfaces(
    t: float = DEFAULT_T,
    gamma: float = DEFAULT_GAMMA,
    amplitude: float = DEFAULT_AMPLITUDE,
    omega: float = DEFAULT_OMEGA,
    vx: float = DEFAULT_VX,
    vy: float = DEFAULT_VY,
    xmin: float = XMIN,
    xmax: float = XMAX,
    ymin: float = YMIN,
    ymax: float = YMAX,
    num_points: int = SURFACE_NUM_POINTS,
):
    """
    Evaluate the heavy-ball RHS on the x-y plane for fixed t, vx, and vy.
    """
    x_grid, y_grid = make_xy_grid(xmin, xmax, ymin, ymax, num_points=num_points)
    surfaces = rhs_components_on_xy_grid(
        x_grid,
        y_grid,
        t=t,
        gamma=gamma,
        amplitude=amplitude,
        omega=omega,
        vx=vx,
        vy=vy,
    )
    return x_grid, y_grid, surfaces


def plot_rhs_surfaces(
    t: float = DEFAULT_T,
    gamma: float = DEFAULT_GAMMA,
    amplitude: float = DEFAULT_AMPLITUDE,
    omega: float = DEFAULT_OMEGA,
    vx: float = DEFAULT_VX,
    vy: float = DEFAULT_VY,
    xmin: float = XMIN,
    xmax: float = XMAX,
    ymin: float = YMIN,
    ymax: float = YMAX,
    num_points: int = SURFACE_NUM_POINTS,
):
    """Plot 3D surfaces for the four components of the heavy-ball RHS."""
    x_grid, y_grid, surfaces = compute_rhs_surfaces(
        t=t,
        gamma=gamma,
        amplitude=amplitude,
        omega=omega,
        vx=vx,
        vy=vy,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        num_points=num_points,
    )

    fig = plt.figure(figsize=(14, 10))
    component_names = ["x_dot", "y_dot", "vx_dot", "vy_dot"]

    for index, name in enumerate(component_names, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        surface = ax.plot_surface(
            x_grid,
            y_grid,
            surfaces[name],
            cmap="viridis",
            linewidth=0,
        )
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(name)
        fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.08)

    fig.suptitle(
        "Heavy-ball RHS surfaces "
        f"(t={t}, gamma={gamma}, A={amplitude}, omega={omega}, vx={vx}, vy={vy})"
    )
    fig.tight_layout()
    return fig


def evaluate_rhs_at_point(
    x: float,
    y: float,
    vx: float = DEFAULT_VX,
    vy: float = DEFAULT_VY,
    t: float = DEFAULT_T,
    gamma: float = DEFAULT_GAMMA,
    amplitude: float = DEFAULT_AMPLITUDE,
    omega: float = DEFAULT_OMEGA,
):
    """Convenience wrapper around rhs for a single state."""
    return rhs(
        t=t,
        state=np.array([x, y, vx, vy], dtype=float),
        gamma=gamma,
        amplitude=amplitude,
        omega=omega,
    )


if __name__ == "__main__":
    plot_equilibrium_contours(show_candidates=True)
    plot_potential_surface()
    plot_gradient_surfaces()
    plot_rhs_surfaces()
    if "agg" not in plt.get_backend().lower():
        plt.show()
