import matplotlib.pyplot as plt
import numpy as np

from classify_equilibria import classify_equilibria
from find_equilibria import cluster_points, refine_roots, scan_equilibrium_grid
from heavy_ball_system import dfdx, dfdy, potential, rhs


def make_xy_grid(
    xmin: float = -2.0,
    xmax: float = 2.0,
    ymin: float = -2.0,
    ymax: float = 2.0,
    num_points: int = 201,
):
    """Create a uniform 2D grid in the x-y plane."""
    x_values = np.linspace(xmin, xmax, num_points)
    y_values = np.linspace(ymin, ymax, num_points)
    return np.meshgrid(x_values, y_values, indexing="ij")


def find_equilibrium_roots(
    scan_tol: float = 0.05,
    cluster_distance: float = 0.1,
    num_points: int = 161,
):
    """Run the full scan-cluster-refine pipeline and return equilibrium roots."""
    candidates = scan_equilibrium_grid(num_points=num_points, tol=scan_tol)
    guesses = cluster_points(candidates, distance_threshold=cluster_distance)
    roots = refine_roots(guesses)
    return candidates, guesses, roots


def plot_equilibrium_contours(
    xmin: float = -2.0,
    xmax: float = 2.0,
    ymin: float = -2.0,
    ymax: float = 2.0,
    num_points: int = 401,
    scan_tol: float = 0.05,
    cluster_distance: float = 0.1,
    show_candidates: bool = False,
    gamma: float = 1.0,
):
    """
    Plot the zero contours of dfdx and dfdy and overlay classified equilibria.
    """
    x_grid, y_grid = make_xy_grid(xmin, xmax, ymin, ymax, num_points=num_points)
    fx_grid = dfdx(x_grid, y_grid)
    fy_grid = dfdy(x_grid, y_grid)

    candidates, guesses, roots = find_equilibrium_roots(
        scan_tol=scan_tol,
        cluster_distance=cluster_distance,
        num_points=max(161, num_points // 2),
    )
    records, _ = classify_equilibria(
        gamma=gamma,
        scan_tol=scan_tol,
        cluster_distance=cluster_distance,
        num_points=max(161, num_points // 2),
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
    xmin: float = -2.0,
    xmax: float = 2.0,
    ymin: float = -2.0,
    ymax: float = 2.0,
    num_points: int = 201,
):
    """Plot the potential f(x, y)."""
    x_grid, y_grid = make_xy_grid(xmin, xmax, ymin, ymax, num_points=num_points)
    z_grid = potential(x_grid, y_grid)
    return plot_surface(x_grid, y_grid, z_grid, "Potential surface f(x, y)", "f")


def plot_gradient_surfaces(
    xmin: float = -2.0,
    xmax: float = 2.0,
    ymin: float = -2.0,
    ymax: float = 2.0,
    num_points: int = 201,
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
    t: float = 0.0,
    gamma: float = 0.2,
    amplitude: float = 0.0,
    omega: float = 1.0,
    vx: float = 0.0,
    vy: float = 0.0,
    xmin: float = -2.0,
    xmax: float = 2.0,
    ymin: float = -2.0,
    ymax: float = 2.0,
    num_points: int = 201,
):
    """
    Evaluate the heavy-ball RHS on the x-y plane for fixed t, vx, and vy.
    """
    x_grid, y_grid = make_xy_grid(xmin, xmax, ymin, ymax, num_points=num_points)

    x_dot = np.full_like(x_grid, vx, dtype=float)
    y_dot = np.full_like(y_grid, vy, dtype=float)
    vx_dot = -gamma * vx - dfdx(x_grid, y_grid) - amplitude * np.sin(omega * t)
    vy_dot = -gamma * vy - dfdy(x_grid, y_grid) - amplitude * np.sin(omega * t)

    return x_grid, y_grid, {
        "x_dot": x_dot,
        "y_dot": y_dot,
        "vx_dot": vx_dot,
        "vy_dot": vy_dot,
    }


def plot_rhs_surfaces(
    t: float = 0.0,
    gamma: float = 0.2,
    amplitude: float = 0.0,
    omega: float = 1.0,
    vx: float = 0.0,
    vy: float = 0.0,
    xmin: float = -2.0,
    xmax: float = 2.0,
    ymin: float = -2.0,
    ymax: float = 2.0,
    num_points: int = 201,
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
    vx: float = 0.0,
    vy: float = 0.0,
    t: float = 0.0,
    gamma: float = 0.2,
    amplitude: float = 0.0,
    omega: float = 1.0,
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
