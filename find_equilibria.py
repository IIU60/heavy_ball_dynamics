import numpy as np
from scipy.optimize import fsolve

from heavy_ball_system import dfdx, dfdy


def scan_equilibrium_grid(
    xmin: float = -2.0,
    xmax: float = 2.0,
    ymin: float = -2.0,
    ymax: float = 2.0,
    num_points: int = 161,
    tol: float = 0.05,
):
    """
    Scan a uniform grid and return points where both derivatives are small.

    Each result is returned as a row of
    [x, y, dfdx_value, dfdy_value].
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    x_values = np.linspace(xmin, xmax, num_points)
    y_values = np.linspace(ymin, ymax, num_points)
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="ij")

    fx_grid = dfdx(x_grid, y_grid)
    fy_grid = dfdy(x_grid, y_grid)

    mask = (np.abs(fx_grid) < tol) & (np.abs(fy_grid) < tol)

    print(np.sum(mask))

    return np.column_stack(
        (
            x_grid[mask],
            y_grid[mask],
            fx_grid[mask],
            fy_grid[mask],
        )
    )


def cluster_points(points, distance_threshold: float = 0.1):
    """
    Cluster nearby 2D points using a simple distance threshold.

    The input can be an array with shape (N, 2) or (N, M) where the first
    two columns are interpreted as x and y. One representative point, given
    by the cluster centroid, is returned for each cluster.
    """
    points = np.asarray(points, dtype=float)

    if points.size == 0:
        return np.empty((0, 2))

    xy = points[:, :2]
    n_points = len(xy)
    assigned = np.zeros(n_points, dtype=bool)
    clusters = []

    for i in range(n_points):
        if assigned[i]:
            continue

        cluster_indices = [i]
        assigned[i] = True

        changed = True
        while changed:
            changed = False
            cluster_members = xy[cluster_indices]

            for j in range(n_points):
                if assigned[j]:
                    continue

                distances = np.linalg.norm(cluster_members - xy[j], axis=1)
                if np.any(distances <= distance_threshold):
                    cluster_indices.append(j)
                    assigned[j] = True
                    changed = True

        clusters.append(xy[cluster_indices].mean(axis=0))

    return np.array(clusters)


def equilibrium_residual(point):
    """Return F(x, y) = (dfdx, dfdy)."""
    x, y = point
    return np.array([dfdx(x, y), dfdy(x, y)], dtype=float)


def refine_roots(initial_guesses, root_tol: float = 1e-10, duplicate_tol: float = 1e-6):
    """
    Refine clustered initial guesses into true roots using scipy.optimize.fsolve.

    Returns an array with rows [x, y, dfdx, dfdy].
    """
    initial_guesses = np.asarray(initial_guesses, dtype=float)

    if initial_guesses.size == 0:
        return np.empty((0, 4))

    roots = []

    for guess in initial_guesses:
        root, info, ier, _ = fsolve(
            equilibrium_residual,
            x0=guess,
            full_output=True,
            xtol=root_tol,
        )

        residual = equilibrium_residual(root)
        if ier == 1 and np.linalg.norm(residual, ord=np.inf) < root_tol:
            roots.append(np.array([root[0], root[1], residual[0], residual[1]]))

    if not roots:
        return np.empty((0, 4))

    unique_roots = []
    for root_data in roots:
        root_xy = root_data[:2]
        is_duplicate = any(
            np.linalg.norm(root_xy - saved_root[:2]) <= duplicate_tol
            for saved_root in unique_roots
        )
        if not is_duplicate:
            unique_roots.append(root_data)

    return np.array(unique_roots)


if __name__ == "__main__":
    candidates = scan_equilibrium_grid()
    clustered_guesses = cluster_points(candidates, distance_threshold=0.1)
    roots = refine_roots(clustered_guesses)

    print(f"Found {len(candidates)} candidate grid points.")
    print(f"Reduced to {len(clustered_guesses)} clustered initial guesses.")
    print(f"Refined to {len(roots)} distinct roots.")

    for x, y, fx, fy in roots:
        print(f"x={x: .8f}, y={y: .8f}, dfdx={fx: .2e}, dfdy={fy: .2e}")
