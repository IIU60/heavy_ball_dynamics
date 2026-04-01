import numpy as np

from find_equilibria import cluster_points, refine_roots, scan_equilibrium_grid
from heavy_ball_system import potential


def hessian(x, y):
    """
    Hessian of
    f(x, y) = 1/4 (x^2 + y^2 - 1)^2 + 0.1 (cos(2x) + cos(2y)).
    """
    f_xx = (3.0 * x**2 + y**2 - 1.0) - 0.4 * np.cos(2.0 * x)
    f_yy = (x**2 + 3.0 * y**2 - 1.0) - 0.4 * np.cos(2.0 * y)
    f_xy = 2.0 * x * y

    return np.array([[f_xx, f_xy], [f_xy, f_yy]], dtype=float)


def jacobian(x, y, gamma: float = 1.0):
    """
    Jacobian of the heavy-ball system at an equilibrium with vx = vy = 0.

    This follows the supplied structure, with the damping coefficient denoted
    by gamma in the original system.
    """
    hess = hessian(x, y)
    return np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-hess[0, 0], -hess[0, 1], -gamma, 0.0],
            [-hess[1, 0], -hess[1, 1], 0.0, -gamma],
        ],
        dtype=float,
    )


def classify_hessian(hess, tol: float = 1e-8):
    """Classify an equilibrium as a minimum, maximum, saddle, or degenerate."""
    eigenvalues = np.linalg.eigvalsh(hess)

    if np.all(eigenvalues > tol):
        label = "minimum"
    elif np.all(eigenvalues < -tol):
        label = "maximum"
    elif np.any(eigenvalues > tol) and np.any(eigenvalues < -tol):
        label = "saddle"
    else:
        label = "degenerate"

    return label, eigenvalues


def classify_jacobian(jac, tol: float = 1e-8):
    """Classify linear stability from Jacobian eigenvalues."""
    eigenvalues = np.linalg.eigvals(jac)
    real_parts = np.real(eigenvalues)

    if np.all(real_parts < -tol):
        label = "stable"
    elif np.any(real_parts > tol):
        label = "unstable"
    else:
        label = "marginal"

    return label, eigenvalues


def find_all_equilibria(
    scan_tol: float = 0.05,
    cluster_distance: float = 0.1,
    num_points: int = 161,
):
    """Run the scan-cluster-refine pipeline."""
    candidates = scan_equilibrium_grid(num_points=num_points, tol=scan_tol)
    guesses = cluster_points(candidates, distance_threshold=cluster_distance)
    roots = refine_roots(guesses)
    return candidates, guesses, roots


def classify_equilibria(
    gamma: float = 1.0,
    scan_tol: float = 0.05,
    cluster_distance: float = 0.1,
    num_points: int = 161,
    value_tol: float = 1e-8,
):
    """Compute equilibrium classifications and identify global minima."""
    _, _, roots = find_all_equilibria(
        scan_tol=scan_tol,
        cluster_distance=cluster_distance,
        num_points=num_points,
    )

    records = []
    for root in roots:
        x, y = root[:2]
        hess = hessian(x, y)
        hessian_type, hessian_eigs = classify_hessian(hess)
        jac = jacobian(x, y, gamma=gamma)
        stability, jacobian_eigs = classify_jacobian(jac)
        f_value = float(potential(x, y))

        records.append(
            {
                "x": float(x),
                "y": float(y),
                "f": f_value,
                "hessian_type": hessian_type,
                "hessian_eigs": hessian_eigs,
                "stability": stability,
                "jacobian_eigs": jacobian_eigs,
            }
        )

    minimum_values = [record["f"] for record in records if record["hessian_type"] == "minimum"]
    global_min_value = min(minimum_values) if minimum_values else None

    for record in records:
        if record["hessian_type"] != "minimum":
            record["minimum_scope"] = "n/a"
        elif global_min_value is not None and abs(record["f"] - global_min_value) <= value_tol:
            record["minimum_scope"] = "global"
        else:
            record["minimum_scope"] = "local"

    records.sort(key=lambda item: (item["f"], item["x"], item["y"]))
    return records, global_min_value


def format_complex(z, digits: int = 6):
    """Format a complex number for table output."""
    real = np.real(z)
    imag = np.imag(z)

    if abs(imag) < 1e-12:
        return f"{real:.{digits}f}"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.{digits}f}{sign}{abs(imag):.{digits}f}i"


def print_plain_table(records):
    """Print a readable plain-text summary table."""
    headers = [
        "x",
        "y",
        "f(x,y)",
        "type",
        "min class",
        "stability",
        "H eigvals",
        "J eigvals",
    ]

    rows = []
    for record in records:
        rows.append(
            [
                f"{record['x']:.8f}",
                f"{record['y']:.8f}",
                f"{record['f']:.8f}",
                record["hessian_type"],
                record["minimum_scope"],
                record["stability"],
                ", ".join(format_complex(z) for z in record["hessian_eigs"]),
                ", ".join(format_complex(z) for z in record["jacobian_eigs"]),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for i, item in enumerate(row):
            widths[i] = max(widths[i], len(item))

    header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(item.ljust(widths[i]) for i, item in enumerate(row)))


def print_latex_equilibria_table(records):
    """Print a LaTeX table for all equilibria."""
    print("\nLaTeX table for all equilibria:\n")
    print(r"\begin{tabular}{rrrrlll}")
    print(r"\hline")
    print(r"$x$ & $y$ & $f(x,y)$ & Type & Minimum class & Stability & Jacobian eigenvalues \\")
    print(r"\hline")
    for record in records:
        eigvals = ", ".join(format_complex(z) for z in record["jacobian_eigs"])
        print(
            f"{record['x']:.8f} & "
            f"{record['y']:.8f} & "
            f"{record['f']:.8f} & "
            f"{record['hessian_type']} & "
            f"{record['minimum_scope']} & "
            f"{record['stability']} & "
            f"{eigvals} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")


def print_latex_minima_table(records, global_min_value):
    """Print a LaTeX table restricted to minima."""
    minima = [record for record in records if record["hessian_type"] == "minimum"]

    print("\nLaTeX table for minima only:\n")
    print(r"\begin{tabular}{rrrrll}")
    print(r"\hline")
    print(r"$x$ & $y$ & $f(x,y)$ & Minimum class & Stability & Hessian eigenvalues \\")
    print(r"\hline")
    for record in minima:
        eigvals = ", ".join(format_complex(z) for z in record["hessian_eigs"])
        print(
            f"{record['x']:.8f} & "
            f"{record['y']:.8f} & "
            f"{record['f']:.8f} & "
            f"{record['minimum_scope']} & "
            f"{record['stability']} & "
            f"{eigvals} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")

    if global_min_value is not None:
        print(f"\nGlobal minimum value: f_min = {global_min_value:.10f}")


if __name__ == "__main__":
    records, global_min_value = classify_equilibria(gamma=1.0)

    print("\nEquilibrium classification summary:\n")
    print_plain_table(records)
    print_latex_equilibria_table(records)
    print_latex_minima_table(records, global_min_value)
