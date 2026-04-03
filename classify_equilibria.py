import numpy as np

from find_equilibria import find_equilibria_pipeline
from heavy_ball_system import (
    CLASSIFICATION_TOL,
    CLUSTER_DISTANCE,
    DEFAULT_GAMMA,
    GLOBAL_MIN_TOL,
    SCAN_NUM_POINTS,
    SCAN_TOL,
    hessian,
    jacobian,
    potential,
)


def classify_hessian(hess, tol: float = CLASSIFICATION_TOL):
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


def classify_jacobian(jac, tol: float = CLASSIFICATION_TOL):
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


def classify_equilibria(
    roots=None,
    gamma: float = DEFAULT_GAMMA,
    scan_tol: float = SCAN_TOL,
    cluster_distance: float = CLUSTER_DISTANCE,
    num_points: int = SCAN_NUM_POINTS,
    value_tol: float = GLOBAL_MIN_TOL,
):
    """Compute equilibrium classifications and identify global minima."""
    if roots is None:
        _, _, roots = find_equilibria_pipeline(
            num_points=num_points,
            scan_tol=scan_tol,
            cluster_distance=cluster_distance,
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
    records, global_min_value = classify_equilibria()

    print("\nEquilibrium classification summary:\n")
    print_plain_table(records)
    print_latex_equilibria_table(records)
    print_latex_minima_table(records, global_min_value)
