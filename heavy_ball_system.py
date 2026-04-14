from typing import Sequence

import numpy as np


K = 2.0
EPS = 0.1
DEFAULT_GAMMA = 1.0
DEFAULT_AMPLITUDE = 0.0
DEFAULT_OMEGA = 1.0

XMIN = -2.0
XMAX = 2.0
YMIN = -2.0
YMAX = 2.0

SCAN_NUM_POINTS = 500
SCAN_TOL = 0.05
CLUSTER_DISTANCE = 0.1
ROOT_TOL = 1e-10
DUPLICATE_TOL = 1e-6
CLASSIFICATION_TOL = 1e-8
GLOBAL_MIN_TOL = 1e-8

SURFACE_NUM_POINTS = 201
CONTOUR_NUM_POINTS = 401

DEFAULT_T = 0.0
DEFAULT_VX = 0.0
DEFAULT_VY = 0.0


def potential(x, y, eps: float = EPS, k: float = K):
    """Return f(x, y)."""
    r2 = x**2 + y**2
    return 0.25 * (r2 - 1.0) ** 2 + eps * (np.cos(k * x) + np.cos(k * y))


def dfdx(x, y, eps: float = EPS, k: float = K):
    """Return df/dx."""
    return x * (x**2 + y**2 - 1.0) - eps * k * np.sin(k * x)


def dfdy(x, y, eps: float = EPS, k: float = K):
    """Return df/dy."""
    return y * (x**2 + y**2 - 1.0) - eps * k * np.sin(k * y)


def gradient(x, y, eps: float = EPS, k: float = K):
    """Return grad f = (dfdx, dfdy)."""
    return np.array([dfdx(x, y, eps=eps, k=k), dfdy(x, y, eps=eps, k=k)], dtype=float)


def hessian(x, y, eps: float = EPS, k: float = K):
    """Return the Hessian of f."""
    f_xx = (3.0 * x**2 + y**2 - 1.0) - eps * k**2 * np.cos(k * x)
    f_yy = (x**2 + 3.0 * y**2 - 1.0) - eps * k**2 * np.cos(k * y)
    f_xy = 2.0 * x * y
    return np.array([[f_xx, f_xy], [f_xy, f_yy]], dtype=float)


def jacobian(
    x,
    y,
    gamma: float = DEFAULT_GAMMA,
    eps: float = EPS,
    k: float = K,
):
    """Return the Jacobian at an equilibrium with vx = vy = 0."""
    hess = hessian(x, y, eps=eps, k=k)
    return np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-hess[0, 0], -hess[0, 1], -gamma, 0.0],
            [-hess[1, 0], -hess[1, 1], 0.0, -gamma],
        ],
        dtype=float,
    )

def forcing(
    t: float = DEFAULT_T,
    amplitude: float = DEFAULT_AMPLITUDE,
    omega: float = DEFAULT_OMEGA,
) -> float:
    """Return the external forcing term A sin(omega t)."""
    return amplitude * np.sin(omega * t)


def rhs(
    t: float = DEFAULT_T,
    state: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    gamma: float = DEFAULT_GAMMA,
    amplitude: float = DEFAULT_AMPLITUDE,
    omega: float = DEFAULT_OMEGA,
    eps: float = EPS,
    k: float = K,
):
    """Return the heavy-ball system right-hand side for state (x, y, vx, vy)."""
    x, y, vx, vy = state
    drive = forcing(t=t, amplitude=amplitude, omega=omega)

    x_dot = vx
    y_dot = vy
    vx_dot = -gamma * vx - dfdx(x, y, eps=eps, k=k) - drive
    vy_dot = -gamma * vy - dfdy(x, y, eps=eps, k=k) - drive

    return np.array([x_dot, y_dot, vx_dot, vy_dot], dtype=float)


def rhs_components_on_xy_grid(
    x_grid,
    y_grid,
    t: float = DEFAULT_T,
    gamma: float = DEFAULT_GAMMA,
    amplitude: float = DEFAULT_AMPLITUDE,
    omega: float = DEFAULT_OMEGA,
    vx: float = DEFAULT_VX,
    vy: float = DEFAULT_VY,
    eps: float = EPS,
    k: float = K,
):
    """Evaluate all four RHS components on an x-y grid for fixed vx and vy."""
    drive = forcing(t=t, amplitude=amplitude, omega=omega)
    x_dot = np.full_like(x_grid, vx, dtype=float)
    y_dot = np.full_like(y_grid, vy, dtype=float)
    vx_dot = -gamma * vx - dfdx(x_grid, y_grid, eps=eps, k=k) - drive
    vy_dot = -gamma * vy - dfdy(x_grid, y_grid, eps=eps, k=k) - drive

    return {
        "x_dot": x_dot,
        "y_dot": y_dot,
        "vx_dot": vx_dot,
        "vy_dot": vy_dot,
    }
