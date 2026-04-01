from typing import Sequence

import numpy as np


K = 6.0
EPS = 0.3


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

## not used ##
def forcing(t: float, amplitude: float, omega: float) -> float:
    """Return the external forcing term A sin(omega t)."""
    return amplitude * np.sin(omega * t)


def rhs(
    t: float,
    state: Sequence[float],
    gamma: float,
    amplitude: float,
    omega: float,
    eps: float = EPS,
    k: float = K,
):
    """
    Return the heavy-ball system right-hand side.

    State is ordered as (x, y, vx, vy).
    """
    x, y, vx, vy = state
    drive = forcing(t, amplitude, omega)

    x_dot = vx
    y_dot = vy
    vx_dot = -gamma * vx - dfdx(x, y, eps=eps, k=k) - drive
    vy_dot = -gamma * vy - dfdy(x, y, eps=eps, k=k) - drive

    return np.array([x_dot, y_dot, vx_dot, vy_dot], dtype=float)
