import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from heavy_ball_system import rhs


def get_poincare_x(
    amplitude,
    gamma=0.5,
    omega=1.0,
    n_cycles=1000,
    warmup_fraction=0.5,
):
    T        = 2 * np.pi / omega
    t_warmup = warmup_fraction * n_cycles * T

    sol_warm = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        [0.0, t_warmup],
        np.array([1.0, 1.0, 0.0, 0.0]),
        method="RK45", rtol=1e-10, atol=1e-12,
    )
    if not sol_warm.success:
        return np.array([])

    t_start = t_warmup
    t_end   = t_warmup + (1 - warmup_fraction) * n_cycles * T
    t_eval  = np.arange(t_start, t_end, T)

    sol = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        [t_start, t_end],
        sol_warm.y[:, -1],
        t_eval=t_eval,
        method="RK45", rtol=1e-10, atol=1e-12,
    )
    return sol.y[0] if sol.success else np.array([])


# ── parameters ────────────────────────────────────────────────────────────────
gamma      = 0.1
omega      = 1.0
amplitudes = np.linspace(0.1, 0.3, 60)

# ── sweep ─────────────────────────────────────────────────────────────────────
bifurcation_a = []
bifurcation_x = []

print("Running bifurcation sweep...")
for i, a in enumerate(amplitudes):
    x_pts = get_poincare_x(a, gamma=gamma, omega=omega)
    bifurcation_a.extend([a] * len(x_pts))
    bifurcation_x.extend(x_pts.tolist())

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(amplitudes)} done")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle(
    rf"Bifurcation diagram — $\gamma={gamma},\ \Omega={omega}$",
    fontsize=14,
)

ax.scatter(
    bifurcation_a, bifurcation_x,
    s=5.0, color="red", alpha=0.4, rasterized=True,
)
ax.set_xlabel("Forcing amplitude $A$", fontsize=12)
ax.set_ylabel("Poincaré $x$",          fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()