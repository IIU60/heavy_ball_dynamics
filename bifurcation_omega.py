import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from heavy_ball_system import rhs


def get_poincare_x_omega(
    omega,
    amplitude=0.15,
    gamma=0.5,
    n_cycles=1000,
    warmup_fraction=0.5,
):
    T        = 2 * np.pi / omega
    # for low omega ensure enough cycles during warmup
    t_warmup = max(warmup_fraction * n_cycles * T, 100 * 2 * np.pi / omega)

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
    t_eval = t_eval[t_eval < t_end] 

    sol = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        [t_start, t_end],
        sol_warm.y[:, -1],
        t_eval=t_eval,
        method="RK45", rtol=1e-10, atol=1e-12,
    )
    return sol.y[0] if sol.success else np.array([])


# ── parameters ────────────────────────────────────────────────────────────────
amplitude = 0.15
gamma     = 0.5
omegas    = np.linspace(0.1, 3.0, 50)  # start at 0.1 to avoid omega=0

# ── sweep ─────────────────────────────────────────────────────────────────────
bifurcation_o = []
bifurcation_x = []

print("Running bifurcation sweep (omega)...")
for i, o in enumerate(omegas):
    x_pts = get_poincare_x_omega(o, amplitude=amplitude, gamma=gamma)
    bifurcation_o.extend([o] * len(x_pts))
    bifurcation_x.extend(x_pts.tolist())

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(omegas)} done")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle(
    rf"Bifurcation diagram — $A={amplitude},\ \gamma={gamma}$",
    fontsize=14,
)

ax.scatter(
    bifurcation_o, bifurcation_x,
    s=4.0, color="red", alpha=0.4, rasterized=True,
)
ax.set_xlabel(r"Driving frequency $\Omega$", fontsize=12)
ax.set_ylabel("Poincaré $x$",               fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()