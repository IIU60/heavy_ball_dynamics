import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from heavy_ball_system import rhs


def calculate_lle_benettin_omega(
    omega,
    amplitude=0.15,
    gamma=0.5,
    t_warmup=500.0,
    t_max=1000.0,
    n_steps=500,
    d0=1e-8,
):
    """
    Estimates the LLE vs omega using Benettin's rescaling algorithm.
    Fixes over the naive estimator:
      1. Continuous rescaling prevents saturation.
      2. Measurement runs over [t_warmup, t_warmup + t_max] so
         A*sin(omega*t) is never reset.
    """
    # --- PHASE 1: WARM-UP ---
    s_initial = np.array([1.0, 1.0, 0.0, 0.0])
    sol_warm = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        [0.0, t_warmup],
        s_initial,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol_warm.success:
        return np.nan

    s1 = sol_warm.y[:, -1]
    s2 = s1 + np.array([d0, 0.0, 0.0, 0.0])

    # --- PHASE 2: BENETTIN RESCALING LOOP ---
    t_start    = t_warmup
    t_end      = t_warmup + t_max
    step_times = np.linspace(t_start, t_end, n_steps + 1)

    log_sum = 0.0

    for i in range(n_steps):
        t0, t1 = step_times[i], step_times[i + 1]

        sol1 = solve_ivp(
            lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
            [t0, t1], s1, method="RK45", rtol=1e-10, atol=1e-12,
        )
        sol2 = solve_ivp(
            lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
            [t0, t1], s2, method="RK45", rtol=1e-10, atol=1e-12,
        )

        if not sol1.success or not sol2.success:
            return np.nan

        s1 = sol1.y[:, -1]
        s2 = sol2.y[:, -1]

        d_new = np.linalg.norm(s1 - s2)

        if d_new == 0.0:
            return -np.inf

        log_sum += np.log(d_new / d0)
        s2 = s1 + d0 * (s2 - s1) / d_new

    return log_sum / t_max


# --- Parameter sweep over omega ---
omegas     = np.linspace(0.0001, 3.0, 50)  # start at 0.1, not 0.0 (see note below)
lle_values = []

print(f"{'Omega':<12} | {'LLE':<10}")
print("-" * 25)

for o in omegas:
    val = calculate_lle_benettin_omega(o)
    lle_values.append(val)
    print(f"{o:<12.4f} | {val:<10.4f}")

# --- Plotting ---
lle_arr = np.array(lle_values)

plt.figure(figsize=(10, 6))
plt.plot(
    omegas, lle_arr,
    "o-", markersize=5, color="black", linewidth=1.5,
    label="Benettin LLE",
)
plt.axhline(0, color="red", linestyle="--", alpha=0.8,
            label=r"Chaos threshold ($\lambda = 0$)")
plt.fill_between(omegas, 0, lle_arr, where=(lle_arr > 0),  color="red",   alpha=0.1)
plt.fill_between(omegas, 0, lle_arr, where=(lle_arr <= 0), color="green", alpha=0.05)

plt.title(r"LLE vs. driving frequency ($\Omega$)", fontsize=14)
plt.xlabel(r"Driving frequency ($\Omega$)",                 fontsize=12)
plt.ylabel(r"Largest Lyapunov exponent ($\lambda$)",        fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()