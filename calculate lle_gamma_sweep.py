import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from heavy_ball_system import rhs

def calculate_lle(gamma, amplitude=1.0, omega=1.0, t_max=400):
    """
    Estimates the Largest Lyapunov Exponent by tracking two nearby trajectories.
    """
    d0 = 1e-8  # Tiny initial separation
    s1 = np.array([0.1, 0.1, 0.0, 0.0])    # Reference trajectory
    s2 = s1 + np.array([d0, 0, 0, 0])       # Perturbed trajectory

    sol1 = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        [0, t_max], s1, method='RK45', rtol=1e-10, atol=1e-12
    )
    sol2 = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        [0, t_max], s2, method='RK45', rtol=1e-10, atol=1e-12
    )

    if not sol1.success or not sol2.success:
        return np.nan  # Guard against failed integrations

    # LLE via final-time divergence: λ ≈ ln(d_final / d_initial) / t_max
    dist_final = np.linalg.norm(sol1.y[:, -1] - sol2.y[:, -1])

    if dist_final == 0:
        return -np.inf  # Trajectories never separated

    lle = np.log(dist_final / d0) / t_max
    return lle

# --- Parameter Sweep ---
gammas = np.linspace(0.0, 3.0, 50)
lle_values = []

print("Starting Parameter Sweep... (this may take a minute)")
for g in gammas:
    val = calculate_lle(g)
    lle_values.append(val)
    print(f"gamma={g:.2f} | LLE={val:.4f}")

# --- Plotting ---
lle_arr = np.array(lle_values)

plt.figure(figsize=(10, 6))
plt.plot(gammas, lle_arr, 'o-', markersize=4, color='black',
         label="Largest Lyapunov Exponent")
plt.axhline(0, color='red', linestyle='--', label=r"Chaotic Threshold ($\lambda=0$)")

threshold_idx = np.where(lle_arr > 0)[0]
if len(threshold_idx) > 0:
    g_crit = gammas[threshold_idx[0]]
    #plt.axvline(g_crit, color='blue', alpha=0.3,
               # label=f"Threshold γ ≈ {g_crit:.2f}")
    plt.fill_between(gammas, 0, lle_arr,
                     where=(lle_arr > 0), color='red', alpha=0.1)

plt.title("Complexity Diagnostic: LLE vs. Damping Coefficient")
plt.xlabel("Damping Coefficient (γ)")
plt.ylabel(r"Largest Lyapunov Exponent ($\lambda$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()