import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from heavy_ball_system import rhs

def calculate_lle_omega(omega, amplitude=1.0, gamma=0.5, t_max=400):
    """
    Estimates the Largest Lyapunov Exponent for a specific Omega.
    """
    d0 = 1e-8  
    s1 = np.array([0.1, 0.1, 0.0, 0.0])
    s2 = s1 + np.array([d0, 0, 0, 0])

    # Tight tolerances are recommended for frequency sweeps
    sol1 = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        [0, t_max], s1, t_eval=[t_max], method='RK45', rtol=1e-10, atol=1e-12
    )
    sol2 = solve_ivp(
        lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
        [0, t_max], s2, t_eval=[t_max], method='RK45', rtol=1e-10, atol=1e-12
    )

    if not sol1.success or not sol2.success:
        return np.nan 

    dist_final = np.linalg.norm(sol1.y[:, -1] - sol2.y[:, -1])
    
    if dist_final <= 0:
        return -np.inf

    return np.log(dist_final / d0) / t_max

# --- Parameter Sweep ---
# Varying Omega from slow (0.1) to fast (3.0)
omegas = np.linspace(0.1, 10.0, 50)
lle_values = []

print("Starting Omega Sweep...")
for o in omegas:
    val = calculate_lle_omega(o)
    lle_values.append(val)
    print(f"Omega={o:.2f} | LLE={val:.4f}")

# --- Plotting ---
lle_arr = np.array(lle_values)

plt.figure(figsize=(10, 6))
plt.plot(omegas, lle_arr, 'o-', markersize=4, color='black',
         label="Largest Lyapunov Exponent")
plt.axhline(0, color='red', linestyle='--', label=r"Chaotic Threshold ($\lambda=0$)")

# Using your requested structure for the red filling
threshold_idx = np.where(lle_arr > 0)[0]
if len(threshold_idx) > 0:
    plt.fill_between(omegas, 0, lle_arr,
                     where=(lle_arr > 0), color='red', alpha=0.1)

plt.title(r"Complexity Diagnostic: LLE vs. Driving Frequency ($\Omega$)")
plt.xlabel(r"Driving Frequency ($\Omega$)")
plt.ylabel(r"Largest Lyapunov Exponent ($\lambda$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()