import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from heavy_ball_system import rhs

def get_bifurcation_data(a_start=0
                         , a_end=10, num_steps=500):
    # Parameters
    gamma = 0.1
    omega = 1.0
    T = 2 * np.pi / omega
    
    # Simulation settings
    # We use fewer cycles because we 'seed' each run with the previous result
    n_warmup = 150   
    n_observe = 40   
    t_span = (0, (n_warmup + n_observe) * T)
    t_eval = np.arange(n_warmup * T, (n_warmup + n_observe) * T, T)
    
    amplitudes = np.linspace(a_start, a_end, num_steps)
    bif_a = []
    bif_x = []

    # Initial state for the very first A
    current_state = np.array([1.0, 1.0, 0.0, 0.0])

    print(f"Starting sweep: A={a_start} to A={a_end}")

    for i, a in enumerate(amplitudes):
        sol = solve_ivp(
            lambda t, y: rhs(t, y, gamma=gamma, amplitude=a, omega=omega),
            t_span,
            current_state,
            t_eval=t_eval,
            method="RK45", 
            rtol=1e-8, 
            atol=1e-10
        )
        
        if sol.success:
            x_pts = sol.y[0]
            bif_x.extend(x_pts.tolist())
            bif_a.extend([a] * len(x_pts))
            
            # Update current_state to the LAST point of this simulation
            # This ensures the next amplitude starts right where the last one left off
            current_state = sol.y[:, -1]

        if (i + 1) % 3 == 0:
            print(f"  Progress: {i+1}/{num_steps} (A = {a:.3f})")

    return bif_a, bif_x

# --- Run and Plot ---
a_vals, x_vals = get_bifurcation_data()

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(a_vals, x_vals, s=0.5, color="red", alpha=0.5, marker='o')

ax.set_title(r"Bifurcation Sweep: Detection of the 2-Cycle transition", fontsize=14)
ax.set_xlabel("Forcing Amplitude $A$")
ax.set_ylabel("Poincaré Sample $x$")
ax.grid(True, alpha=0.2)

# Mark your specific 2-cycle point
#ax.axvline(x=0.3, color='blue', linestyle='--', alpha=0.6, label='Your Case (A=0.3)')
ax.legend()

plt.tight_layout()
plt.show()