import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from heavy_ball_system import rhs, potential

def get_poincare_data(amplitude, gamma, omega, n_cycles=600):
    """
    Integrates the system with control over Amplitude, Gamma, AND Omega.
    """
    T = 2 * np.pi / omega
    t_span = (0, n_cycles * T)
    t_eval = np.arange(0, n_cycles * T, T) # Strobe matches the specific omega
    
    initial_state = [1.0, 1.0, 0.0, 0.0] #[x0, y0, v_x0, v_y0]
    
    sol = solve_ivp(
    lambda t, y: rhs(t, y, gamma=gamma, amplitude=amplitude, omega=omega),
    t_span, 
    initial_state, 
    t_eval=t_eval, 
    method='RK45', 
    rtol=1e-10,  # Tightened from 1e-8
    atol=1e-12   # Added absolute tolerance
)
    
    warm_up = int(0.3 * len(sol.y[0]))
    return sol.y#[:, warm_up:]

def plot_dynamics(amplitude, gamma, omega):
    data = get_poincare_data(amplitude, gamma, omega)
    x, vx = data[0], data[2]
    
    # Calculate Energy
    v_mag2 = data[2]**2 + data[3]**2
    pot = potential(data[0], data[1])
    energy = 0.5 * v_mag2 + pot
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Analysis: $A={amplitude}, \gamma={gamma}, \Omega={omega}$", fontsize=14)
    
    # 1. Plot all points in blue
    axes[0].scatter(x, vx, s=15, color='blue', alpha=0.6, label='Trajectory')
    
    # 2. Highlight the LAST point in red and larger
    # x[-1] and vx[-1] access the final value in the arrays
    axes[0].scatter(x[-1], vx[-1], s=50, color='red', edgecolors='black', zorder=5, label='End State')
    
    axes[0].set_title("Poincaré Section $(x, v_x)$")
    axes[0].set_xlabel("Position $x$")
    axes[0].set_ylabel("Velocity $v_x$")
    axes[0].legend() # Added a legend to help identify the red dot
    
    axes[1].plot(energy, color='red', linewidth=0.8)
    axes[1].set_title("Energy at sampled periods")
    axes[1].set_xlabel("Cycle (n)")
    axes[1].set_ylabel("Energy V")
    
    plt.tight_layout()
    plt.show()



# convergence
#plot_dynamics(amplitude=0.1, gamma=0.5, omega=1.0)

# periodic locking
plot_dynamics(amplitude=0.3, gamma=0.1, omega=1.0)

# chaotic behavior
#plot_dynamics(amplitude=0.5, gamma=0.5, omega=1.0)
