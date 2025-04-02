import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# Parameters
N = 1.0  # Average photon number in initial thermal state
Gamma = 0.1  # Decay rate
t = 1.0  # Time evolution parameter
alpha0 = 0.05  # Squeezing parameter (beta0 = alpha0)

# Covariance matrix at long time (diagonal form due to M1 = M2 = 0)
sigma_inf = np.array([[N + 0.5, 0], [0, N + 0.5]])

# Wigner function definition
def wigner_function(x, p):
    sigma_t = sigma_inf  # Since sigma(0) term vanishes for large t
    det_sigma = np.linalg.det(sigma_t)
    norm_factor = (2 * np.pi * np.sqrt(det_sigma))**-1
    
    X = np.array([x, p])
    exponent = -0.5 * X.T @ np.linalg.inv(sigma_t) @ X
    return norm_factor * np.exp(exponent)

# Grid for phase space plot
x_vals = np.linspace(-3, 3, 100)
p_vals = np.linspace(-3, 3, 100)
X, P = np.meshgrid(x_vals, p_vals)
W_vals = np.array([[wigner_function(x, p) for x in x_vals] for p in p_vals])

# Plot Wigner function
plt.figure(figsize=(8,6))
plt.contourf(X, P, W_vals, levels=50, cmap="RdBu_r")
plt.colorbar(label="Wigner function W(x,p)")
plt.xlabel("x")
plt.ylabel("p")
plt.title("Wigner Function for Given Quantum State")
plt.show()

# Compute expectation values
x2_integral = dblquad(lambda x, p: x**2 * wigner_function(x, p), -np.inf, np.inf, lambda p: -np.inf, lambda p: np.inf)[0]
p2_integral = dblquad(lambda x, p: p**2 * wigner_function(x, p), -np.inf, np.inf, lambda p: -np.inf, lambda p: np.inf)[0]

# Mean photon number
mean_n = 0.5 * (x2_integral + p2_integral - 1)
print(f"Mean photon number: {mean_n:.4f}")

# Compute second moment ⟨n²⟩
n2_integral = 0.25 * dblquad(lambda x, p: (x**2 + p**2 - 1)**2 * wigner_function(x, p), 
                             -np.inf, np.inf, lambda p: -np.inf, lambda p: np.inf)[0]

# Variance in photon number
variance_n = n2_integral - mean_n**2
print(f"Variance in photon number: {variance_n:.4f}")