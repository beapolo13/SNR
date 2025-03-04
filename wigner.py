import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Wigner function (or a similar radial function)
def wigner_function(q, p):
    r = np.sqrt(q**2 + p**2)  # Radial coordinate
    return (1/np.pi) * np.exp(-r**2) * np.cos(2 * np.pi * r)  # Example function

def gaussian_wigner(q, p, sigma_q=1.0, sigma_p=1.0):
    norm_factor = 1 / (np.pi * sigma_q * sigma_p)
    return norm_factor * np.exp(- (q**2 / sigma_q**2) - (p**2 / sigma_p**2))

# Define the grid
q = np.linspace(-4, 4, 100)
p = np.linspace(-4, 4, 100)
Q, P = np.meshgrid(q, p)
W=gaussian_wigner(Q,P)
#W = wigner_function(Q, P)  # Compute function values

# Create 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(Q, P, W, cmap='plasma', edgecolor= 'k', alpha=0.8)

# Add contour lines
ax.contour(Q, P, W, levels=10, linewidths=0.1, colors='black', linestyles='solid')

# Labels and aesthetics
ax.set_xlabel("q")
ax.set_ylabel("p")
ax.set_zlabel("W")
ax.set_title("Wigner Function (Example)")

plt.show()