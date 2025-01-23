import numpy as np
import matplotlib.pyplot as plt

# Define the energy expectation value and uncertainty functions
def energy_expectation(alpha_squared, hbar=1, omega=1):
    """
    Calculates the expected energy \langle E \rangle for the truncated coherent state.
    """
    return hbar * omega * (alpha_squared / (1 + alpha_squared) + 0.5)

def energy_uncertainty(alpha_squared, hbar=1, omega=1):
    """
    Calculates the energy uncertainty \Delta E for the truncated coherent state.
    """
    # <n> and <n^2>
    n_expectation = alpha_squared / (1 + alpha_squared)
    n_squared_expectation = n_expectation  # Since <n^2> = <n> in the truncated space

    # <H^2> and <H>
    h_squared = (hbar * omega)**2 * (n_squared_expectation + n_expectation + 0.25)
    h_mean = energy_expectation(alpha_squared, hbar, omega)

    # Variance: <H^2> - <H>^2
    variance = h_squared - h_mean**2
    return np.sqrt(variance)

# Generate data for the plot
alpha_squared_values = np.linspace(0, 10, 500)  # Values of |alpha|^2
hbar = 1
omega = 1

# Calculate \langle E \rangle / \Delta E for each alpha^2
ratios = []
for alpha_squared in alpha_squared_values:
    energy_mean = energy_expectation(alpha_squared, hbar, omega)
    uncertainty = energy_uncertainty(alpha_squared, hbar, omega)
    if uncertainty > 0:
        ratios.append(energy_mean / uncertainty)
    else:
        ratios.append(0)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(alpha_squared_values, ratios, label=r"$\langle E \rangle / \Delta E$")
plt.xlabel(r"$|\alpha|^2$")
plt.ylabel(r"$\langle E \rangle / \Delta E$")
plt.title("Energy Expectation vs. Uncertainty Ratio for Truncated Coherent States")
plt.grid(True)
plt.legend()
plt.show()
