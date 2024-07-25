import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip import coherent, destroy, expect

# Define parameters
alpha = 2 +0j  # Coherent state parameter
N = 30          # Truncation dimension for the Fock space

# Define the coherent states
psi_plus = (coherent(N, alpha) + coherent(N, -alpha)).unit()
psi_minus = (coherent(N, alpha) - coherent(N, -alpha)).unit()

# Define the cat state (even cat state)
cat_state = psi_plus

# Define the photon number operator
a = destroy(N)
n = a.dag() * a

# Calculate the expectation value of the photon number
expectation_value = expect(n, cat_state)

print(f"Expectation value of the photon number: {expectation_value}")

# Define parameters
alpha = 15  # Coherent state amplitude

# Create the coherent states |alpha> and |-alpha>
coherent_alpha = coherent(30, alpha)
coherent_neg_alpha = coherent(30, -alpha)

# Create the odd cat state
odd_cat_state = (coherent_alpha - coherent_neg_alpha).unit()

# Create the phase space grid
x = np.linspace(-5, 5, 500)
p = np.linspace(-5, 5, 500)

# Calculate the Wigner function
W = wigner(odd_cat_state, x, p)

# Plot the Wigner function
plt.figure(figsize=(8, 6))
plt.contourf(x, p, W, 100, cmap='RdBu')
plt.colorbar()
plt.title("Wigner Function of the Odd Cat State")
plt.xlabel(r'$x$')
plt.ylabel(r'$p$')
plt.show()
