from sympy import symbols, sign, sqrt, Abs
from sympy.solvers import solve

# Define the variable
z1 = symbols('z1')

# Define the expression
expression = 0.5*(z1 - 1)**2*(-2*z1**2 + 0.25*z1 - 2.25 + (z1**2 + 1)**2/(2*z1**2))/((-0.25*(z1 - 1)**2 + (z1**2 + 1)**2/z1)**(3/2)*(1-z1)) + 0.5*(2*z1 - 2)/(sqrt(-0.25*(z1 - 1)**2 + (z1**2 + 1)**2/z1)*(1-z1)) - 0.5*(-1)/sqrt(-0.25*(z1 - 1)**2 + (z1**2 + 1)**2/z1)

# Find the roots
roots = solve(expression, z1)
print("Roots:", roots)