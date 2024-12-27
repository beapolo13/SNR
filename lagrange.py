import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
# Define the variables and parameters
from sympy import symbols, Eq, diff, solve, Function

# Define variables and the Lagrange multiplier
y, z, lambda_ , nu= symbols('y z lambda nu')
# Define the function to minimize and the constraint
f = (nu**2/8)*(z**2 + 1/z**2) + nu*y*z  # The function to minimize
g = (nu/4)*(z+1/z-2) + y - 1  # The constraint function

# Lagrange's equations
lagrange_eqs = [
    diff(f, var) - lambda_ * diff(g, var) for var in (y, z)
]
print(lagrange_eqs)
# Add the constraint equation
g_constraint = Eq(g, 0)

# Solve the system of equations
solutions = solve(lagrange_eqs + [g_constraint], (y, z, lambda_))

# Print the solutions
print("Solutions:")
print(solutions)

def find_optimal_gaussian(t_vec): 
    # Define the variable (symbol)
    z = sp.Symbol('z', real=True)
    nu = sp.symbols('nu', real=True)
    poly_expr = 1-k*z**2 - (4+2*k)*z**3 + k*z**4 #the polynomial that we input here is that given by the method of larange multipliers
    roots = sp.solve(poly_expr, z)
    #find which of the roots satisfies that it is real and within (0,1) by substituting at any k (e.g k=1)
    found_root=False
    root_index= None
    i=0
    while found_root == False:
        x= roots[i].subs({k:1}) 
        if x.is_real == True:
            if np.float64(x) > 0 and np.float64(x) < 1:
                found_root =True
                root_index = i
        else:
            i +=1
    nu_vec = [1/np.tanh(1/(2* t)) for t in t_vec]
    z_opt=[]
    for nu in nu_vec:
        z_opt +=[roots[root_index].subs({k:nu})]
    return z_opt

#find_optimal_gaussian()
# Example usage
x = sp.Symbol('x', real=True)
k = sp.Symbol('k', real=True)
  # Polynomial: x^3 - 6x^2 + 11x - 6
roots = find_roots_sympy(poly_expr)
print("Roots (symbolic):", roots)
print(len(roots))

t_vec = np.linspace(0.0001,1,60)
nu_vec = [1/np.tanh(1/(2* t)) for t in t_vec]
z_opt=[]
for nu in nu_vec:
    z_opt +=[roots[2].subs({k:nu})]

plt.plot(t_vec,z_opt)
plt.show()