import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
# Define the variables and parameters
def lagrange_method():
    z,y = sp.symbols('z y', real=True, nonnegative=True)  # Variables
    k = sp.symbols('k', real=True, nonnegative=True)
    lambda_ = sp.symbols('lambda')  # Lagrange multiplier

    # Define the objective function f(x1, x2; p1, p2)
    f = (1/8)*k**2 *(z**2 + 1/z**2) -1/4 + k*z*y

    # Define the constraint g(x1, x2) = 0
    g = (1/4)*k*(z+(1/z)-2) + y -1

    # Define the Lagrangian L(x1, x2, lambda)
    L = f - lambda_ * g

    # Compute the gradients of the Lagrangian with respect to x1, x2, and lambda
    grad_z = sp.diff(L, z)
    grad_y = sp.diff(L, y)
    grad_lambda = sp.diff(L, lambda_)

    # Solve the system of equations (grad_x1 = 0, grad_x2 = 0, grad_lambda = 0)
    solution = sp.solve([grad_z, grad_y, grad_lambda], [z, y, lambda_])
    print(solution)
    return solution

def find_optimal_gaussian(t_vec): 
    # Define the variable (symbol)
    z = sp.Symbol('z', real=True)
    k = sp.symbols('k', real=True)
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

find_optimal_gaussian()
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