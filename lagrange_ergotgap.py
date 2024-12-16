import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import sin, cos
# Define the variables and parameters
def lagrange_method():
    a,b,c,d,s = sp.symbols('a,b,c,d,s', real=True, nonnegative=True)  # Variables
    lambda_ = sp.symbols('lambda')  # Lagrange multiplier

    # Define the objective function f(x1, x2; p1, p2)
    f = (a+b)**2/(a**2+b**2+2*c*d+2*sp.sqrt(a**2*b**2+c**2*d**2+a*b*(c**2+d**2)))

    # Define the constraint g(x1, x2) = 0
    g = a**2*b**2+c**2*d**2-a*b*(c**2+d**2)-a**2-b**2+2*c*d+1-s**2

    # Define the Lagrangian L(x1, x2, lambda)
    L = f - lambda_ * g

    # Compute the gradients of the Lagrangian with respect to x1, x2, and lambda
    grad_a = sp.diff(L, a)
    grad_b = sp.diff(L, b)
    grad_c = sp.diff(L, c)
    grad_d = sp.diff(L, d)
    grad_s = sp.diff(L, s)
    grad_lambda = sp.diff(L, lambda_)

    # Solve the system of equations (grad_x1 = 0, grad_x2 = 0, grad_lambda = 0)
    solution = sp.solve([grad_a, grad_b, grad_c,grad_d, grad_s, grad_lambda], [a,b,c,d,s, lambda_])
    print(solution)
    return solution

def lagrange_method_version2():
    k1,k2,theta,z1,z2,s = sp.symbols('k1,k2,theta,z1,z2,s', real=True, nonnegative=True)  # Variables
    lambda_ = sp.symbols('lambda')  # Lagrange multiplier

    # Define the objective function f(x1, x2; p1, p2)
    f = ((k1**2+k2**2)*(cos(theta)**4+sin(theta)**4))/(k1+k2)**2 + (2*k1*k2*(z1**2+z2**2)*cos(theta)**2*sin(theta)**2)/((k1+k2)**2*(z1*z2))

    # Define the constraint g(x1, x2) = 0
    g = z1*z2*(1+k1**2*k2**2-k1**2-k2**2)-4*cos(theta)**2*sin(theta)**2*(k1*k2*(z1**2+z2**2)-(k1**2+k2**2)*z1*z2)-s**2

    # Define the Lagrangian L(x1, x2, lambda)
    L = f - lambda_ * g

    # Compute the gradients of the Lagrangian with respect to x1, x2, and lambda
    grad_k1 = sp.diff(L, k1)
    grad_k2 = sp.diff(L, k2)
    grad_theta = sp.diff(L, theta)
    grad_z1 = sp.diff(L, z1)
    grad_z2 = sp.diff(L, z2)
    grad_s =sp.diff(L,s)
    grad_lambda = sp.diff(L, lambda_)
    print(grad_k1)
    print('')
    print(grad_k2)
    print('')
    print(grad_theta)
    print('')
    print(grad_z1)
    print('')
    print(grad_z1)
    print('')
    print(grad_z2)
    print('')
    print(grad_lambda)

    # Solve the system of equations (grad_x1 = 0, grad_x2 = 0, grad_lambda = 0)
    solution = sp.solve([grad_k1, grad_k2, grad_theta,grad_z1,grad_z2, grad_s, grad_lambda], [k1,k2,theta,z1,z2,s, lambda_])
    print(solution)
    return solution
lagrange_method_version2()