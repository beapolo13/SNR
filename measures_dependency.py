from sympy import symbols, sqrt, cos, sin, log, Matrix, simplify, Rational

# Define variables
z1, z2, theta = symbols('z1 z2 theta')

# Define the functions
argument = z1**2 + 6*z1*z2 + z2**2 - (z1 - z2)**2 * cos(4*theta)
f = sqrt((z2*cos(theta)**2 + z1*sin(theta)**2)*(z1*cos(theta)**2 + z2*sin(theta)**2) / (z1*z2)) - 1
g = Rational(1, 4) * (
    -8 * log(8, 2) +
    (4 - sqrt(2) * sqrt(argument / (z1 * z2))) * log(-4 + sqrt(2) * sqrt(argument / (z1 * z2)), 2) +
    (4 + sqrt(2) * sqrt(argument / (z1 * z2))) * log(4 + sqrt(2) * sqrt(argument / (z1 * z2)), 2))

# Compute the gradients
grad_f = Matrix([f.diff(z1), f.diff(z2), f.diff(theta)])
grad_g = Matrix([g.diff(z1), g.diff(z2), g.diff(theta)])

# Construct the Jacobian
J = Matrix([grad_f, grad_g])

# Compute the rank of the Jacobian
rank = J.rank()

# Print results
print("Gradient of f:", simplify(grad_f))
print("Gradient of g:", simplify(grad_g))
print("Jacobian matrix:")
print(simplify(J))
print("Rank of the Jacobian matrix:", rank)

# Check dependency
if rank < 2:
    print("The gradients are dependent.")
else:
    print("The gradients are independent.")
