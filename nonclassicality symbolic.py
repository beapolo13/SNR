import sympy as sp
from sympy import sqrt
import numpy as np
from numpy import tanh, sinh, log
import matplotlib.pyplot as plt

def hermite_polynomial00(m, n):
    tau, rho = sp.symbols('tau rho', real=True)
    f = sp.exp(-tau * rho)
    for i in range(m):
        f = sp.diff(f, tau)
    for i in range(n):
        f = sp.diff(f, rho)
    return f.subs({tau: 0, rho: 0})

def sqfactor(z):
    return sp.log(z) / 2

def normalization(m, n, z):
    lamb = sqfactor(z)
    multiplicative_constant = ((sp.factorial(m) * sp.factorial(n))**2) * (0.5 * sp.sinh(2 * lamb))**(m + n)
    acc = 0
    for l in range(m + 1):
        for k in range(n + 1):
            acc += (sp.tanh(lamb)**(l + k) * (sp.Abs(hermite_polynomial00(m - l, n - k))**2)) / \
                   (sp.factorial(l) * sp.factorial(k) * ((sp.factorial(m - l) * sp.factorial(n - k))**2))
    return multiplicative_constant * acc

def arbitrary_expval(m, n, z, p, q, r, s):
    lamb = sqfactor(z)
    sum_expr = 0
    for l in range(min(m + p + 1, m + q + 1)):
        for k in range(min(n + r + 1, n + s + 1)):
            sum_expr += ((sp.tanh(lamb)**(l + k) / (sp.factorial(l) * sp.factorial(k))) *
                         ((sp.factorial(m + p) * sp.factorial(n + r)) /
                          (sp.factorial(m + p - l) * sp.factorial(n + r - k))) *
                         ((sp.factorial(m + q) * sp.factorial(n + s)) /
                          (sp.factorial(m + q - l) * sp.factorial(n + s - k))) *
                         hermite_polynomial00(m + q - l, n + s - k) * hermite_polynomial00(m + p - l, n + r - k))
    return sum_expr * (0.5 * sp.sinh(2 * lamb))**(m + n) * (-1)**(q + s) * \
           (-0.5 * sp.sinh(2 * lamb))**((p + q + r + s) / 2) / normalization(m, n, z)

def checks(m, n, z):
    check1 = arbitrary_expval(m, n, z, 1, 1, 0, 0) - normalization(m + 1, n, z) / normalization(m, n, z)
    check2 = arbitrary_expval(m, n, z, 0, 0, 1, 1) - normalization(m, n + 1, z) / normalization(m, n, z)
    check3 = arbitrary_expval(m, n, z, 2, 2, 0, 0) - normalization(m + 2, n, z) / normalization(m, n, z)
    check4 = arbitrary_expval(m, n, z, 0, 0, 2, 2) - normalization(m, n + 2, z) / normalization(m, n, z)
    return sp.round(check1, 4), sp.round(check2, 4), sp.round(check3, 4), sp.round(check4, 4)

def n(m, n, z):
    return arbitrary_expval(m, n, z, 1, 1, 0, 0) +arbitrary_expval(m, n, z, 0, 0, 1, 1)

def delta_n(m, n, z):
    return sqrt(arbitrary_expval(m, n, z, 2, 2, 0, 0) + arbitrary_expval(m, n, z, 1, 1, 0, 0) +
            2 * arbitrary_expval(m, n, z, 1, 1, 1, 1) + arbitrary_expval(m, n, z, 0, 0, 2, 2) +
            arbitrary_expval(m, n, z, 0, 0, 1, 1) - (arbitrary_expval(m, n, z, 1, 1, 0, 0) +
                                                     arbitrary_expval(m, n, z, 0, 0, 1, 1))**2)

def signal_to_noise(m, n, z):
    numerator = arbitrary_expval(m, n, z, 1, 1, 0, 0) + arbitrary_expval(m, n, z, 0, 0, 1, 1)
    denominator = sqrt(arbitrary_expval(m, n, z, 2, 2, 0, 0) + arbitrary_expval(m, n, z, 1, 1, 0, 0) +
                   2 * arbitrary_expval(m, n, z, 1, 1, 1, 1) + arbitrary_expval(m, n, z, 0, 0, 2, 2) +
                   arbitrary_expval(m, n, z, 0, 0, 1, 1) - numerator**2)
    return numerator / denominator


z = sp.symbols('z')
# Define the expression for delta_n using the previously defined functions
n_expr= sp.simplify(n(0,0,z))
delta_expr = sp.simplify(delta_n(0, 0, z))
snr_exp=signal_to_noise(0,0,z)
print('N2', sp.simplify(arbitrary_expval(0, 0, z, 2, 2, 0, 0) + arbitrary_expval(0, 0, z, 1, 1, 0, 0) +
            2 * arbitrary_expval(0, 0, z, 1, 1, 1, 1) + arbitrary_expval(0, 0, z, 0, 0, 2, 2) +
            arbitrary_expval(0, 0, z, 0, 0, 1, 1)))
print('N', n_expr)
print('delta N',delta_expr)
print('snr',snr_exp)


#comparison between the expression for the snr in the paper's and isserlis formulations
z_vec=np.arange(0.01,3.99,0.01)
y_vec1=[-1.0*(-sinh(log(z)))**1.0*tanh(log(z)/2)/sqrt(-1.0*(-sinh(log(z)))**1.0*tanh(log(z)/2) + 0.5*(-sinh(log(z)))**2.0*(tanh(log(z)/2)**2 + 1)) for z in z_vec]
y_vec2=[0.707106781186547*(z1 - 1)**2/(z1*sqrt(0.5 + 1.0/z1 + 0.5/z1**2)*abs(z1 - 1)) for z1 in z_vec]
plt.plot(z_vec,y_vec1)
plt.plot(z_vec,y_vec2)
plt.show()