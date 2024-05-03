#Following ref: 'Nonclassicality and entanglement of photon subtracted TMS coherent states

import numpy as np
from sympy import symbols, diff, exp
import math
from math import factorial

#in out zero-mean case, the variables A and B are zero. We compute the hermite polynomials for these values of x=y=0

def hermite_polynomial00(m,n):  #m is the number of photon subtractions on mode 1 and n the number of photon subtractions on mode 2
    tau, rho = symbols('tau rho', real=True)
    f= exp(-tau*rho)
    for i in range(m):
        f= diff(f,tau)
    for i in range(n):
        f= diff(f,rho)
    return f.subs({tau: 0, rho:0})

#hermite_polynomial00(0,0)

def sqfactor(z): #find the lambda squeezing factor as a function of z of the covariance matrix
    return np.log(z)/2

def normalization(m,n,lamb):
    multiplicative_constant= ((factorial(m)*factorial(n))**2) * (0.5*np.sinh(2*lamb))**(m+n)
    acc=0
    for l in range(m+1):
        for k in range(n+1):
            acc+=(np.tanh(lamb)**(l+k)*(np.abs(hermite_polynomial00(m-l,n-k))**2) )/ (factorial(l)*factorial(k)*((factorial(m-l)*factorial(n-k))**2))
    return multiplicative_constant*acc

#normalization(0,1,1)


#sanity checks (we apply formulas 16 and 17 from the paper to check thet our results are consistent with those obtained with our NN
def N1(m,n,lamb):
    return normalization(m+1,n,lamb)/normalization(m,n,lamb)


def N2(m,n,lamb):
    return normalization(m,n+1,lamb)/normalization(m,n,lamb)

def N1_squared(m,n,lamb):
    return normalization(m+2,n,lamb)/normalization(m,n,lamb)

def N2_squared(m,n,lamb):
    return normalization(m,n+2,lamb)/normalization(m,n,lamb)

print(normalization(1,0,0.5))
print(N1(1,0,sqfactor(0.5)))
print(N2(1,0,sqfactor(0.5)))