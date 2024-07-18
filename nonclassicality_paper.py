#Following ref: 'Nonclassicality and entanglement of photon subtracted TMS coherent states

import numpy as np
from sympy import symbols, diff, exp, simplify
import cmath
from cmath import sqrt
import math
from math import factorial, sinh, cosh, tanh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from pprint import pprint
from scipy.linalg import block_diag
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from numpy import where
import matplotlib.colors as mcolors

def sqfactor(z): #find the lambda squeezing factor as a function of z of the covariance matrix
    return -np.log(z)/2


def A_func(alpha1,alpha2,z):
    lamb=sqfactor(z)
    A= alpha1*cosh(lamb)+ np.conjugate(alpha2)*sinh(lamb)
    return A*1j/sqrt(sinh(lamb)*cosh(lamb))

def B_func(alpha1,alpha2,z):
    lamb=sqfactor(z)
    B= np.conjugate(alpha1)*sinh(lamb)+ alpha2*cosh(lamb)
    return B*1j/sqrt(sinh(lamb)*cosh(lamb))


def hermite_polynomial(x,y,m,n):  #m is the number of photon subtractions on mode 1 and n the number of photon subtractions on mode 2
    tau, rho = symbols('tau rho', real=True)
    f= exp(-tau*rho+tau*x+rho*y)
    for i in range(m):
        f= diff(f,tau)
    for i in range(n):
        f= diff(f,rho)
    return f.subs({tau: 0, rho:0})

#print(hermite_polynomial(0,0,1,0))

#in our zero-mean case, the variables A and B are zero. We compute the hermite polynomials for these values of x=y=0

def hermite_polynomial00(m,n):  #m is the number of photon subtractions on mode 1 and n the number of photon subtractions on mode 2
    tau, rho = symbols('tau rho', real=True)
    f= exp(-tau*rho)
    for i in range(m):
        f= diff(f,tau)
    for i in range(n):
        f= diff(f,rho)
    return f.subs({tau: 0, rho:0})

#print(hermite_polynomial00(1,0))

   

def normalization(m,n,alpha1,alpha2,z):
    lamb=sqfactor(z)
    A=A_func(alpha1,alpha2,z)
    B=B_func(alpha1,alpha2,z)
    multiplicative_constant= ((factorial(m)*factorial(n))**2) * (0.5*sinh(2*lamb))**(m+n)
    acc=0
    for l in range(m+1):
        for k in range(n+1):
            acc+=(tanh(lamb)**(l+k)*(np.abs(hermite_polynomial(A,B,m-l,n-k))**2) )/ (factorial(l)*factorial(k)*((factorial(m-l)*factorial(n-k))**2))
    return multiplicative_constant*acc

#sanity checks (we apply formulas 16 and 17 from the paper to check thet our results are consistent with those obtained with our NN
def N1(m,n,alpha1,alpha2,z):
    lamb=sqfactor(z)
    return normalization(m+1,n,alpha1,alpha2,z)/normalization(m,n,alpha1,alpha2,z)

def N2(m,n,alpha1,alpha2,z):
    lamb=sqfactor(z)
    return normalization(m,n+1,alpha1,alpha2,z)/normalization(m,n,alpha1,alpha2,z)

def N1_squared(m,n,alpha1,alpha2,z):
    lamb=sqfactor(z)
    return normalization(m+2,n,alpha1,alpha2,z)/normalization(m,n,alpha1,alpha2,z)

def N2_squared(m,n,alpha1,alpha2,z):
    lamb=sqfactor(z)
    return normalization(m,n+2,alpha1,alpha2,z)/normalization(m,n,alpha1,alpha2,z)

def arbitrary_expval(m,n,alpha1,alpha2,z,p,q,r,s):
    #m and n are the number of photon subtractions (on modes 1 and 2) of our non-gaussian state
    #lamb is the squeezing factor of our state, associated to parameter z
    lamb=sqfactor(z) 
    A=A_func(alpha1,alpha2,z)
    B=B_func(alpha1,alpha2,z)
    #p, q, r, s are the exponents of adag_1, a_1, adag_2, a_2, respectively, of the single-photon operations we perform on each mode
    #first we calculate de summatory:
    sum=0
    for l in range(min(m+p+1,m+q+1)):
        for k in range(min(n+r+1,n+s+1)):
            sum+=(tanh(lamb)**(l+k)/(factorial(l)*factorial(k)))*((factorial(m+p)*factorial(n+r))/(factorial(m+p-l)*factorial(n+r-k)))*((factorial(m+q)*factorial(n+s))/(factorial(m+q-l)*factorial(n+s-k)))*hermite_polynomial(A,B,m+q-l,n+s-k)*hermite_polynomial(np.conjugate(A),np.conjugate(B),m+p-l,n+r-k)
    return sum*(0.5*np.sinh(2*lamb))**(m+n)*(-1)**(q+s)*(-0.5*np.sinh(2*lamb))**((p+q+r+s)/2)/normalization(m,n,alpha1,alpha2,z)

def checks(m,n,z): #we define a function to check if all expressions from the paper's equation (16,17) yield the same as the arbitrary one
    check1, check2, check3, check4 = np.random.rand(4)
    check1= arbitrary_expval(m,n,z,1,1,0,0)-normalization(m+1,n,z)/normalization(m,n,z)
    check2= arbitrary_expval(m,n,z,0,0,1,1)-normalization(m,n+1,z)/normalization(m,n,z)
    check3= arbitrary_expval(m,n,z,2,2,0,0)-normalization(m+2,n,z)/normalization(m,n,z)
    check4= arbitrary_expval(m,n,z,0,0,2,2)-normalization(m,n+2,z)/normalization(m,n,z)
    return np.round(float(check1)), np.round(float(check2)), np.round(float(check3),4), np.round(float(check4),4)

#print(checks(1,3,0.7))

def energy(m,n,alpha1,alpha2,z):
    return arbitrary_expval(m,n,alpha1,alpha2,z,1,1,0,0) + arbitrary_expval(m,n,alpha1,alpha2,z,0,0,1,1)

def delta_n(m,n,alpha1,alpha2,z):
    return sqrt(arbitrary_expval(m,n,alpha1,alpha2,z,2,2,0,0) + arbitrary_expval(m,n,alpha1,alpha2,z,1,1,0,0) + 2*arbitrary_expval(m,n,alpha1,alpha2,z,1,1,1,1) + arbitrary_expval(m,n,alpha1,alpha2,z,0,0,2,2) + arbitrary_expval(m,n,alpha1,alpha2,z,0,0,1,1) - (arbitrary_expval(m,n,alpha1,alpha2,z,1,1,0,0) + arbitrary_expval(m,n,alpha1,alpha2,z,0,0,1,1))**2)

def signal_to_noise(m,n,alpha1,alpha2,z):
    numerator= arbitrary_expval(m,n,alpha1,alpha2,z,1,1,0,0) + arbitrary_expval(m,n,alpha1,alpha2,z,0,0,1,1) 
    denominator= sqrt(arbitrary_expval(m,n,alpha1,alpha2,z,2,2,0,0) + arbitrary_expval(m,n,alpha1,alpha2,z,1,1,0,0) + 2*arbitrary_expval(m,n,alpha1,alpha2,z,1,1,1,1) + arbitrary_expval(m,n,alpha1,alpha2,z,0,0,2,2) + arbitrary_expval(m,n,alpha1,alpha2,z,0,0,1,1) - numerator**2)
    return numerator/denominator

def sv_criterion(m,n,alpha1,alpha2,z):
    sum1= (normalization(m+1,n,alpha1,alpha2,z)/normalization(m,n,alpha1,alpha2,z) -0.5)*(normalization(m,n+1,alpha1,alpha2,z)/normalization(m,n,alpha1,alpha2,z) -0.5)
    sum2= arbitrary_expval(m,n,alpha1,alpha2,z,1,0,1,0)*arbitrary_expval(m,n,alpha1,alpha2,z,0,1,0,1)
    return sum1-sum2

def g2(m,n,alpha1,alpha2,z):
    return 1/signal_to_noise(m,n,alpha1,alpha2,z)**2 - (1/energy(m,n,alpha1,alpha2,z)) +1

def check(m,n,alpha1,alpha2,z):
    return signal_to_noise(m,n,alpha1,alpha2,z)/delta_n(m,n,alpha1,alpha2,z)

def gaussian_displacement(function):
    z_vec=np.linspace(0.01,0.95,50)
    displacement=np.linspace(0,2,25)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    cmap=cm.rainbow
    norm = mcolors.Normalize(vmin=z_vec.min(),vmax=z_vec.max())
    for i in range(len(z_vec)):
        yvec= [function(0,0,alpha,alpha,z_vec[i]) for alpha in displacement]
        ax.plot(displacement,yvec, color=cmap(norm(z_vec[i])))
    ax.plot(displacement, [1]*len(displacement), linestyle='dashed')
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, location='right') 
    cbar.set_label('Squeezing parameter')
    ax.set_xlabel(r'Displacement $\alpha$')
    ax.set_ylabel(f'{function}')
    plt.savefig('snr div by delta gaussian with displacement.pdf')
    plt.show()
    return

gaussian_displacement(check)



