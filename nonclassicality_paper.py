#Following ref: 'Nonclassicality and entanglement of photon subtracted TMS coherent states

import numpy as np
from sympy import symbols, diff, exp, simplify
import math
from math import factorial, sqrt
import matplotlib.pyplot as plt


#in out zero-mean case, the variables A and B are zero. We compute the hermite polynomials for these values of x=y=0

def hermite_polynomial00(m,n):  #m is the number of photon subtractions on mode 1 and n the number of photon subtractions on mode 2
    tau, rho = symbols('tau rho', real=True)
    f= exp(-tau*rho)
    for i in range(m):
        f= diff(f,tau)
    for i in range(n):
        f= diff(f,rho)
    return f.subs({tau: 0, rho:0})

print(hermite_polynomial00(0,0))

def sqfactor(z): #find the lambda squeezing factor as a function of z of the covariance matrix
    return np.log(z)/2
    

def normalization(m,n,z):
    lamb=sqfactor(z)
    multiplicative_constant= ((factorial(m)*factorial(n))**2) * (0.5*np.sinh(2*lamb))**(m+n)
    acc=0
    for l in range(m+1):
        for k in range(n+1):
            acc+=(np.tanh(lamb)**(l+k)*(np.abs(hermite_polynomial00(m-l,n-k))**2) )/ (factorial(l)*factorial(k)*((factorial(m-l)*factorial(n-k))**2))
    return multiplicative_constant*acc

#sanity checks (we apply formulas 16 and 17 from the paper to check thet our results are consistent with those obtained with our NN
def N1(m,n,z):
    lamb=sqfactor(z)
    return normalization(m+1,n,z)/normalization(m,n,z)

def N2(m,n,z):
    lamb=sqfactor(z)
    return normalization(m,n+1,z)/normalization(m,n,z)

def N1_squared(m,n,z):
    lamb=sqfactor(z)
    return normalization(m+2,n,z)/normalization(m,n,z)

def N2_squared(m,n,z):
    lamb=sqfactor(z)
    return normalization(m,n+2,z)/normalization(m,n,z)

def arbitrary_expval(m,n,z,p,q,r,s):
    #m and n are the number of photon subtractions (on modes 1 and 2) of our non-gaussian state
    #lamb is the squeezing factor of our state, associated to parameter z
    lamb=sqfactor(z) 
    #p, q, r, s are the exponents of adag_1, a_1, adag_2, a_2, respectively, of the single-photon operations we perform on each mode
    #first we calculate de summatory:
    sum=0
    for l in range(min(m+p+1,m+q+1)):
        for k in range(min(n+r+1,n+s+1)):
            sum+=(np.tanh(lamb)**(l+k)/(factorial(l)*factorial(k)))*((factorial(m+p)*factorial(n+r))/(factorial(m+p-l)*factorial(n+r-k)))*((factorial(m+q)*factorial(n+s))/(factorial(m+q-l)*factorial(n+s-k)))*hermite_polynomial00(m+q-l,n+s-k)*hermite_polynomial00(m+p-l,n+r-k)
    return sum*(0.5*np.sinh(2*lamb))**(m+n)*(-1)**(q+s)*(-0.5*np.sinh(2*lamb))**((p+q+r+s)/2)/normalization(m,n,z)

def checks(m,n,z): #we define a function to check if all expressions from the paper's equation (16,17) yield the same as the arbitrary one
    check1, check2, check3, check4 = np.random.rand(4)
    check1= arbitrary_expval(m,n,z,1,1,0,0)-normalization(m+1,n,z)/normalization(m,n,z)
    check2= arbitrary_expval(m,n,z,0,0,1,1)-normalization(m,n+1,z)/normalization(m,n,z)
    check3= arbitrary_expval(m,n,z,2,2,0,0)-normalization(m+2,n,z)/normalization(m,n,z)
    check4= arbitrary_expval(m,n,z,0,0,2,2)-normalization(m,n+2,z)/normalization(m,n,z)
    return np.round(float(check1)), np.round(float(check2)), np.round(float(check3),4), np.round(float(check4),4)

#print(checks(1,3,0.7))


def delta_n(m,n,z):
    return sqrt(arbitrary_expval(m,n,z,2,2,0,0) + arbitrary_expval(m,n,z,1,1,0,0) + 2*arbitrary_expval(m,n,z,1,1,1,1) + arbitrary_expval(m,n,z,0,0,2,2) + arbitrary_expval(m,n,z,0,0,1,1) - (arbitrary_expval(m,n,z,1,1,0,0) + arbitrary_expval(m,n,z,0,0,1,1))**2)

def signal_to_noise(m,n,z):
    numerator= arbitrary_expval(m,n,z,1,1,0,0) + arbitrary_expval(m,n,z,0,0,1,1)
    denominator= sqrt(arbitrary_expval(m,n,z,2,2,0,0) + arbitrary_expval(m,n,z,1,1,0,0) + 2*arbitrary_expval(m,n,z,1,1,1,1) + arbitrary_expval(m,n,z,0,0,2,2) + arbitrary_expval(m,n,z,0,0,1,1) - numerator**2)
    return numerator/denominator

def sv_criterion(m,n,z):
    sum1= (normalization(m+1,n,z)/normalization(m,n,z) -0.5)*(normalization(m,n+1,z)/normalization(m,n,z) -0.5)
    sum2= arbitrary_expval(m,n,z,1,0,1,0)*arbitrary_expval(m,n,z,0,1,0,1)
    return sum1-sum2



z_vec=list(np.arange(0.10,0.95,0.01))
index=len(z_vec)
yvec1= [N1(2,0,z)+N2(2,0,z) for z in z_vec]
yvec2= [delta_n(2,0,z) for z in z_vec]
yvec3=[signal_to_noise(2,0,z) for z in z_vec]
yvec_left= [sv_criterion(0,0,z) for z in z_vec[0:index]]
yvec_right= [sv_criterion(1,0,z) for z in z_vec[index+1:]]

plt.plot(z_vec,yvec1)
plt.plot(z_vec,yvec2)
plt.plot(z_vec, yvec3)
plt.legend(['n', 'delta n', 'snr'])
plt.title('1 photon subtraction, balanced beamsplitter')
plt.show()



