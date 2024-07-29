import numpy as np
import matplotlib.pyplot as plt
import qutip as q
#from qutip import *
from qutip import coherent, create, destroy, expect, basis, displace, squeeze, commutator, thermal_dm, ket2dm, mesolve


#PARAMETERS
L = 30        # Truncation dimension for the Fock space
N=1       #number of modes


#auxiliary function
def r(z):
    return -np.log(z)/2

#Define operators
a=destroy(L)
adag=create(L)
num=adag*a

def D(alpha):
    return displace(L,alpha)

def S(z):
    return squeeze(L, r(z))

def n2():
    return(num**2)

def expect_delta_n(state):
    return expect(n2(),state)-(expect(num,state))**2

def snr(state,ref_state):
    return expect(num,state.unit())/expect_delta_n(state.unit())-expect(num,ref_state.unit())/expect_delta_n(state.unit())


def snr_for_mixed_state(state_dm,ref_state_dm):
    numerator=((num*state_dm).tr())/state_dm.tr()
    denominator=((n2()*state_dm).tr())/state_dm.tr() - (((num*state_dm).tr())/state_dm.tr())**2
    numerator_ref=(num*ref_state_dm).tr()
    return numerator/denominator-numerator_ref/denominator

#create different states
vac=basis(L,0)

def coherent(alpha):
    return D(alpha)*vac

def squeezed(z):
    return S(z)*vac

#coherent+squeezed state
def cv_state(alpha,z):
    return S(z)*D(alpha)*basis(L,0)

def cv_state_thermal_dm(alpha,z,avg_n):
    return D(alpha)*S(z)*thermal_dm(L,avg_n)*S(z).dag()*D(alpha).dag()


def cat_even(alpha): #pure state
    return (coherent(alpha) + coherent(-alpha)).unit()
def cat_odd(alpha): #pure state
    return (coherent(alpha) - coherent(-alpha)).unit()
def cat_even_dm(alpha):
    return ket2dm(cat_even(alpha))
def cat_odd_dm(alpha):
    return ket2dm(cat_odd(alpha))

def thermal_cat_even(alpha,avg_n):#create cat states thermalised at some temperature T (that gives rise/ is equivalent to specifying their avg_n)
    kappa = 0.01 # Coupling to the thermal bath
    tlist = np.linspace(0, 10, 100)  # Time over which to evolve the system
    # Define collapse operators for interaction with a thermal bath
    c_ops = [np.sqrt(kappa * (1 + avg_n)) * a, np.sqrt(kappa * avg_n) * adag]
    # Time evolution of the system
    result = mesolve(adag* a, cat_even_dm(alpha), tlist, c_ops, [])
    # The final state after thermalization
    return result.states[-1]

def thermal_cat_odd(alpha,avg_n):#create cat states thermalised at some temperature T (that gives rise/ is equivalent to specifying their avg_n)
    kappa = 0.01  # Coupling to the thermal bath
    tlist = np.linspace(0, 10, 100)  # Time over which to evolve the system
    # Define collapse operators for interaction with a thermal bath
    c_ops = [np.sqrt(kappa * (1 + avg_n)) * a, np.sqrt(kappa * avg_n) * adag]
    # Time evolution of the system
    result = mesolve(adag* a, cat_odd_dm(alpha), tlist, c_ops, [])
    # The final state after thermalization
    return result.states[-1]

z_vec=np.linspace(0.1,1,50)
alpha_vec=np.linspace(0.01,4,50)
avg_n=0.25
ref_state=cv_state(0,1)
ref_state_dm=cv_state_thermal_dm(0,1,avg_n)
snr_vec=[]
snr_vec_1pha=[]
snr_vec_cat_odd=[]
snr_vec_cat_even=[]
for alpha in alpha_vec:
    snr_vec+=[max([np.real(snr_for_mixed_state(cv_state_thermal_dm(alpha,z,avg_n),ref_state_dm)) for z in z_vec])]
    snr_vec_1pha+=[max([np.real(snr_for_mixed_state(adag*cv_state_thermal_dm(alpha,z,avg_n)*a,ref_state_dm)) for z in z_vec])]
    snr_vec_cat_odd+=[snr_for_mixed_state(thermal_cat_odd(alpha,avg_n),ref_state_dm)]
    snr_vec_cat_even+=[snr_for_mixed_state(thermal_cat_even(alpha,avg_n),ref_state_dm)]
plt.plot(alpha_vec,snr_vec,'b')
plt.plot(alpha_vec,snr_vec_1pha,'r')
plt.plot(alpha_vec,snr_vec_cat_odd,'g')
plt.plot(alpha_vec,snr_vec_cat_even,'y')
plt.show()
