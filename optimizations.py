import numpy as np
from numpy import transpose, real, sqrt, sin, cos, linalg, cosh, sinh
import scipy
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import optimize
from scipy.optimize import minimize, Bounds
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from pprint import pprint
from scipy.linalg import block_diag
import os

from utils import *
from expectation_values import *

def global_optimization():  #global optimizer: too slow even for fixed squeezing
  for N in range(2,6):
    print('Number of modes', N)
    theta=0
    modesBS=[1,2]
    phi=[0]*N
    free_pars=2*np.pi*np.random.rand(4*N**2) #z is included in this parameter list
    def cost(free_pars):
        return np.real(1/SNR_ng(V_tms([0.5]*N,theta,modesBS,phi,free_pars),nongaussian_ops)) #we take the inverse of the SNR to minimize
    #def constraint_function(free_pars):
        #return N-sum(free_pars[:N])
    #constraint = {'type': 'eq', 'fun': constraint_function}
    start = time.time()
    bounds=[(-np.inf,np.inf) for _ in range(len(free_pars))]
    out=optimize.shgo(cost,bounds)
    print(out)
    end = time.time()
    print('Time taken to find maximum ratio', end - start)
    print('optimal ratio:',1/out.fun)
    print('optimal squeezing:',out.x[:N-1])
    print('')
  return

nongaussian_ops=[-1]  #-n for photon subtraction on mode n, or +n if its photon addition on mode n
n_max=10

#optimization over all parameters: z (con constraint en la suma total), x del BS, phi del dephasing y passive optics (params)
def optimization():
  ratio_vec=[]
  for N in range(2,n_max):
    print('Number of modes', N, 'Non-gaussian operations', nongaussian_ops)
    z=list(np.random.rand(N))
    theta=2*np.pi*np.random.rand((N*(N-1))//2)
    phi=2*np.pi*np.random.rand(N)
    params=2*np.pi*np.random.rand(N**2)
    free_pars=[]
    for i in range(len(z)):
      free_pars+=[z[i]]
    for i in range(len(theta)):
        free_pars+=[theta[i]]
    for i in range(len(phi)):
      free_pars+=[phi[i]]
    for i in range(len(params)):
      free_pars+=[params[i]]

    def cost(free_pars):
      return np.real(1/SNR_ng(V_tms(free_pars[:N],free_pars[N:(N*(N-1))//2],free_pars[(N*(N-1))//2:(N*(N-1))//2+2*N],free_pars[(N*(N-1))//2+2*N:]),nongaussian_ops)) #we take the inverse of the SNR to minimize
    def constraint_function(free_pars):
      return sum(free_pars[:N])- N
    def constraint_function_2(free_pars): #constraint of non-negativity and to avoid value to be too close to zero
        return free_pars[:N]-10**-3
    def constraint_function_3(free_pars): #constraint so that squeezing is not one
      return np.abs(free_pars[:N]-1)-0.1
    constraint = {'type': 'eq', 'fun': constraint_function}
    constraint2={'type': 'ineq', 'fun': constraint_function_2}
    constraint3={'type': 'ineq', 'fun': constraint_function_3}
    start = time.time()
    out=minimize(cost,free_pars, constraints=[constraint,constraint2, constraint3])
    ratio_vec+=[1/out.fun]
    print(out)
    end = time.time()
    print('Time taken to find maximum ratio', end - start)
    print('optimal ratio:',1/out.fun)
    print('optimal squeezing:',out.x[:N])
    print('')
    
  plt.plot(np.arange(2,n_max),ratio_vec,'o')
  plt.show()
  return



#NOTA: En realidad, puesto que el BS y el dephasing son operaciones gaussianas, no hace falta optimizar sobre ellas, sino sobre la matriz de cov gaussiana genérica.
#Entonces: 1º optimizar y buscar el mejor ratio para una sigma gaussiana genérica (sólo optimizar sobre z y params).
#2º ver si ese ratio se puede mejorar haciendo operaciones no gaussianas sobre los modos (y ver qué operaciones lo favorecen)
#3º ver como escala esta evolución con N

def optimization_1(): #without constraints on the values of z (just the sum of them =N and non negativity)
  ratio_vec=[]
  for N in range(2,n_max):
    print('Number of modes', N, 'Non-gaussian operations', nongaussian_ops)
    theta=[0]*(N*(N-1))//2
    phi=[0]*N
    free_pars=2*np.pi*np.random.rand(N+N**2) #z is included in this parameter list
    def cost(free_pars):
        return np.real(1/SNR_ng(V_tms(free_pars[:N],theta,phi,free_pars[N:]),nongaussian_ops)) #we take the inverse of the SNR to minimize
    def non_negativity(free_pars):
        return free_pars[:N]-10**-3
    def constraint_function(free_pars):
        return np.abs(N-np.sum(free_pars[:N]))-10**-5
    nonneg= {'type': 'ineq', 'fun': non_negativity}
    constraint = {'type': 'eq', 'fun': constraint_function}
    print(constraint)
    start = time.time()
    out=minimize(cost,free_pars,constraints=[nonneg,constraint])
    ratio_vec+=[1/out.fun]
    print(out)
    end = time.time()
    print('Time taken to find maximum ratio', end - start)
    print('optimal ratio:',1/out.fun)
    print('optimal squeezing:',out.x[:N])
    print(N-sum(out.x[:N]))
    print('')
  plt.plot(np.arange(2,n_max),ratio_vec,'o')
  plt.show()
  return

def optimization_2():  #with constraints on the values of z (CANNOT BE TOO CLOSE TO 1)
  ratio_vec=[]
  for N in range(2,n_max):
    print('Number of modes', N, 'Non-gaussian operations', nongaussian_ops)
    theta=[0]*(N*(N-1))//2
    phi=[0]*N
    free_pars=2*np.pi*np.random.rand(N+N**2) #z is included in this parameter list
    def cost(free_pars):
        return np.real(1/SNR_ng(V_tms(free_pars[:N],theta,phi,free_pars[N:]),nongaussian_ops)) #we take the inverse of the SNR to minimize
    def constraint_function(free_pars):
        return N-sum(free_pars[:N])
    def constraint_function_2(free_pars): #constraint of non-negativity
      return free_pars[:N]-0.1
    def constraint_function_3(free_pars): #constraint so that squeezing is not one
      return np.abs(free_pars[:N]-1)-0.1
    constraint = {'type': 'eq', 'fun': constraint_function}
    constraint2={'type': 'ineq', 'fun': constraint_function_2}
    constraint3={'type': 'ineq', 'fun': constraint_function_3}
    start = time.time()
    out=minimize(cost,free_pars, constraints=[constraint,constraint2, constraint3])
    ratio_vec+=[1/out.fun]
    print(out)
    end = time.time()
    print('Time taken to find maximum ratio', end - start)
    print('optimal ratio:',1/out.fun)
    print('optimal squeezing:',out.x[:N])
    print('')
  plt.plot(np.arange(2,n_max),ratio_vec,'o')
  plt.show()
  return


#OPTIMIZATION VERSION 3: Only optimization on the passive optics parameters (squeezing z is fixed too)

def optimization_3():
  ratio_vec=[]
  z_vector=[[0.5,1.5],[0.5,1.7,0.8],[0.5,2,0.8,0.7],[0.5,2,0.1,0.6,1.8],[0.5, 2, 0.1, 0.2, 1.5, 1.7],[0.5, 2, 1.1, 0.2, 0.4, 1.3, 1.5],
           [0.5, 2, 1.1, 0.9, 0.4, 2.1, 0.6, 0.4],[0.5, 2, 1.1, 0.9, 0.5, 2.1, 0.6, 0.4,0.9]]

  for N in range(2,n_max):
    print('Number of modes', N, 'Non-gaussian operations', nongaussian_ops)
    z=z_vector[N-2]
    theta=[0]*(N*(N-1))//2
    phi=[0]*N
    free_pars=2*np.pi*np.random.rand(N**2) #z is included in this parameter list
    def cost(free_pars):
        return np.real(1/SNR_ng(V_tms(z,theta,phi,free_pars),nongaussian_ops)) #we take the inverse of the SNR to minimize
    start = time.time()
    out=minimize(cost,free_pars)
    ratio_vec+=[1/out.fun]
    print(out)
    end = time.time()
    print('Time taken to find maximum ratio', end - start)
    print('optimal ratio:',1/out.fun)
    print('')
  plt.plot(np.arange(2,n_max),ratio_vec,'o')
  plt.show()
  return


def optimization_4(K): #K is the maximum number of non-gaussian operations to perform. Fixed N=2
  nongaussian_ops=[]
  ratio_vec=[]
  N=2
  for i in range(K):
    print('Number of modes', N, 'Non-gaussian operations', nongaussian_ops)
    z=[0.5]
    theta=[0]
    phi=[0,0]
    params=None
    free_pars=[]
    for i in range(len(z)):
      free_pars+=[z[i]]
    for i in range(len(theta)):
      free_pars+=[theta[i]]
    for i in range(len(phi)):
      free_pars+=[phi[i]] 
    def cost(free_pars):
        return np.real(1/SNR_ng(V_tms([free_pars[0],1/free_pars[0]],free_pars[1],free_pars[2:],None),nongaussian_ops)) #we take the inverse of the SNR to minimize
    bounds_opt = Bounds(0.001, 0.97)
    start = time.time()
    out=minimize(cost,free_pars, bounds=bounds_opt)
    ratio_vec+=[1/out.fun]
    print(out)
    end = time.time()
    print('Time taken to find maximum ratio', end - start)
    print('optimal ratio:',1/out.fun)
    print('')
    nongaussian_ops+=[-1]
  plt.plot(np.arange(0,K),ratio_vec,'o')
  plt.show()
  

  return


#OPTIMIZATION ON NUMBER OF MODES
#now we only optimize in the passive optics operations that we know how to 'physically implement' (z,bs,ps)
def optimization_5(nongaussian_ops, n_max):
  ratios=[[0]*(n_max-2)]*len(nongaussian_ops)
  T=0.25
  for j in range(len(nongaussian_ops)):
    for N in range(2,n_max):
      print('Number of modes', N, 'Non-gaussian operations', nongaussian_ops[j])
      sigma0=V_thermal([T]*N,[1]*N,[0]*((N*(N-1))//2),[0]*N,params=None)
      z=[0.5]*N
      theta=[0.125]*((N*(N-1))//2)
      phi=[0]*N
      params=None
      free_pars=[]
      for i in range(len(z)):
          free_pars+=[z[i]]
      for i in range(len(theta)):
          free_pars+=[theta[i]]
      for i in range(len(phi)):
        free_pars+=[phi[i]] 
      def cost(free_pars):
          return np.real(1/SNR_ng_extr(V_thermal([T]*N,free_pars[:N],2*np.pi*free_pars[N:(N*(N-1))//2+N],2*np.pi*free_pars[(N*(N-1))//2+N:],params),nongaussian_ops[j],sigma0)) #we take the inverse of the SNR to minimize
      bounds_opt = Bounds([0.25]*N+[0]*(len(free_pars)-N), [0.75]*N+[1]*(len(free_pars)-N))
      start = time.time()
      out=minimize(cost,free_pars,bounds=bounds_opt, method='L-BFGS-B')
      ratios[j][N-2]=1/out.fun
      print(out)
      end = time.time()
      print('Time taken to find maximum ratio', end - start)
      print('optimal ratio:',1/out.fun)
      print('')
    plt.plot(np.arange(2,n_max),ratios[j],linestyle='dashdot', marker='o')
  plt.title('Evolution of maximum SNR with number of modes N')
  plt.legend(nongaussian_ops)
  plt.savefig('opti.png')
  plt.show()
  return


print(optimization_5([[],[-1],[-1,-1],[-1,-1,-1]],6))
#print(optimization_5([1],7))
#print(optimization_4(6))
