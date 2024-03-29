import sympy as sym
from sympy import symbols, Matrix, simplify
import numpy as np
from numpy import transpose, real, sqrt,linalg, cosh, sinh
import scipy
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import optimize
from scipy.optimize import minimize
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from pprint import pprint
from scipy.linalg import block_diag
import os
from mpl_toolkits.mplot3d import Axes3D
from covariance_matrix import *
from utils import * 


def V_tms_sym(z,x, phi, params, ordering='xxpp'): 
    N=len(z)
    #beamsplitter
    B_total=Matrix.eye(2*N)
    #print('x:',cos(x),sin(x))
    index=0
    for i in range(N):
        for j in range(i+1,N):
            B=Matrix.eye(2*N)
            B[i,i]=B[j,j]=B[N+i,N+i]=B[N+j,N+j]= sym.cos(x[index])
            B[i,j]=B[N+i,N+j]= sym.sin(x[index])
            B[j,i]=B[N+j,N+i]= -sym.sin(x[index])
            index+=1
            B_total=B_total@B
    #print('B',np.round(B_total,2))
    #dephasing
    P= Matrix.eye(2*N)
    for i in range(N):
      P[i,i]=P[i+N,i+N]= sym.cos(phi[i])
      P[i,i+N]=sym.sin(phi[i])
      P[i+N,i]=-sym.sin(phi[i])
    #print('P',P)
    S=sq(z)
    #print('S',S)
    if params is not None:
      O1= Orth(params)
      result= P @ B_total @ O1 @ S @ transpose(O1) @ transpose(B_total)  @ transpose(P)
    else:
      result= P @ B_total @ S @ transpose(B_total) @transpose(P)

    if ordering == 'xpxp': 
      return convention_switch(simplify(result),'xxpp',format='String')
    else:
      return simplify(result)

def id1_sym(sigma,l,k,ordering='xxpp'): #function to compute Tr(a^dag_l a^dag_k rho)
    N= int(np.sqrt(len(sigma))//2)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    return (1/4)*simplify(sigma[l-1,k-1]-sigma[l+N-1,k+N-1]-1j*(sigma[l-1,k+N-1]+sigma[l+N-1,k-1]))

def id2_sym(sigma,l,k,ordering='xxpp'): #function to compute Tr(a_l a_k rho)
    N= int(np.sqrt(len(sigma))//2)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    return np.conjugate(id1_sym(sigma,l,k))

def id3_sym(sigma,l,k,ordering='xxpp'): #function to compute Tr(a^dag_l a_k rho)
    N= int(np.sqrt(len(sigma))//2)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    delta=0
    if l==k:
        delta+=1
    return (1/4)*simplify(sigma[l-1,k-1]+sigma[l+N-1,k+N-1]+1j*(sigma[l-1,k+N-1]-sigma[l+N-1,k-1])-2*delta)


def id4_sym(sigma,l,k,ordering='xxpp'):  #function to compute Tr(a^_l a^dag_k rho)
    N= int(np.sqrt(len(sigma))//2)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    delta2=0
    if l==k:
        delta2+=1
    return id3_sym(sigma,l,k)+delta2

#function to compute traces (defined in the paper)
def trace_func_sym(sigma,l,k,case):
    if case==1:
        #print('id1',l,k)
        return id1_sym(sigma,l,k)
    elif case==2:
        #print('id2',l,k)
        return id2_sym(sigma,l,k)
    elif case==3:
        #print('id3',l,k)
        return id3_sym(sigma,l,k)
    elif case==4:
        #print('id4',l,k)
        return id4_sym(sigma,l,k)

def expectationvalue_sym(covmat,operatorlist,modeslist):
    indices=[i for i in range(len(operatorlist))]
    trace=0
    #print('Perfect matchings',perfect_matchings(indices))
    for matching in p_matchings(indices):
        #print('matching:',matching)
        factor=1
        for pair in matching:
            #print(pair)
            l,k= modeslist[pair[0]],modeslist[pair[1]]
            #print('l,k:',l,k)
            if operatorlist[pair[0]]=='adag' and operatorlist[pair[1]]=='adag':
                case=1
            elif operatorlist[pair[0]]=='a' and operatorlist[pair[1]]=='a':
                case=2
            elif operatorlist[pair[0]]=='adag' and operatorlist[pair[1]]=='a':
                case=3
            elif operatorlist[pair[0]]=='a' and operatorlist[pair[1]]=='adag':
                case=4
            #print('case',case)
            factor*=trace_func_sym(covmat,l,k,case)
        trace+=factor
    return trace


#GAUSSIAN STATE (symbolic/analytical expression)
from sympy import sqrt

#Expectation value of N
def expvalN_sym(sigma): #input a 2N x 2N np.array of parameters for M
    N = int(np.sqrt(len(sigma))//2)
    #print(N)
    #print('sigma',sigma)
    K=0
    for i in range(2*N):
        K+=sigma[i,i]
    #print('K=',K) #K is tr(sigma)

    #now let's calculate the tr(prod(a's)rho). The amount of ladder operators is twice the number of modes (2N)
    #the amount of destruction operators is N, and the amount of creation is also N
    sum=0
    for i in range(1,N+1):
      ops=['adag','a']
      modes=[i,i]
      sum+=expectationvalue_sym(sigma,ops,modes)

    return simplify(sum/K)

#Expectation value of N^2

def N2_sym(sigma): #dispersion of number operator on gaussian state (rho0)
    N = int(np.sqrt(len(sigma))//2)
    #We now compute exp(N^2):
    K=0
    for i in range(2*N):
        K+=sigma[i,i]
    #print('K=',K) #K is tr(sigma)
    sum=0
    for i in range(1,N+1):
      ops= ['adag','a','adag','a']
      modes=[i,i,i,i]
      sum+=expectationvalue_sym(sigma,ops,modes)
    for i in range(1,N+1):
      for j in range(i+1,N+1):
        ops= ['adag','a','adag','a']
        modes=[i,i,j,j]
        sum+=2*expectationvalue_sym(sigma,ops,modes)
    return simplify((1/K)*sum)


def varianceN_sym(sigma):
    return  simplify(sqrt(N2_sym(sigma) - (expvalN_sym(sigma))**2))

def SNR_gaussian_sym(sigma):
  return simplify(expvalN_sym(sigma)/varianceN_sym(sigma))


#Non gaussian state SYMBOLIC expectation values
def K_ng_sym(sigma, nongaussian_ops):
    ops=['rho']
    modes=['rho']
    for item in nongaussian_ops:
      if item<0: #subtraction
        ops=['a']+ops+['adag']
      if item>0: #addition
        ops=['adag']+ops+['a']
      modes=[np.abs(item)]+ modes+[np.abs(item)]
    cut = ops.index('rho')
    ops= ops[cut+1:]+ops[:cut]
    modes= modes[cut+1:]+modes[:cut]
    #print(ops)
    #print(modes)
    return expectationvalue_sym(sigma,ops,modes)

#expectation value of N for the non-gaussian state
def expvalN_ng_sym(sigma,nongaussian_ops):
    N = int(np.sqrt(len(sigma))//2)
    #construct the trace we want to calculate ((adag, a) per mode + non-gaussian cov matrix)
    sum=0
    for i in range(1,N+1):
      ops=['rho']
      modes=['rho']
      for item in nongaussian_ops:
        if item<0: #subtraction
          ops=['a']+ops+['adag']
        if item>0: #addition
          ops=['adag']+ops+['a']
        modes=[np.abs(item)]+ modes+[np.abs(item)]
      ops=['adag','a']+ops
      modes=[i,i]+modes
      cut = ops.index('rho')
      ops= ops[cut+1:]+ops[:cut]
      modes= modes[cut+1:]+modes[:cut]
      #print(ops)
      #print(modes)
      sum+=expectationvalue_sym(sigma,ops,modes)
    return (1/K_ng_sym(sigma,nongaussian_ops))*sum

#expectation value of N^2 for the non-gaussian state
def N2_ng_sym(sigma,nongaussian_ops):
    N = int(np.sqrt(len(sigma))//2)
    sum=0
    for i in range(1,N+1):
      for j in range(1,N+1):
        ops=['rho']
        modes=['rho']
        for item in nongaussian_ops:
          if item<0: #subtraction
            ops=['a']+ops+['adag']
          if item>0: #addition
            ops=['adag']+ops+['a']
          modes=[np.abs(item)]+ modes+[np.abs(item)]
        ops=['adag','a','adag','a']+ops
        modes=[i,i,j,j]+modes
        cut = ops.index('rho')
        ops= ops[cut+1:]+ops[:cut]
        modes= modes[cut+1:]+modes[:cut]
        #print(ops)
        #print(modes)
        sum+=expectationvalue_sym(sigma,ops,modes)
    return (1/K_ng_sym(sigma,nongaussian_ops))*sum

def varianceN_ng_sym(sigma,nongaussian_ops):
    return  sqrt(N2_ng_sym(sigma,nongaussian_ops) - (expvalN_ng_sym(sigma,nongaussian_ops))**2)

def SNR_ng_sym(sigma,nongaussian_ops):
  return expvalN_ng_sym(sigma,nongaussian_ops)/varianceN_ng_sym(sigma,nongaussian_ops)

#Print results of analytical calculations:

#DON'T FORGET TO indicate symbolic variables are real (real=True)



def analytical_results(z1,z2,x1,phi1,phi2): #so far only for N =2
  z_values = [z1,z2]  #1:N+1
  theta_values = [x1] #N*(N-1)//2 +1
  phi_values = [phi1,phi2]  #1:N+1
  covmat= V_tms_sym(z_values,theta_values,phi_values, params=None)
  N_gauss=expvalN_sym(covmat)
  N2_gauss=N2_sym(covmat)
  delta_gauss=varianceN_sym(covmat)
  ratio=SNR_gaussian_sym(covmat)
  print('covariance matrix:',covmat)
  print('')
  print('N=',N_gauss)
  print('')
  print('N2=',N2_gauss)
  print('')
  print('deltaN=',delta_gauss)
  print('')
  print('SNR=',ratio)
  print('')
  diff_z1 = ratio.diff(z1)
  print('diff z1=',diff_z1)
  print('')
  #diff_z2 = ratio.diff(z2)
  #print('diff z2=',diff_z2)
  print('')
  diff_x1 = ratio.diff(x1)
  print('diff x1=',diff_x1)
  print('')
  diff_phi1 = ratio.diff(phi1)
  print('diff phi1=',diff_phi1)
  print('')
  diff_phi2 = ratio.diff(phi2)
  print('diff phi2=',diff_phi2)
  print('')

   # Solve the expression for one variable in terms of the other 
  solutions = sym.solve(diff_z1, z1)

# Plot one variable as a function of the other
# Let's say we plot x1 as a function of z1
  x1_values = np.arange(0, 2*np.pi)  # Adjust the range as needed
  for s in solutions:
    z1_values = [s.evalf(subs={x1: x}) for x in x1_values]
    plt.plot(x1_values, z1_values)
  plt.xlabel('x1')
  plt.ylabel('z1')
  plt.title('Plot of z1 as a function of x1')
  plt.grid(True)
  plt.show()


  print('NON GAUSSIAN')
  nongaussian_ops=[1]
  N_ng=expvalN_ng_sym(covmat,nongaussian_ops)
  N2_ng=N2_ng_sym(covmat,nongaussian_ops)
  delta_ng=varianceN_ng_sym(covmat,nongaussian_ops)
  ratio_ng=SNR_ng_sym(covmat,nongaussian_ops)
  print('N=',N_ng)
  print('')
  print('N2=',N2_ng)
  print('')
  print('deltaN=',delta_ng)
  print('')
  print('SNR=',ratio_ng)
  print('')
  diff_z1_ng = ratio_ng.diff(z1)
  print('diff z1=',diff_z1_ng)
  print('')
  #diff_z2_ng = ratio_ng.diff(z2)
  #print('diff z2=',diff_z2_ng)
  print('')
  diff_x1_ng = ratio_ng.diff(x1)
  print('diff x1=',diff_x1_ng)
  print('')
  diff_phi1_ng = ratio_ng.diff(phi1)
  print('diff phi1=',diff_phi1_ng)
  print('')
  diff_phi2_ng = ratio_ng.diff(phi2)
  print('diff phi2=',diff_phi2_ng)

  # Solve the expression for one variable in terms of the other
  solutions = sym.solve(diff_z1_ng, z1)

# Plot one variable as a function of the other
# Let's say we plot x1 as a function of z1
  x1_values = np.arange(0, 2*np.pi)  # Adjust the range as needed
  for s in solutions:
    z1_values = [s.evalf(subs={x1: x}) for x in x1_values]
    plt.plot(x1_values, z1_values)
  plt.xlabel('x1')
  plt.ylabel('z1')
  plt.title('Plot of z1 as a function of x1')
  plt.grid(True)
  plt.show()
  return


N=2
z1,z2,x1 =symbols('z1,z2,x1',real=True, RealNumber=True)
phi1,phi2 =symbols('phi1,phi2',zero=True)
z2=1/z1
z_values = [z1,z2]  #1:N+1
theta_values = [x1] #N*(N-1)//2 +1
phi_values = [phi1,phi2]  #1:N+1

sigma=V_tms_sym(z_values,theta_values, phi_values, params=None, ordering='xxpp')
#print(sigma)
#print(simplify(N2_sym(sigma)))
print(analytical_results(z1,z2,x1,phi1,phi2))



