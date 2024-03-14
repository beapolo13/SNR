from sympy import symbols, cos, sin, Matrix, simplify
import numpy as np
from numpy import transpose, real, sqrt, sin, cos, linalg, cosh, sinh
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

def V_tms_sym(z,x, phi, params): 
    N=len(z)
    if ordering == 'xpxp': #this only works for N=2 so far
      B=np.array([[cos(x),0,sin(x),0],[0,cos(x),0,sin(x)],[-sin(x),0,cos(x),0],[0,-sin(x),0,cos(x)]])
      P1=np.array([[cos(phi1),sin(phi1)], [-sin(phi1), cos(phi1)]])
      P2=np.array([[cos(phi2),sin(phi2)], [-sin(phi2), cos(phi2)]])
      P= np.block([[P1,np.zeros((N,N))],[np.zeros((N,N)), P2]])
    elif ordering == 'xxpp':
      #beamsplitter
      B_total=Matrix.eye(2*N)
      #print('x:',cos(x),sin(x))
      index=0
      for i in range(N):
          for j in range(i+1,N):
              B=Matrix.eye(2*N)
              B[i,i]=B[j,j]=B[N+i,N+i]=B[N+j,N+j]= cos(x[index])
              B[i,j]=B[N+i,N+j]= sin(x[index])
              B[j,i]=B[N+j,N+i]= -sin(x[index])
              index+=1
              B_total=B_total@B
      #print('B',np.round(B_total,2))
      #dephasing
      P= Matrix.eye(2*N)
      for i in range(N):
        P[i,i]=P[i+N,i+N]= cos(phi[i])
        P[i,i+N]=sin(phi[i])
        P[i+N,i]=-sin(phi[i])
      #print('P',P)
    S=sq(z)
    #print('S',S)
    if params is not None:
      O1= Orth(params)
      result= P @ B_total @ O1 @ S @ transpose(O1) @ transpose(B_total) @ transpose(P)
    else:
      result= P @ B_total @ S @ transpose(B_total) @ transpose(P)
    return simplify(result)

def id1_sym(sigma,l,k): #function to compute Tr(a^dag_l a^dag_k rho)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    return (1/4)*simplify(sigma[l-1,k-1]-sigma[l+N-1,k+N-1]-1j*(sigma[l-1,k+N-1]+sigma[l+N-1,k-1]))

def id2_sym(sigma,l,k): #function to compute Tr(a_l a_k rho)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    return np.conjugate(id1_sym(sigma,l,k))

def id3_sym(sigma,l,k): #function to compute Tr(a^dag_l a_k rho)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    delta=0
    if l==k:
        delta+=1
    return (1/4)*simplify(sigma[l-1,k-1]+sigma[l+N-1,k+N-1]+1j*(sigma[l-1,k+N-1]-sigma[l+N-1,k-1])-2*delta)


def id4_sym(sigma,l,k):  #function to compute Tr(a^_l a^dag_k rho)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    delta2=0
    if l==k:
        delta2+=1
    return id3_sym(sigma,l,k)+delta2

#function to compute traces (defined in the paper)
def trace_func_sym(sigma,l,k,case):
    if case==1:
        return id1_sym(sigma,l,k)
    elif case==2:
        return id2_sym(sigma,l,k)
    elif case==3:
        return id3_sym(sigma,l,k)
    elif case==4:
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
    #print('sigma',np.round(sigma,3))
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

    return simplify(sum/np.abs(K))

#Expectation value of N^2

def N2_sym(sigma): #dispersion of number operator on gaussian state (rho0)
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

covmat= V_tms_sym(z_values,theta_values,phi_values, params=None)

print('N=',expvalN_sym(covmat))
print('')
print('N2=',N2_sym(covmat))
print('')
print('deltaN=',varianceN_sym(covmat))
print('')
print('SNR=',SNR_gaussian_sym(covmat))
print('')
ratio= SNR_gaussian_sym(covmat)
diff_z1 = simplify(ratio.diff(z1))
print('diff z1=',diff_z1)
print('')
diff_z2 = simplify(ratio.diff(z2))
print('diff z2=',diff_z2)
print('')
diff_x1 = simplify(ratio.diff(x1))
print('diff x1=',diff_x1)
print('')
diff_phi1 = simplify(ratio.diff(phi1))
print('diff phi1=',diff_phi1)
print('')
diff_phi2 = simplify(ratio.diff(phi2))
print('diff phi2=',diff_phi2)
print('')
jacobian_matrix = ratio.jacobian([z1, z2, x1,phi1,phi2])
print('jac=',jacobian_matrix)


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
    return simplify((1/K_ng_sym(sigma,nongaussian_ops))*sum)

#expectation value of N^2 for the non-gaussian state
def N2_ng_sym(sigma,nongaussian_ops):
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
    return simplify((1/K_ng_sym(sigma,nongaussian_ops))*sum)

def varianceN_ng_sym(sigma,nongaussian_ops):
    return  simplify(sqrt(N2_ng_sym(sigma,nongaussian_ops) - (expvalN_ng_sym(sigma,nongaussian_ops))**2))

def SNR_ng_sym(sigma,nongaussian_ops):
  return simplify(expvalN_ng_sym(sigma,nongaussian_ops)/varianceN_ng_sym(sigma,nongaussian_ops))


print('N=',expvalN_sym(covmat))
print('')
print('N2=',N2_sym(covmat))
print('')
print('deltaN=',varianceN_sym(covmat))
print('')
print('SNR=',SNR_gaussian_sym(covmat))
print('')
ratio= SNR_gaussian_sym(covmat)
diff_z1 = simplify(ratio.diff(z1))
print('diff z1=',diff_z1)
print('')
diff_z2 = simplify(ratio.diff(z2))
print('diff z2=',diff_z2)
print('')
diff_x1 = simplify(ratio.diff(x1))
print('diff x1=',diff_x1)
print('')
diff_phi1 = simplify(ratio.diff(phi1))
print('diff phi1=',diff_phi1)
print('')
diff_phi2 = simplify(ratio.diff(phi2))
print('diff phi2=',diff_phi2)
print('')
jacobian_matrix = ratio.jacobian([z1, z2, x1,phi1,phi2])
print('jac=',jacobian_matrix)