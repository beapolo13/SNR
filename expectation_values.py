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

from utils import *


#GAUSSIAN STATE

#Expectation value of N
def expvalN(sigma): #input a 2N x 2N np.array of parameters for M
    N=len(sigma)//2
    #print('sigma',np.round(sigma,3))
    K=0
    for i in range(2*N):
        K+=sigma[i][i]
    #print('K=',K) #K is tr(sigma)

    #now let's calculate the tr(prod(a's)rho). The amount of ladder operators is twice the number of modes (2N)
    #the amount of destruction operators is N, and the amount of creation is also N
    sum=0
    for i in range(1,N+1):
      ops=['adag','a']
      modes=[i,i]
      sum+=expectationvalue(sigma,ops,modes)

    return sum/np.abs(K)


#Expectation value of N^2

def N2(sigma): #dispersion of number operator on gaussian state (rho0)
    #We now compute exp(N^2):
    N=len(sigma)//2
    K=0
    for i in range(2*N):
        K+=sigma[i][i]
    #print('K=',K) #K is tr(sigma)
    sum=0
    for i in range(1,N+1):
      ops= ['adag','a','adag','a']
      modes=[i,i,i,i]
      sum+=expectationvalue(sigma,ops,modes)
    for i in range(1,N+1):
      for j in range(i+1,N+1):
        ops= ['adag','a','adag','a']
        modes=[i,i,j,j]
        sum+=2*expectationvalue(sigma,ops,modes)
    return (1/K)*sum


def varianceN(sigma):
    return  np.sqrt(N2(sigma) - (expvalN(sigma))**2)

def SNR_gaussian(sigma):
  return expvalN(sigma)/varianceN(sigma)


#NON-GAUSSIAN STATE

#fist calculate the normalization value (will need to divide by it in every expectation value)


def K_ng(sigma, nongaussian_ops):
    N= len(sigma)//2
    if nongaussian_ops==[]:
      K=0
      for i in range(2*N):
        K+=sigma[i][i]
      return K
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
    return expectationvalue(sigma,ops,modes)

#expectation value of N for the non-gaussian state
def expvalN_ng(sigma,nongaussian_ops):
    N= len(sigma)//2
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
      sum+=expectationvalue(sigma,ops,modes)
    return (1/K_ng(sigma,nongaussian_ops))*sum


#expectation value of N^2 for the non-gaussian state
def N2_ng(sigma,nongaussian_ops):
    N= len(sigma)//2
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
        sum+=expectationvalue(sigma,ops,modes)
    return (1/K_ng(sigma,nongaussian_ops))*sum

def varianceN_ng(sigma,nongaussian_ops):
  return  np.sqrt(N2_ng(sigma,nongaussian_ops) - (expvalN_ng(sigma,nongaussian_ops))**2)

def SNR_ng(sigma,nongaussian_ops):
  return expvalN_ng(sigma,nongaussian_ops)/varianceN_ng(sigma,nongaussian_ops)

