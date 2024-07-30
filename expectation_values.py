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
def expvalN(sigma, dispvector): #input a 2N x 2N np.array of parameters for M and a displacement vector of means of length 2N
    N=len(sigma)//2
    #print('sigma',np.round(sigma,3))

    #now let's calculate the tr(prod(a's)rho). The amount of ladder operators is twice the number of modes (2N)
    #the amount of destruction operators is N, and the amount of creation is also N
    sum=0
    for i in range(1,N+1):
      ops=['adag','a']
      modes=[i,i]
      sum+=expectationvalue_with_disp(sigma,dispvector,ops,modes)
    return sum 

def ergotropy(sigma,sigma0,dispvector):  
    N=len(sigma)//2
    return expvalN(sigma,dispvector)-expvalN(sigma0,[0]*(2*N))

#Expectation value of N^2

def N2(sigma,dispvector): #dispersion of number operator on gaussian state (rho0)
    #We now compute exp(N^2):
    N=len(sigma)//2
    sum=0
    for i in range(1,N+1):
      ops= ['adag','a','adag','a']
      modes=[i,i,i,i]
      sum+=expectationvalue_with_disp(sigma,dispvector,ops,modes)
    for i in range(1,N+1):
      for j in range(i+1,N+1):
        ops= ['adag','a','adag','a']
        modes=[i,i,j,j]
        sum+=2*expectationvalue_with_disp(sigma,dispvector,ops,modes)
    return sum

def varianceN(sigma,dispvector):
    return  np.sqrt(N2(sigma,dispvector) - (expvalN(sigma,dispvector))**2) 

def std_dev(sigma,sigma0,displacement):
    N=len(sigma)//2
    return varianceN(sigma,displacement)-varianceN(sigma0,[0]*(2*N))

def SNR_gaussian(sigma,dispvector):
  N=len(sigma)//2
  return (expvalN(sigma,dispvector))/varianceN(sigma,dispvector)**2

def SNR_gaussian_extr(sigma,sigma0,dispvector):
  N=len(sigma)//2
  return ergotropy(sigma,sigma0,dispvector)/varianceN(sigma,dispvector)**2


#NON-GAUSSIAN STATE
#fist calculate the normalization value (will need to divide by it in every expectation value)

def K_ng(sigma, dispvector, nongaussian_ops):
    N= len(sigma)//2
    if nongaussian_ops==[]:
      return 1
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
    return expectationvalue_with_disp(sigma,dispvector,ops,modes)

#expectation value of N for the non-gaussian state
def expvalN_ng(sigma,dispvector,nongaussian_ops):
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
      sum+=expectationvalue_with_disp(sigma,dispvector,ops,modes)
    return (1/K_ng(sigma,dispvector,nongaussian_ops))*sum

def ergotropy_ng(sigma,sigma0,dispvector,nongaussian_ops):  
    N=len(sigma)//2
    return expvalN_ng(sigma,dispvector,nongaussian_ops)-expvalN(sigma0,[0]*(2*N))


#expectation value of N^2 for the non-gaussian state
def N2_ng(sigma,dispvector,nongaussian_ops):
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
        sum+=expectationvalue_with_disp(sigma,dispvector,ops,modes)
    return (1/K_ng(sigma,dispvector,nongaussian_ops))*sum

def varianceN_ng(sigma,dispvector,nongaussian_ops):
  return  np.sqrt(N2_ng(sigma,dispvector,nongaussian_ops) - (expvalN_ng(sigma,dispvector, nongaussian_ops))**2)

def std_dev_ng(sigma,sigma0,displacement,nongaussian_ops):
    N=len(sigma)//2
    return varianceN_ng(sigma,displacement,nongaussian_ops)-varianceN(sigma0,[0]*(2*N))

def SNR_ng(sigma,dispvector,nongaussian_ops):
  N=len(sigma)//2
  return (expvalN_ng(sigma,dispvector,nongaussian_ops))/varianceN_ng(sigma,dispvector,nongaussian_ops)**2

def SNR_ng_extr(sigma,dispvector,nongaussian_ops,sigma0):
  N=len(sigma)//2
  return ergotropy_ng(sigma,sigma0,dispvector,nongaussian_ops)/varianceN_ng(sigma,dispvector,nongaussian_ops)**2

def antibunching_one_mode(sigma,dispvector,nongaussian_ops): #N=1 only
  N= len(sigma)//2
  sum1=0
  ops=['rho']
  modes=['rho']
  for item in nongaussian_ops:
    if item<0: #subtraction
      ops=['a']+ops+['adag']
    if item>0: #addition
      ops=['adag']+ops+['a']
    modes=[np.abs(item)]+ modes+[np.abs(item)]
  ops=['adag','adag','a','a']+ops
  modes=[1,1,1,1]+modes
  cut = ops.index('rho')
  ops= ops[cut+1:]+ops[:cut]
  modes= modes[cut+1:]+modes[:cut]
  #print(ops)
  #print(modes)
  sum1+=expectationvalue_with_disp(sigma,dispvector,ops,modes)

  sum2=0
  ops=['rho']
  modes=['rho']
  for item in nongaussian_ops:
    if item<0: #subtraction
      ops=['a']+ops+['adag']
    if item>0: #addition
      ops=['adag']+ops+['a']
    modes=[np.abs(item)]+ modes+[np.abs(item)]
  ops=['adag','a']+ops
  modes=[1,1]+modes
  cut = ops.index('rho')
  ops= ops[cut+1:]+ops[:cut]
  modes= modes[cut+1:]+modes[:cut]
  #print(ops)
  #print(modes)
  sum2+=expectationvalue_with_disp(sigma,dispvector,ops,modes)

  return sum1/(sum2**2)


def antibunching_two_mode(sigma,dispvector,nongaussian_ops): #N=2 only
  N= len(sigma)//2
  sum1=0
  ops=['rho']
  modes=['rho']
  for item in nongaussian_ops:
    if item<0: #subtraction
      ops=['a']+ops+['adag']
    if item>0: #addition
      ops=['adag']+ops+['a']
    modes=[np.abs(item)]+ modes+[np.abs(item)]
  ops=['adag','adag','a','a']+ops
  modes=[1,1,1,1]+modes
  cut = ops.index('rho')
  ops= ops[cut+1:]+ops[:cut]
  modes= modes[cut+1:]+modes[:cut]
  #print(ops)
  #print(modes)
  sum1+=expectationvalue_with_disp(sigma,dispvector,ops,modes)
    
  sum2=0
  ops=['rho']
  modes=['rho']
  for item in nongaussian_ops:
    if item<0: #subtraction
      ops=['a']+ops+['adag']
    if item>0: #addition
      ops=['adag']+ops+['a']
    modes=[np.abs(item)]+ modes+[np.abs(item)]
  ops=['adag','adag','a','a']+ops
  modes=[2,2,2,2]+modes
  cut = ops.index('rho')
  ops= ops[cut+1:]+ops[:cut]
  modes= modes[cut+1:]+modes[:cut]
  #print(ops)
  #print(modes)
  sum2+=expectationvalue_with_disp(sigma,dispvector,ops,modes)

  sum3=0
  ops=['rho']
  modes=['rho']
  for item in nongaussian_ops:
    if item<0: #subtraction
      ops=['a']+ops+['adag']
    if item>0: #addition
      ops=['adag']+ops+['a']
    modes=[np.abs(item)]+ modes+[np.abs(item)]
  ops=['adag','adag','a','a']+ops
  modes=[1,2,1,2]+modes
  cut = ops.index('rho')
  ops= ops[cut+1:]+ops[:cut]
  modes= modes[cut+1:]+modes[:cut]
  #print(ops)
  #print(modes)
  sum3+=expectationvalue_with_disp(sigma,dispvector,ops,modes)

  return ((sum1+sum2)+2*sum3)/(expvalN_ng(sigma,dispvector,nongaussian_ops))**2


def SV(sigma,nongaussian_ops): #N=2 only
  N= len(sigma)//2
  sum1=0
  ops=['rho']
  modes=['rho']
  for item in nongaussian_ops:
    if item<0: #subtraction
      ops=['a']+ops+['adag']
    if item>0: #addition
      ops=['adag']+ops+['a']
    modes=[np.abs(item)]+ modes+[np.abs(item)]
  ops=['adag','a']+ops
  modes=[1,1]+modes
  cut = ops.index('rho')
  ops= ops[cut+1:]+ops[:cut]
  modes= modes[cut+1:]+modes[:cut]
  #print(ops)
  #print(modes)
  sum1+=expectationvalue(sigma,ops,modes)/K_ng(sigma,nongaussian_ops)
    
  sum2=0
  ops=['rho']
  modes=['rho']
  for item in nongaussian_ops:
    if item<0: #subtraction
      ops=['a']+ops+['adag']
    if item>0: #addition
      ops=['adag']+ops+['a']
    modes=[np.abs(item)]+ modes+[np.abs(item)]
  ops=['adag','a']+ops
  modes=[2,2]+modes
  cut = ops.index('rho')
  ops= ops[cut+1:]+ops[:cut]
  modes= modes[cut+1:]+modes[:cut]
  #print(ops)
  #print(modes)
  sum2+=expectationvalue(sigma,ops,modes)/K_ng(sigma,nongaussian_ops)

  sum3=0
  ops=['rho']
  modes=['rho']
  for item in nongaussian_ops:
    if item<0: #subtraction
      ops=['a']+ops+['adag']
    if item>0: #addition
      ops=['adag']+ops+['a']
    modes=[np.abs(item)]+ modes+[np.abs(item)]
  ops=['adag','adag']+ops
  modes=[1,2]+modes
  cut = ops.index('rho')
  ops= ops[cut+1:]+ops[:cut]
  modes= modes[cut+1:]+modes[:cut]
  #print(ops)
  #print(modes)
  sum3+=expectationvalue(sigma,ops,modes)/K_ng(sigma,nongaussian_ops)

  sum4=0
  ops=['rho']
  modes=['rho']
  for item in nongaussian_ops:
    if item<0: #subtraction
      ops=['a']+ops+['adag']
    if item>0: #addition
      ops=['adag']+ops+['a']
    modes=[np.abs(item)]+ modes+[np.abs(item)]
  ops=['a','a']+ops
  modes=[1,2]+modes
  cut = ops.index('rho')
  ops= ops[cut+1:]+ops[:cut]
  modes= modes[cut+1:]+modes[:cut]
  #print(ops)
  #print(modes)
  sum4+=expectationvalue(sigma,ops,modes)/K_ng(sigma,nongaussian_ops)

  return ((sum1*sum2)-(sum3*sum4))




#EXPECTATION VALUES FOR ODD CAT STATE (as a function of displacement)  for N=1 mode
def N(displacement,theta): #normalization factor of the superposition. This is for a general superposition of angle theta
  alpha=np.sqrt(displacement[0]**2+displacement[1]**2)
  return np.sqrt(2+2*np.cos(theta)*np.exp(-2*alpha**2))


#the rest is just for a cat odd state

def expvalN_odd_cat(displacement):
#alpha is the modulus of the displacement vector 
  alpha=np.sqrt(displacement[0]**2+displacement[1]**2)
  norm=N(displacement,np.pi)
  return alpha**2*(2+2*np.exp(-2*alpha**2))/norm**2

def N2_odd_cat(displacement):
  alpha=np.sqrt(displacement[0]**2+displacement[1]**2)
  norm=N(displacement,np.pi)
  return (2/norm**2)*(alpha**4+alpha**2-(alpha**4-alpha**2)*np.exp(-2*alpha**2))

def g_cat_odd(displacement):
  return (N2_odd_cat(displacement)-expvalN_odd_cat(displacement)**2)/(expvalN_odd_cat(displacement))**2 - 1/expvalN_odd_cat(displacement) + 1

def snr_cat_odd(displacement):
  return expvalN_odd_cat(displacement)/(N2_odd_cat(displacement)-expvalN_odd_cat(displacement)**2)**2
   #snr for cat state at zero temperature


