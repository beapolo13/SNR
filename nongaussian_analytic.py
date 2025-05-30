#Here we compute analytical values of N, \DeltaN and \Gamma for a Gaussian non-displaced state (one mode) that undergoes successive photon subtractions of additions


import numpy as np
from numpy import transpose, real, sqrt, sin, cos, linalg, cosh, sinh, diag
import sympy as sp
import math
from math import factorial
from sympy import symbols, Matrix, simplify, exp, sqrt, tanh, diag, cos, sin, coth
import scipy
import matplotlib.pyplot as plt
import random
import itertools
from itertools import combinations
from scipy import optimize
from scipy.optimize import minimize, fsolve, NonlinearConstraint, shgo, differential_evolution
import time
import sys
from scipy.special import factorial, binom
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from pprint import pprint
from scipy.linalg import block_diag
import os
from mpl_toolkits.mplot3d import Axes3D
import winsound
import pickle

def I1(sigma):
    return (1/4)*(sigma[0,0]-sigma[1,1]-1j*(sigma[0,1]+sigma[1,0]))

def I2(sigma):
    return np.conjugate(I1(sigma))

def I3(sigma): 
    delta=1
    return (1/4)*(sigma[0,0]+sigma[1,1]+1j*(sigma[0,1]-sigma[1,0])-2*delta)

def I4(sigma):  #function to compute Tr(a^_l a^dag_k rho)
    delta2=1
    return np.conjugate(I3(sigma))+delta2


def b(r,s): #function that computes the number of different possible matchings of r creation and r annihilation ops, such that s of the pairs are inhomogeneous (adagger a / a adagger)
    return factorial(s)* (binom(r,s))**2 * (factorial(r-s)/(2**((r-s)/2)*factorial((r-s)/2)))**2

def f(sigma,r, operation): 
    i1 = I1(sigma)
    i2 = I2(sigma)
    i3 = I3(sigma)
    i4 = I4(sigma)
    result=0
    if operation=='addition':
        if r%2==0:
            for j in range(0,(r//2)+1):
                result+= b(r,2*j) * i4**(2*j) * i1**((r-2*j)//2) * i2**((r-2*j)/2)
            return result
        elif r%2 ==1:
            for j in range(0,((r-1)//2)+1):
                result+= b(r,2*j+1) * i4**(2*j+1) * i1**((r-2*j-1)//2) * i2**((r-2*j-1)/2)
            return result
    elif operation=='subtraction':
        if r%2==0:
            for j in range(0,(r/2)+1):
                result+= b(r,2*j) * i3**(2*j) * i1**((r-2*j)//2) * i2**((r-2*j)/2)
            return result
        elif r%2 ==1:
            for j in range(0,((r-1)/2)+1):
                result+= b(r,2*j+1) * i3**(2*j+1) * i1**((r-2*j-1)//2) * i2**((r-2*j-1)/2)
            return result
        
def K_ng(sigma, r, operation): #r is the number of photon additions /subtractions performed
    return f(sigma,r, operation)

def N_ng(sigma,r,operation):
    return f(sigma,r+1,operation)-f(sigma,r,operation)

def N2_ng(sigma,r,operation):
    return f(sigma,r+2,operation)-3*f(sigma,r+1, operation)+f(sigma,r, operation)

def gamma_ng(noise_factor, sigma,r,operation):
    numerator = N_ng(sigma,r,operation)/K_ng(sigma,r,operation) - 0.5*(noise_factor-1)
    denumerator = sqrt(N2_ng(sigma,r,operation)/K_ng(sigma,r,operation)-(N_ng(sigma,r,operation)/K_ng(sigma,r,operation))**2)
    return numerator/denumerator

noise=np.linspace(1,100,1000)
sq=0.3
def sigma_test(noise):
    return np.array([[noise,0],[0,noise]])

def sigma_test2(noise,sq):
    return np.array([[noise*sq,0],[0,noise/sq]])

plt.plot(noise,[np.cfloat(gamma_ng(n,sigma_test(n),1,'addition')) for n in noise],'r')
plt.plot(noise,[np.cfloat(gamma_ng(n,sigma_test2(n,sq),1,'addition')) for n in noise],'b')
plt.show()