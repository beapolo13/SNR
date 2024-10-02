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


