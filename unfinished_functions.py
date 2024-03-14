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
from covariance_matrix import *
from expectation_values import *

#Check for separability of the gaussian covariance matrix

def check(sigma,N):  #only for N=2
  #print('input sigma')
  #pprint(sigma)

  separability=True

  print('sigma after reordering')
  pprint(np.round(sigma,2))
  omega1=[[0,1],[-1,0]]
  block_vector=[]
  for i in range(N):
    block_vector+=[omega1]
  print('block_vector',block_vector)
  print(tuple(block_vector))
  Omega= block_diag(omega1,omega1)
  print('Omega',Omega)
  eigenvalues, eigenvectors= np.linalg.eig(sigma+1j*Omega)
  print('eigenvalues',eigenvalues)
  check=0
  for item in eigenvalues:
      if np.round(np.real(item),3) < 0:
        check= check
      else:
          check+=1
  if check==2*N:
    print('state is gaussian')
  if check!=2*N:
        print('Not gaussian state')
        return None
  if N!=2:
    print('Need to apply some other criterion for this value of N')
    return None
  sigmaA=np.array([[sigma[0][0],sigma[0][1]],[sigma[1][0],sigma[1][1]]])
  #print('sigmaA', sigmaA)
  sigmaB=np.array([[sigma[2][2],sigma[2][3]],[sigma[3][2],sigma[3][3]]])
  #print('sigmaB', sigmaB)
  sigmaAB=np.array([[sigma[0][2],sigma[0][3]],[sigma[1][2],sigma[1][3]]])
  #print('sigmaAB', sigmaAB)
  detA= np.linalg.det(sigmaA)
  detB= np.linalg.det(sigmaB)
  detAB= np.linalg.det(sigmaAB)
  determinant=np.linalg.det(sigma)
  det_hat= detA +detB -2*detAB
  nu= (det_hat - np.sqrt(det_hat**2-4*determinant))/2
  #print(nu)
  if nu <1:
    separability=False
  return nu

#print('Separability check for rho sep', check(rho_gaussian_sep))
#print('Separability check for rho ent', check(rho_gaussian_ent))





#criterion 1 for separability (PPT) (something is wrong here, check!)
def separability(sigma):
    N= len(sigma)//2
    T=np.diag([1,1,1,-1])
    sigma_hat= T @ sigma @ T
    print('sigma hat')
    pprint(np.round(sigma_hat,1))
    eigenvalues2, eigenvectors= np.linalg.eig(sigma_hat)
    print(eigenvalues2)
    check=0
    for item in eigenvalues2:
        if np.round(np.real(item),3) < 0 or np.round(np.real(item),3)==0 :
            print('Entangled')
        else:
            check+=1
    if check==2*N:
          #print('Separable')
          separability=True
    return 


#3D PLOT:
def three_D_plot():
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  Z = np.abs(expvalN_ng(V_tms(z,T,[0,0],params1))/varianceN_ng(V_tms(z,T,[0,0],params1)))

  # Plot the surface.
  surf = ax.plot_surface(T, S, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

  # Customize the z axis.
  ax.set_zlim(-0.01, 2.01)
  ax.zaxis.set_major_locator(LinearLocator(10))
  # A StrMethodFormatter is used automatically
  ax.zaxis.set_major_formatter('{x:.02f}')
  plt.xlabel('angle x of BS')
  plt.ylabel('phi1')
  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=6)
  plt.title('Evolution of ratio as a function of the angle of dephasing phi1 and angle of beamsplitter')
  plt.show()
  return