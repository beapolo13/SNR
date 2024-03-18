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

def ratio_results(nongaussian_ops,z,theta,phi,params):
  N=len(z)
  print('Initialization parameters:',nongaussian_ops,z,theta,phi,params)
  rho_gaussian_sep= V_tms(z,[0]*((N*(N-1))//2),[0]*N,params)
  print('rho sep',np.round(rho_gaussian_sep,2))
  rho_gaussian_ent=V_tms(z,theta,phi,params)
  print('rho ent',np.round(rho_gaussian_ent,2))

  #GAUSSIAN QUANTITIES
  print('expvalN',expvalN(rho_gaussian_sep))
  print('N2', N2(rho_gaussian_sep))
  print('variance N',varianceN(rho_gaussian_sep))
  print('ratio N/delta(N) gaussian',SNR_gaussian(rho_gaussian_sep))


  print(' ')
  print('With beam splitter + phase shifter: (entanglement)')
  print('N',expvalN(V_tms(z,theta,phi,params)))
  print('N2', N2(V_tms(z,theta,phi,params)))
  print('delta N',varianceN(V_tms(z,theta,phi,params)))
  print('ratio N/delta(N) gaussian',SNR_gaussian(V_tms(z,theta,phi,params)))
  print(' ')

  #NON GAUSSIAN

  print('Non gaussian')
  print('expvalN_ng',expvalN_ng(rho_gaussian_sep,nongaussian_ops))
  print('N2_ng', N2_ng(rho_gaussian_sep,nongaussian_ops))
  print('variance N_ng',varianceN_ng(rho_gaussian_sep,nongaussian_ops))
  print('ratio N/delta(N) non gaussian',SNR_ng(rho_gaussian_sep,nongaussian_ops))
  print(' ')
  print('With beam splitter + phase shifter: (entanglement)')

  print('expvalN_ng',expvalN_ng(V_tms(z,theta,phi,params),nongaussian_ops))
  print('N2_ng', N2_ng(V_tms(z,theta,phi,params),nongaussian_ops))
  print('variance N_ng',varianceN_ng(V_tms(z,theta,phi,params),nongaussian_ops))
  print('ratio N/delta(N) non gaussian',SNR_ng(V_tms(z,theta,phi,params),nongaussian_ops))
  print(' ')

  return


def ratio_plots(N,params=None):  #only makes sense for N=2
  # variable intervals
  t = np.arange(0, 2*np.pi, 0.05) #for angles
  s = np.arange(0.05,0.95, 0.05)  #for squeezing
  
  #gaussian case
  
  print('gaussian case')
  phi=2*np.pi*np.random.rand(N)
  z=[0.5,2]
  fig, ((ax1, ax2, ax3),(ax4,ax5,ax6),(ax7,ax8,ax9),(ax10,ax11,ax12)) = plt.subplots(4, 3, figsize=(10, 10 ))
  ax1.plot(t, [np.real(SNR_gaussian(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params))) for w in t], 'r')
  ax1.plot(t,[np.real(expvalN(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params))) for w in t],'b')
  ax1.plot(t,[np.real(varianceN(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params))) for w in t],'g')
  ax1.set_title('ratio w/ BS (fixed PS and z)')
  ax1.legend(['ratio','energy', 'variance'])


  w_vec=[]
  for i in range(10):
    w_vec += [np.random.rand((N*(N-1))//2)]
  for w in w_vec:
    ax2.plot(t, [np.real(SNR_gaussian(V_tms(z,w,[value]+[0]*(N-1),params))) for value in t],'r')
    ax2.set_title('ratio vs PS (fixed BS and sq)') 
    ax3.plot(s, [np.real(SNR_gaussian(V_tms([sq,1/sq],w,phi,params))) for sq in s],'r')
    ax3.set_title('ratio vs squeezing (fixed BS and PS)')

  #nongaussian case
  nongaussian_ops=[1]
  print(f"{nongaussian_ops}")
  ax4.plot(t, [np.real(SNR_ng(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  ax4.plot(t,[np.real(expvalN_ng(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t],'b')
  ax4.plot(t,[np.real(varianceN_ng(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t],'g')
  ax4.legend(['ratio','energy', 'variance'])

  
  for w in w_vec:
    ax5.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax6.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')
  

  nongaussian_ops=[1,1]
  print(f"{nongaussian_ops}")
  ax7.plot(t, [np.real(SNR_ng(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  ax7.plot(t,[np.real(expvalN_ng(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t],'b')
  ax7.plot(t,[np.real(varianceN_ng(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t],'g')
  ax7.legend(['ratio','energy', 'variance'])

  
  for w in w_vec:
    ax8.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax9.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')


  nongaussian_ops=[1,1,1]
  print(f"{nongaussian_ops}")
  ax10.plot(t, [np.real(SNR_ng(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  ax10.plot(t,[np.real(expvalN_ng(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t],'b')
  ax10.plot(t,[np.real(varianceN_ng(V_tms(z,[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t],'g')
  ax10.legend(['ratio','energy', 'variance'])

  
  for w in w_vec:
    ax11.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax12.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')
  plt.show()
  
  return





