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

def results_and_plots(nongaussian_ops,z,theta,modesBS,phi,params):
  print('Initialization parameters:',nongaussian_ops,z,theta,modesBS,phi,params)
  rho_gaussian_sep= V_tms(z,0,modesBS,phi,params)
  print('rho sep',np.round(V_tms(z,0,modesBS,phi,params),2))
  rho_gaussian_ent=V_tms(z,theta,modesBS,phi,params)
  print('rho ent',np.round(V_tms(z,theta,modesBS,phi,params),2))

  #GAUSSIAN QUANTITIES
  print('expvalN',expvalN(rho_gaussian_sep))
  print('N2', N2(rho_gaussian_sep))
  print('variance N',varianceN(rho_gaussian_sep))
  print('ratio N/delta(N) gaussian',expvalN(rho_gaussian_sep)/varianceN(rho_gaussian_sep))


  print(' ')
  print('With beam splitter + phase shifter: (entanglement)')
  print('N',expvalN(V_tms(z,theta,modesBS,phi,params)))
  print('N2', N2(V_tms(z,theta,modesBS,phi,params)))
  print('delta N',varianceN(V_tms(z,theta,modesBS,phi,params)))
  print('ratio N/delta(N) gaussian',expvalN(V_tms(z,theta,modesBS,phi,params))/varianceN(V_tms(z,theta,modesBS,phi,params)))
  print(' ')

  #NON GAUSSIAN

  print('Non gaussian')
  print('expvalN_ng',expvalN_ng(rho_gaussian_sep,nongaussian_ops))
  print('N2_ng', N2_ng(rho_gaussian_sep,nongaussian_ops))
  print('variance N_ng',varianceN_ng(rho_gaussian_sep,nongaussian_ops))
  print('ratio N/delta(N) non gaussian',SNR_ng(rho_gaussian_sep,nongaussian_ops))

  print(' ')
  print('With beam splitter + phase shifter: (entanglement)')

  print('expvalN_ng',expvalN_ng(V_tms(z,theta,modesBS,phi,params),nongaussian_ops))
  print('N2_ng', N2_ng(V_tms(z,theta,modesBS,phi,params),nongaussian_ops))
  print('variance N_ng',varianceN_ng(V_tms(z,theta,modesBS,phi,params),nongaussian_ops))
  print('ratio N/delta(N) non gaussian',SNR_ng(V_tms(z,theta,modesBS,phi,params),nongaussian_ops))
  print(' ')

  # angle interval for beam splitter
  t = np.arange(0, 2*np.pi, 0.05)
  s = np.arange(0,2*np.pi, 0.05)
  w = np.arange(0,2*np.pi, 0.3)
  T,S = np.meshgrid(t,s)
  TS= np.stack([T,S])
  t_vec=np.arange(0,2*np.pi,0.005)

  #nongaussian case
  print(f"{nongaussian_ops}")
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
  ax1.plot(t, [np.real(SNR_ng(V_tms(z,w,modesBS,phi,params),nongaussian_ops)) for w in t], 'r')
  ax1.set_title('Evolution w/ Bsplitter (fixed phase)')
  ax1.plot(t,[np.real(expvalN_ng(V_tms(z,w,modesBS,phi,params),nongaussian_ops)) for w in t],'b')
  ax1.plot(t,[np.real(varianceN_ng(V_tms(z,w,modesBS,phi,params),nongaussian_ops)) for w in t],'g')
  #ax1.plot(t_vec, [check(V_tms(z,t,modesBS,phi,params)) for t in t_vec],'-')

  ax1.legend(['ratio','energy', 'variance', 'PT eigenvalue nu_ '])
  #plt.show()

  t2_vec=np.arange(0,2*np.pi,0.4)
  for w in t2_vec:
    ax2.plot(s, [np.real(SNR_ng(V_tms(z,w,modesBS,[value,0,0,0],params),nongaussian_ops)) for value in s],'r')
  ax2.set_title('Evolution w/ dephasing (fixed BS)')
  plt.show()
  
  return

def main():
    for N in range(2,5):
      z_first=list(np.random.rand(N-1))
      z=z_first+[N-sum(z_first)]
      nongaussian_ops=[]  #-n for photon subtraction on mode n, or +n if its photon addition on mode n
      theta=np.pi/7
      modesBS=[1,2]
      phi=[0]*N
      params=None #leave them at none if we want to start from vacuum and random if we want the most general cov matrix
      results_and_plots(nongaussian_ops,z,theta,modesBS,phi,params)
      results_and_plots(nongaussian_ops,[0.5]*(N//2)+[0.7]*(N//2),theta,modesBS,phi,params)
      
    
    return

main()