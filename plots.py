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
from numpy import where

from utils import *
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

def ratio_plots_reduced(N,params=None):  #only makes sense for N=2
  # variable intervals
  t = np.arange(0, 2*np.pi, 0.05) #for angles
  s = np.arange(0.05,3.95, 0.05)  #for squeezing
  fig, ((ax1, ax2, ax3),(ax4,ax5,ax6),(ax7,ax8,ax9),(ax10,ax11,ax12)) = plt.subplots(4, 3, figsize=(10, 10 ))
  phi=2*np.pi*np.random.rand(N)
  z=[0.5,2]
  z_vec=np.linspace(0.05,0.95,15)

  w_vec=[]
  for i in range(15):
    w_vec += [np.random.rand((N*(N-1))//2)]

  #gaussian case
  print('gaussian case')
  
  for q in z_vec:
    ax1.plot(t, [np.real(SNR_gaussian(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params))) for w in t], 'r')
    ax1.set_title('ratio w/ BS (fixed PS and z)')

  for w in w_vec:
    ax2.plot(t, [np.real(SNR_gaussian(V_tms(z,w,[value]+[0]*(N-1),params))) for value in t],'r')
    ax2.set_title('ratio vs PS (fixed BS and sq)') 
    ax3.plot(s, [np.real(SNR_gaussian(V_tms([sq,1/sq],w,phi,params))) for sq in s],'r')
    ax3.set_title('ratio vs squeezing (fixed BS and PS)')

  #nongaussian case
  nongaussian_ops=[-1]
  print(f"{nongaussian_ops}")
  ax4.plot(t, [np.real(SNR_ng(V_tms([0.5,2],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'b')
  for q in z_vec:
    ax4.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  
  
  for w in w_vec:
    ax5.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax6.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')
  

  nongaussian_ops=[-1,-1]
  print(f"{nongaussian_ops}")
  ax7.plot(t, [np.real(SNR_ng(V_tms([0.5,2],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'b')
  for q in z_vec:
    ax7.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  
  for w in w_vec:
    ax8.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax9.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')


  nongaussian_ops=[-1,-1,-1]
  print(f"{nongaussian_ops}")
  ax10.plot(t, [np.real(SNR_ng(V_tms([0.5,2],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'b')
  for q in z_vec:
    ax10.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  
  for w in w_vec:
    ax11.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax12.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')

  plt.show()
  
  return


def ratio_plots(N,params=None):  #only makes sense for N=2
  # variable intervals
  t = np.arange(0, 2*np.pi, 0.05) #for angles
  s = np.arange(0.05,0.95, 0.05)  #for squeezing
  fig, ((ax1, ax2, ax3),(ax4,ax5,ax6),(ax7,ax8,ax9),(ax10,ax11,ax12),(ax13,ax14,ax15),(ax16,ax17,ax18)) = plt.subplots(6, 3, figsize=(10, 10 ))
  phi=2*np.pi*np.random.rand(N)
  z=[0.5,2]
  z_vec=[]
  for i in range(10):
    z_vec+=[np.random.rand()]

  w_vec=[]
  for i in range(10):
    w_vec += [np.random.rand((N*(N-1))//2)]

  #gaussian case
  print('gaussian case')
  
  for q in z_vec:
    ax1.plot(t, [np.real(SNR_gaussian(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params))) for w in t], 'r')
    ax1.set_title('ratio w/ BS (fixed PS and z)')

  for w in w_vec:
    ax2.plot(t, [np.real(SNR_gaussian(V_tms(z,w,[value]+[0]*(N-1),params))) for value in t],'r')
    ax2.set_title('ratio vs PS (fixed BS and sq)') 
    ax3.plot(s, [np.real(SNR_gaussian(V_tms([sq,1/sq],w,phi,params))) for sq in s],'r')
    ax3.set_title('ratio vs squeezing (fixed BS and PS)')

  #nongaussian case
  nongaussian_ops=[-1]
  print(f"{nongaussian_ops}")
  for q in z_vec:
    ax4.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  
  
  for w in w_vec:
    ax5.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax6.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')
  

  nongaussian_ops=[-1,-1]
  print(f"{nongaussian_ops}")
  for q in z_vec:
    ax7.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  
  for w in w_vec:
    ax8.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax9.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')


  nongaussian_ops=[-1,-1,-1]
  print(f"{nongaussian_ops}")
  for q in z_vec:
    ax10.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  
  for w in w_vec:
    ax11.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax12.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')

  nongaussian_ops=[-1,-1,-1,-1]
  print(f"{nongaussian_ops}")
  for q in z_vec:
    ax13.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  
  for w in w_vec:
    ax14.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax15.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')

  nongaussian_ops=[-1,-1,-1,-1,-1]
  print(f"{nongaussian_ops}")
  for q in z_vec:
    ax16.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  
  for w in w_vec:
    ax17.plot(t, [np.real(SNR_ng(V_tms(z,w,[value]+[0]*(N-1),params),nongaussian_ops)) for value in t],'r')
    ax18.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')
  
  plt.show()
  
  return


def surface_plots(N,params=None):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  X = np.arange(0, 2*np.pi, 0.05) #for angles
  Y = np.arange(0.05,0.95, 0.05)  #for squeezing
  X, Y = np.meshgrid(X, Y)

  #gaussian case
  Z= [[np.real(SNR_gaussian(V_tms([y,1/y],x,[0]*N,params))) for y in Y] for x in X]
  # Plot the surface.
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
  ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
  ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()

  #nongaussian case
  nongaussian_ops=[-1]
  for i in range(3):
    for y in Y:
      for x in X:
        Z = np.real(SNR_ng(V_tms([y,1/y],x,[0]*N,params),nongaussian_ops)) 
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
  # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
  # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    nongaussian_ops+=[-1]

  return 

def single_photon_op(N,operation): #just for N=2
  nongaussian_ops=[operation]
  z_vec=np.linspace(0.05,0.95,50)
  w_vec=[]
  for i in range(10):
    w_vec += [np.random.rand((N*(N-1))//2)]
  t = np.arange(0, 2*np.pi, 0.05) #for BS angles (PS doesn't affect)
  s = np.arange(0.005,3.995, 0.005)  #for squeezing
  print(f"{nongaussian_ops}")
  fig,((ax1,ax2)) = plt.subplots(2, 1, figsize=(10, 10 ))
  for q in z_vec:
    ax1.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], 'r')
  for w in w_vec:
    ax2.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'r')
  plt.show()
  return


#ratio_plots_reduced(2)
#single_photon_op(2,1)


def ratio_plots_superreduced(N,params=None):  #only makes sense for N=2
  # variable intervals
  t = np.arange(0, 2*np.pi, 0.05) #for angles
  s = np.arange(0.05,3.95, 0.05)  #for squeezing
  fig, ((ax1),(ax2),(ax3),(ax4)) = plt.subplots(4, 1, figsize=(10, 25 ))
  phi=2*np.pi*np.random.rand(N)
  print(phi)
  z=[0.5,2]
  z_vec=np.linspace(0.05,0.95,15)
  my_array=np.linspace(0, 1, len(z_vec))
  colors = plt.cm.viridis(my_array)
  idx=np.abs(my_array - 0.5).argmin()
  print(idx)

  w_vec=[]
  for i in range(15):
    w_vec += [np.random.rand((N*(N-1))//2)]

  #gaussian case
  print('gaussian case')
  
  i=0
  for q in z_vec:
    ax1.plot(t, [np.real(SNR_gaussian(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params))) for w in t], color=colors[i])
    i+=1
  ax1.set_xlabel('Beamsplitter angle')
  ax1.set_ylabel('SNR')

  
  #nongaussian case
  nongaussian_ops=[-1]
  print(f"{nongaussian_ops}")
  ax2.plot(t, [np.real(SNR_ng(V_tms([0.5,2],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], color=colors[idx])
  i=0
  for q in z_vec:
    ax2.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], color=colors[i])
    i+=1
  ax2.set_xlabel('Beamsplitter angle')
  ax2.set_ylabel('SNR')
  

  nongaussian_ops=[-1,-1]
  print(f"{nongaussian_ops}")
  ax3.plot(t, [np.real(SNR_ng(V_tms([0.5,2],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], color=colors[idx])
  i=0
  for q in z_vec:
    ax3.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], color=colors[i])
    i+=1
  ax3.set_xlabel('Beamsplitter angle')
  ax3.set_ylabel('SNR')

  nongaussian_ops=[-1,-1,-1]
  print(f"{nongaussian_ops}")
  ax4.plot(t, [np.real(SNR_ng(V_tms([0.5,2],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], color=colors[idx])
  i=0
  for q in z_vec:
    ax4.plot(t, [np.real(SNR_ng(V_tms([q,1/q],[w]+[0]*((N*(N-1))//2 -1),phi,params),nongaussian_ops)) for w in t], color=colors[i])
    i+=1
  ax4.set_xlabel('Beamsplitter angle')
  ax4.set_ylabel('SNR')

  cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=[ax1, ax2,ax3,ax4], location='right')
  cbar.set_label('Squeezing factor z')

# Adjust layout
  plt.tight_layout(rect=[0.05, 0.05, 0.75, 0.95])  # Adjust the layout to make space for the colorbar
  plt.show()
  
  return


def scaling_with_nongaussianity(N):
  fig, ax = plt.subplots(1, 1, figsize=(15, 15 ))
  s = np.arange(0.05,0.95, 0.05)  #for squeezing
  my_array=np.linspace(0, 1, len(s))
  colors = plt.cm.viridis(my_array)
  #phi=np.random.rand(N)
  phi=[0,0]
  t = np.arange(0, 2*np.pi, 0.05) #for angles
  #non_gauss_vect=[[],[-1],[-1,-1]]
  non_gauss_vect=[[],[-1],[-1,-1]]
  #non_gauss_vect=[[],[-1],[-1,-1],[-1,-1],[-1,-1,-1],[-1,-1,-1,-1]]
  lengths= [len(item) for item in non_gauss_vect]
  i=0
  for sq in s:
    SN_ratios=[]
    for i in range(len(non_gauss_vect)):
      SN_ratios+=[np.max([SNR_ng(V_tms([sq,1/sq],[w]+[0]*((N*(N-1))//2 -1),phi,None),non_gauss_vect[i]) for w in t])]
    ax.plot(lengths, SN_ratios,'-o', color=colors[i])
    i+=1
  #plt.legend(s)
  cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'),ax=ax, location='right')
  cbar.set_label('Squeezing factor z')
  plt.show()

  return

scaling_with_nongaussianity(2)