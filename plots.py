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
import matplotlib.ticker as ticker
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
  phi=[0,0]
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
    ax2.plot(s, [np.real(expvalN_ng(V_tms([sq,1/sq],w,phi,params),nongaussian_ops)) for sq in s],'b')
  plt.show()
  return


#ratio_plots_reduced(2)
#single_photon_op(2,[-1,-2])


def ratio_plots_superreduced(N,params=None):  #only makes sense for N=2
  # variable intervals
  t = np.arange(0, 2*np.pi, 0.05) #for angles
  s = np.arange(0.05,3.95, 0.05)  #for squeezing
  fig, ((ax1),(ax2),(ax3),(ax4)) = plt.subplots(4, 1, figsize=(10, 25))
  phi=2*np.pi*np.random.rand(N)
  print(phi)
  z=[0.5,2]
  z_vec=np.linspace(0.05,0.95,15)
  my_array=np.linspace(0, 1, len(z_vec))
  colors = plt.cm.viridis(my_array)
  idx=np.abs(my_array - 0.5).argmin()
  print(idx)


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

def ratio_plots_superreduced_thermal(N,params=None):  #only makes sense for N=2
  # variable intervals
  t = np.arange(0, 2*np.pi, 0.05) #for angles
  s = np.arange(0.05,3.95, 0.05)  #for squeezing
  fig, ((ax1),(ax2),(ax3),(ax4)) = plt.subplots(4, 1, figsize=(10, 25))
  phi1=2*np.pi*np.random.rand(N)
  phi2=2*np.pi*np.random.rand(N)
  z_vec=np.linspace(0.05,0.95,15)
  my_array=np.linspace(0, 1, len(z_vec))
  colors = plt.cm.YlOrRd(my_array)
  idx=np.abs(my_array - 0.5).argmin()
  print(idx)


  #gaussian case
  print('gaussian case')

  
  
  i=0
  for q in z_vec:
    ax1.plot(t, [np.real(SNR_gaussian(V_thermal([0.5]*2,[q,1/q],[w],[0],phi1,phi2,params1=None,params2=None))) for w in t], color=colors[i])
    i+=1
  ax1.set_xlabel('Beamsplitter angle')
  ax1.set_ylabel('SNR')

  
  #nongaussian case
  nongaussian_ops=[-1]
  print(f"{nongaussian_ops}")
  ax2.plot(t, [np.real(SNR_ng(V_thermal([0.5]*2,[q,1/q],[w],[0],phi1,phi2,params1=None,params2=None),nongaussian_ops)) for w in t], color=colors[idx])
  i=0
  for q in z_vec:
    ax2.plot(t, [np.real(SNR_ng(V_thermal([0.5]*2,[q,1/q],[w],[0],phi1,phi2,params1=None,params2=None),nongaussian_ops)) for w in t], color=colors[i])
    i+=1
  ax2.set_xlabel('Beamsplitter angle')
  ax2.set_ylabel('SNR')
  

  nongaussian_ops=[-1,-1]
  print(f"{nongaussian_ops}")
  ax3.plot(t, [np.real(SNR_ng(V_thermal([0.5]*2,[q,1/q],[w],[0],phi1,phi2,params1=None,params2=None),nongaussian_ops)) for w in t], color=colors[idx])
  i=0
  for q in z_vec:
    ax3.plot(t, [np.real(SNR_ng(V_thermal([0.5]*2,[q,1/q],[w],[0],phi1,phi2,params1=None,params2=None),nongaussian_ops)) for w in t], color=colors[i])
    i+=1
  ax3.set_xlabel('Beamsplitter angle')
  ax3.set_ylabel('SNR')

  nongaussian_ops=[-1,-1,-1]
  print(f"{nongaussian_ops}")
  ax4.plot(t, [np.real(SNR_ng(V_thermal([0.5]*2,[q,1/q],[w],[0],phi1,phi2,params1=None,params2=None),nongaussian_ops)) for w in t], color=colors[idx])
  i=0
  for q in z_vec:
    ax4.plot(t, [np.real(SNR_ng(V_thermal([0.5]*2,[q,1/q],[w],[0],phi1,phi2,params1=None,params2=None),nongaussian_ops)) for w in t], color=colors[i])
    i+=1
  ax4.set_xlabel('Beamsplitter angle')
  ax4.set_ylabel('SNR')

  cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd'), ax=[ax1, ax2,ax3,ax4], location='right')
  cbar.set_label('Squeezing factor z')

# Adjust layout
  plt.tight_layout(rect=[0.05, 0.05, 0.75, 0.95])  # Adjust the layout to make space for the colorbar
  plt.show()
  
  return

#ratio_plots_superreduced_thermal(2,params=None)


def scaling_with_nongaussianity(N):
  fig, ((ax1,ax2)) = plt.subplots(1, 2, figsize=(30, 10))
  s = np.arange(0.05,0.95, 0.05)  #for squeezing
  my_array=np.linspace(0, 1, len(s))
  colors = plt.cm.viridis(my_array)
  #phi=np.random.rand(N)
  phi1=[0,0]
  phi2=[np.pi/2,0]
  #phi3=[np.pi/2,0]
  t = np.arange(0, 2*np.pi, 0.05) #for angles
  non_gauss_vect=[[],[-1],[-1,-1]]
  #non_gauss_vect=[[],[1],[1,1],[1,1,1]]
  lengths= [len(item) for item in non_gauss_vect]
  j=0
  for sq in s:
    SN_ratios1=[]
    SN_ratios2=[]
    #SN_ratios3=[]
    for i in range(len(non_gauss_vect)):
      SN_ratios1+=[np.max([SNR_ng(V_tms([sq,1/sq],[w]+[0]*((N*(N-1))//2 -1),phi1,None),non_gauss_vect[i]) for w in t])]
      SN_ratios2+=[np.max([SNR_ng(V_tms([sq,1/sq],[w]+[0]*((N*(N-1))//2 -1),phi2,None),non_gauss_vect[i]) for w in t])]
      #SN_ratios3+=[np.max([SNR_ng(V_tms([sq,1/sq],[w]+[0]*((N*(N-1))//2 -1),phi3,None),non_gauss_vect[i]) for w in t])]
    ax1.plot(lengths, SN_ratios1,'-o', color=colors[j])
    ax2.plot(lengths, SN_ratios2,'-o', color=colors[j])
    #ax3.plot(lengths, SN_ratios3,'-o', color=colors[j])
    j+=1
  ax1.set_xlabel('Number of single-photon operations')
  ax1.set_ylabel('Maximum attainable SNR')
  ax2.set_xlabel('Number of single-photon operations')
  ax2.set_ylabel('Maximum attainable SNR')
  #ax3.set_xlabel('Number of single-photon operations')
  #ax3.set_ylabel('Maximum attainable SNR')
  ax1.set_title('No dephasing')
  ax2.set_title(f'Fixed PS phi={phi2[0]}')
  #ax3.set_title(f'Fixed PS phi={phi3[0]}')
  # Change x-axis tick spacing
  ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
  ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
  #ax3.xaxis.set_major_locator(ticker.MultipleLocator(base=1))
  cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'),ax=[ax1,ax2], location='right')
  cbar.set_label('Squeezing factor z')
  fig.suptitle(f'Scaling of Max(SNR) w.r.t photon subtractions.N=2. Nongauss ops: {non_gauss_vect}')
  plt.show()

  return


def evolution_with_squeezing():
  N=2
  s = np.arange(0.05,3.95, 0.05)  #for squeezing
  bs= [np.pi/5]
  phi=[0]*N
  params=None
  fig1, ax0 = plt.subplots(1, 1, figsize=(10, 10))
  ax0.set_xlabel('Squeezing z')
  ax0.set_ylabel('SNR')
  ax0.plot(s, [SNR_gaussian(V_tms([w,1/w],bs,phi,params)) for w in s])
  plt.show()
  fig2, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(2, 4, figsize=(10, 25))
  ax1.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],bs,phi,params),[-1])) for sq in s],'r')
  ax1.set_title('[-1]')
  ax2.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],bs,phi,params),[-1,-1])) for sq in s],'r')
  ax2.set_title('[-1,-1]')
  ax3.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],bs,phi,params),[-1,-1,-1])) for sq in s],'r')
  ax3.set_title('[-1,-1,-1]')
  ax4.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],bs,phi,params),[-1,-1,-1,-1])) for sq in s],'r')
  ax4.set_title('[-1,-1,-1,-1]')
  ax5.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],bs,phi,params),[1])) for sq in s],'r')
  ax5.set_title('[1]')
  ax6.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],bs,phi,params),[1,1])) for sq in s],'r')
  ax6.set_title('[1,1]')
  ax7.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],bs,phi,params),[1,1,1])) for sq in s],'r')
  ax7.set_title('[1,1,1]')
  ax8.plot(s, [np.real(SNR_ng(V_tms([sq,1/sq],bs,phi,params),[1,1,1,1])) for sq in s],'r')
  ax8.set_title('[1,1,1,1]')
  plt.show()

#evolution_with_squeezing()


def SV_plots(nongaussian_ops):
  t = np.arange(0, 2*np.pi, 0.05) #for angles
  z_vec=np.linspace(0.25,0.8,10) #squeezing values
  colors = plt.cm.viridis(z_vec)
  fig,axes=plt.subplots(4,2)
  phi=np.random.rand(2)
  j=0
  for j in range(len(nongaussian_ops)):
    axup = axes[(j//2)*2][j%2]
    axdown= axes[(j//2)*2+1][j%2]
    i=0
    for q in z_vec:
      axup.plot(t, [SV(V_tms([q,1/q],[w],[0,0],params=None),nongaussian_ops[j]) for w in t], color=colors[i]) 
      axup.set_title('SV criterion')
      axdown.plot(t,[SNR_ng(V_tms([q,1/q],[w],[0,0],params=None),nongaussian_ops[j]) for w in t], color=colors[i])
      axdown.set_title('SNR')
      i+=1
  # Adjust layout to make room for the colorbar
  plt.subplots_adjust(right=0.85)

# Add a single colorbar outside the subplot grid
  cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
  cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), cax=cbar_ax)
  cbar.set_label('Squeezing factor z')


  #plt.legend(z_vec)
  plt.show()
  return 

SV_plots([[],[-1],[-1,-1],[-1,-1,-1]])


def evolution_with_noise_gaussian():
  fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(30, 15))
  z=0.5
  noise=np.linspace(0.2,2,15)
  yvec= [expvalN(V_thermal([n]*2,[z,1/z],[np.pi/4],[0],[0,0],[0,0],params1=None,params2=None)) for n in noise]
  yvec_2=[varianceN(V_thermal([n]*2,[z,1/z],[np.pi/4],[0],[0,0],[0,0],params1=None,params2=None)) for n in noise]
  yvec_3=[SNR_gaussian(V_thermal([n]*2,[z,1/z],[np.pi/4],[0],[0,0],[0,0],params1=None,params2=None)) for n in noise]
  ax1.plot(noise,yvec)
  ax1.plot(noise,yvec_2)
  ax1.plot(noise,yvec_3)

  ax1.set_xlabel('noise')
  ax1.set_title('SNR as a function of temperature for fixed z=0.5')
  legend=['Signal N','delta N', 'SNR']
  ax1.legend(legend)

  #second plot
  z_vec=list(np.arange(0.001,4.99,0.001))
  my_array=np.linspace(0.1, 1, len(noise))
  colors = plt.cm.YlOrRd(my_array)
  for i in range(len(noise)): 
    yvec= [SNR_ng(V_thermal([noise[i]]*2,[z,1/z],[np.pi/4],[0],[0,0],[0,0],params1=None,params2=None),[]) for z in z_vec]
    ax2.plot(z_vec,yvec, color=colors[i])
  cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd'), ax=ax2, location='right')
  cbar.set_label('Noise')
  ax2.set_xlabel('z')
  ax2.set_title('SNR as a function of squeezing')
  plt.show()
  return

#evolution_with_noise_gaussian()


