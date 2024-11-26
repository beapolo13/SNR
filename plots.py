import numpy as np
from numpy import transpose, real, sqrt, sin, cos, linalg, cosh, sinh, log
import scipy
import math
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
import matplotlib.colors as mcolors


from utils import *
from expectation_values_cat import *
params = {'axes.linewidth': 2,
         'axes.labelsize': 15,
         'axes.titlesize': 15,
         'axes.linewidth': 1.2,
         'lines.markeredgecolor': "black",
     	'lines.linewidth': 1.2,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10,
         "text.usetex": True,
         "font.serif": ["Palatino"],
         "font.family": "serif"
         }
plt.rcParams.update(params)




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
  x = np.arange(0, np.pi, 0.005) #for angles
  z_vec=np.linspace(0.25,0.85,10) #squeezing values
  colors = plt.cm.viridis(z_vec)
  T=1.3
  sigma0=V_thermal(T,[1,1],[0],[0]*2,params=None)
  phi=np.random.rand(2)
  fig,axes=plt.subplots(2,1,figsize=(15,15))
  j=0
  for j in range(len(nongaussian_ops)):
    axup = axes[(j//2)*2]
    axdown= axes[(j//2)*2+1]
    i=0
    for q in z_vec:
      axup.plot(x, [SV(V_thermal(T,[q,1/q],[w],[0,0],params=None),nongaussian_ops[j]) for w in x], color=colors[i]) 
      axup.set_ylabel('SV criterion')
      axdown.plot(x,[SNR_ng_extr(V_thermal(T,[q,1/q],[w],[0,0],params=None),nongaussian_ops[j],sigma0) for w in x], color=colors[i])
      axdown.set_ylabel('SNR extractable')
      axdown.set_xlabel(r' Beam splitter angle $\displaystyle \theta$')
      i+=1
  
  # Adjust layout to make room for the colorbar
  plt.subplots_adjust(right=0.85)

# Add a single colorbar outside the subplot grid
  cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
  cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), cax=cbar_ax)
  cbar.set_label(r'Squeezing factor $z$')
  plt.savefig('SV-SNR plots nu=1.3, 2 subtractions.pdf')

  #plt.legend(z_vec)
  plt.show()
  return 





def evolution_with_noise_gaussian():
  
  plt.rc('font', family='serif')
  fig, ax = plt.subplots(1, 1, figsize=(30, 15))
  noise=np.linspace(1,3,10)
  T=1/np.log((2/(noise-1) +1))
  #second plot
  z_vec=list(np.arange(0.001,1,0.001))
  cmap=cm.rainbow
  norm = mcolors.Normalize(vmin=T.min(),vmax=T.max())
  for i in range(len(noise)):
    sigma0=V_thermal(noise[i],[1,1],[0],[0]*2,params=None)
    yvec= [SNR_ng_extr(V_thermal(noise[i],[z,1/z],[0],[0,0],params=None),[0,0,0,0],[],sigma0) for z in z_vec]
    ax.plot(z_vec,yvec, color=cmap(norm(T[i])))
  cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, location='right') 
  cbar.set_label(r'Noise $\gamma$')
  ax.set_xlabel('Squeezing parameter z')
  ax.set_ylabel('SNR extractable')
  #ax.set_title('SNR as a function of squeezing')
  plt.savefig('gaussian_extr_snr_with noise.pdf')
  plt.show()
  return

def bounds():
  fig, ax = plt.subplots(1, 1, figsize=(30, 15))
  plt.rc('font', family='serif')
  x_vec=np.linspace(0.00000001,np.pi,100)
  nu=1.3
  factor=-1
  if factor==-1:
    operation='subtraction'
  else:
    operation='addition'
  z_vec=np.linspace(0.0001,1,100)
  sigma=[V_thermal(nu,[z_vec[i],1/z_vec[i]],[0],[0,0],params=None) for i in range(len(z_vec))]
  sigma0=V_thermal(nu,[1,1],[0],[0,0],params=None)
  two_worst=[]
  two_best=[]
  three_worst=[]
  three_best=[]
  for z in z_vec:
    two_worst+=[np.min([SNR_ng_extr(V_thermal(nu,[z,1/z],[0],[0,0]),[factor,factor],sigma0),SNR_ng_extr(V_thermal(nu,[z,1/z],[np.pi/4],[0,0]),[factor,factor],sigma0)])]
    two_best+=[np.max([SNR_ng_extr(V_thermal(nu,[z,1/z],[0],[0,0]),[factor,factor],sigma0),SNR_ng_extr(V_thermal(nu,[z,1/z],[np.pi/4],[0,0]),[factor,factor],sigma0)])]
    three_worst+=[np.min([SNR_ng_extr(V_thermal(nu,[z,1/z],[0],[0,0]),[factor,factor,factor],sigma0),SNR_ng_extr(V_thermal(nu,[z,1/z],[np.pi/4],[0,0]),[factor,factor,factor],sigma0)])]
    three_best+=[np.max([SNR_ng_extr(V_thermal(nu,[z,1/z],[0],[0,0]),[factor,factor,factor],sigma0),SNR_ng_extr(V_thermal(nu,[z,1/z],[np.pi/4],[0,0]),[factor,factor,factor],sigma0)])]

  gaussian=[SNR_gaussian_extr(sigma[i],sigma0) for i in range(len(z_vec))]
  ax.plot(z_vec,gaussian, 'black',linestyle='dashed',label='Gaussian')
  ax.annotate('Gaussian',xy  = ( z_vec[20], gaussian[20]), xytext = (1.02*z_vec[20], gaussian[20]),color  = 'black',fontsize=30)
  y=[SNR_ng_extr(sigma[i],[factor],sigma0) for i in range(len(z_vec))]
  ax.plot(z_vec,y, 'b',label= '1 photon {operation}')
  ax.annotate(f'1 photon {operation}',xy  = ( z_vec[25], 0.85*y[25]), xytext = (1.02*z_vec[25], 0.8*y[25]),color  = 'black',fontsize=30)
  ax.plot(z_vec,two_worst,'b')
  ax.plot(z_vec,two_best, 'b',label='2 photon {operation}')
  y=two_best
  ax.annotate(f'2 photon {operation}s',xy  = ( z_vec[30], 0.90*y[30]), xytext = (0.9*z_vec[30], 0.80*y[30]),color  = 'black',fontsize=30)
  ax.plot(z_vec,three_worst, 'b')
  ax.plot(z_vec,three_best,'b', label='3 photon {operation}')
  y=three_best
  ax.annotate(f'3 photon {operation}s',xy  = ( z_vec[80], 0.9*y[80]), xytext = (1.02*z_vec[80], 1.02*y[80]),color  = 'black',fontsize=30)
  ax.fill_between(z_vec,two_worst,two_best, color='c',alpha=0.3)
  ax.fill_between(z_vec,three_worst,three_best, color='c', alpha=0.3)
  #plt.legend(['Gaussian',f'1 photon {operation}',f'2 photon {operation}s (worst)',f'2 photon {operation}s (best)',f'3 photon {operation}s (worst)',f'3 photon {operation}s (best)'], fontsize=12)
  ax.set_xlabel(r'Squeezing factor $z$')
  ax.set_ylabel('SNR extractable') 
  #plt.title('Evolution of SNR and extractable SNR with squeezing factor' )
  plt.savefig(f'bounds {operation}.pdf')
  plt.show()

def critical_temp():
  epsilon=0.001
  nu_vec=np.linspace(1.5,100,1000)
  sigma0=[V_thermal(nu,[1,1],[0],[0,0],params=None) for nu in nu_vec]
  sigma_right=[V_thermal(nu,[1,1],[0],[0,0],params=None) for nu in nu_vec]
  sigma_left=[V_thermal(nu,[1-epsilon,1/(1-epsilon)],[0],[0,0],params=None) for nu in nu_vec]
  derivative_0 = [(SNR_gaussian_extr(sigma_right[i],sigma0[i])-SNR_gaussian_extr(sigma_left[i],sigma0[i]))/epsilon for i in range(len(nu_vec))]
  derivative_1 = [(SNR_ng_extr(sigma_right[i],[1],sigma0[i])-SNR_ng_extr(sigma_left[i],[1],sigma0[i]))/epsilon for i in range(len(nu_vec))]
  derivative_2 = [(SNR_ng_extr(sigma_right[i],[+1,+1],sigma0[i])-SNR_ng_extr(sigma_left[i],[+1,+1],sigma0[i]))/epsilon for i in range(len(nu_vec))]
  derivative_3 = [(SNR_ng_extr(sigma_right[i],[+1,+1,+1],sigma0[i])-SNR_ng_extr(sigma_left[i],[+1,+1,+1],sigma0[i]))/epsilon for i in range(len(nu_vec))]
  plt.plot(nu_vec,derivative_0)
  plt.plot(nu_vec,derivative_1)
  plt.plot(nu_vec,derivative_2)
  plt.plot(nu_vec,derivative_3)
  plt.plot(nu_vec,[0]*len(nu_vec),'--')
  plt.xlabel('Noise', fontsize=13)
  plt.ylabel('Slope of SNR function at the Fock state', fontsize=13)
  plt.show()

def density_plot_temp(): 
  nu_vec=np.linspace(1.1,5,200)
  z_vec=np.linspace(0.001,1,100)
  X=z_vec
  Y=1/np.log((2/(nu_vec-1) +1))
  sigma0=[V_thermal(nu,[1,1],[0],[0,0],params=None) for nu in nu_vec]
  sigma=[[V_thermal(nu,[z,1/z],[0],[0,0],params=None) for z in z_vec] for nu in nu_vec]
  X_grid, Y_grid =np.meshgrid(X,Y)
  grid= np.vstack([X_grid.ravel(),Y_grid.ravel()]).T 
  W= [[np.real(SNR_ng_extr(sigma[j][i],[0,0]*2,[+1],sigma0[j])) for i in range(len(X))] for j in range(len(Y))]
  print(np.shape(W), type(W))
  fig,ax=plt.subplots(figsize=(10,6))
  c=ax.pcolormesh(X_grid,Y_grid,W,norm=mcolors.LogNorm(vmin=np.min(W), vmax=np.max(W)),cmap='jet')
  cbar=fig.colorbar(c,ax=ax, label='SNR extr')
  ax.set_xlim(X.min(), X.max())
  #ax.set_yscale('log')
  print(Y.min() , Y.max())
  ax.set_ylim(Y.min() , Y.max())
  ax.grid(True, which='both', linestyle='--')
  ax.set_xlabel('Squeezing parameter z', fontsize=22)
  ax.set_ylabel(r'Noise $\gamma$', fontsize=22)
  ax.set_xticks(ticks=[0,0.2,0.4,0.6,0.8,1], labels=['0','0.2','0.4','0.6','0.8',r'$|1\rangle$'])
  ax.set_yticks(ticks=[0.5,1,1.5,2], labels=['0.5','1','1.5','2'])
  c.set_label('SNR extr')
  cbar.ax.set_yticks(ticks=[1,2],labels=['1','2'])

  contour_levels = [1]
  contour = ax.contour(X_grid, Y_grid, W, levels=contour_levels, colors='black', linestyles='dashed', linewidths=1.5)
  ax.clabel(contour, inline=True, fontsize=10,fmt='Gaussian max')
  plt.savefig('density plot temp.pdf')
  plt.show()

def find_optimal_gaussian(t, theta): #finds the optimal squeezing, displacement parameters & optimal SNR for a certain temperature through the lagrange multipliers method
    nu = 1/np.tanh(1/(2* t))
    # Define the variable (symbol)
    z = sp.Symbol('z', real=True)
    k = sp.symbols('k', real=True)
    poly_expr = 1 - k*z**2 - (4*theta+2*k)*z**3 + k*z**4 #the polynomial that we input here is that given by the method of larange multipliers
    roots = sp.solve(poly_expr, z)
    #find which of the roots satisfies that it is real and within (0,1) by substituting at any k (e.g k=1)
    found_root=False
    root_index= None
    i=0
    while found_root == False:
        x= roots[i].subs({k:nu}) 
        if x.is_real == True:
            if np.float64(x) > 0 and np.float64(x) < 1:
                found_root =True
                root_index = i
        else:
            i +=1
    
    z_opt =roots[root_index].subs({k:nu})
    alpha_sq_opt= theta- (1/4)* nu* (z_opt + 1/z_opt -2)
    n_sq_opt=  (1/8)*nu**2*(z_opt**2 + 1/z_opt**2) -1/4 + nu*z_opt*alpha_sq_opt
    delta_n_opt = math.sqrt(n_sq_opt)
    optimal_snr = np.float64(theta/delta_n_opt)

    return optimal_snr

def plot_optimal_gaussian(t_vec,theta):
  alpha_vec=[]
  z_vec=[]
  for t in t_vec:
    nu = 1/np.tanh(1/(2* t))
    # Define the variable (symbol)
    z = sp.Symbol('z', real=True)
    k = sp.symbols('k', real=True)
    poly_expr = 1 - k*z**2 - (4*theta+2*k)*z**3 + k*z**4 #the polynomial that we input here is that given by the method of larange multipliers
    roots = sp.solve(poly_expr, z)
    #find which of the roots satisfies that it is real and within (0,1) by substituting at any k (e.g k=1)
    found_root=False
    root_index= None
    i=0
    while found_root == False:
        x= roots[i].subs({k:nu}) 
        if x.is_real == True:
            if np.float64(x) > 0 and np.float64(x) < 1:
                found_root =True
                root_index = i
        else:
            i +=1
    z_opt = roots[root_index].subs({k:nu})
    z_vec +=[z_opt]
    alpha_vec += [theta- (1/4)* nu* (z_opt + 1/z_opt -2)]
  plt.plot(t_vec,z_vec)
  plt.plot(t_vec,alpha_vec)
  plt.xlabel(r'$T[K]$')
  plt.legend([r'Squeezing parameter $z$',r'Displacement $|\alpha|^2$'])
  plt.title(r'Optimal Gaussian parameters for $\theta =1$')
  plt.show()


def snr_vs_stellar_rank(max_stellar_rank, max_temp, theta):
   #the ergotropy constraint is given by temperature and max_stellar_rank
  t_vec = np.linspace(0.1,max_temp,100)

  gauss_snr_opt =[log(find_optimal_gaussian(t, theta)) for t in t_vec]


  optimal_snr = []
  for i in range(max_stellar_rank+1):
    optimal_snr += [[]]
    i+= 1
  for t in t_vec:
    nu = 1/np.tanh(1/(2* t))
    for rank in range(1, max_stellar_rank+1):
      state = State(1,[random.random()],[],[random.random()],disp=[random.random(),random.random()], temp=[t],nongaussian_ops=[1]*rank, format='number')
      #print(state.__dict__)
      result= state.optimize_ratio(theta,1)
      while result.success == False:
        result= state.optimize_ratio(theta,1)
      optimal_snr[rank]+= [log(-result.fun)]
      optimal_state= State(1,[result.x[2]],[],[random.random()],disp=[result.x[0],result.x[1]], temp=[t],nongaussian_ops=[1]*rank, format='number')
  
  colors=['black','blue','orange','green']
  plt.plot(t_vec,gauss_snr_opt, color='black', linestyle='dashed')
  for rank in range(1, max_stellar_rank+1):
    plt.plot(t_vec, optimal_snr[rank], color= colors[rank])
  plt.legend(['Gaussian bound']+ [f'{rank} photon addition(s)' for rank in range(1,max_stellar_rank+1)])
  plt.xlabel(r'$T [K]$')
  plt.ylabel(r'$SNR_{ext, opt}$')
  plt.savefig('snr_with_stellar_rank.pdf')
  plt.show()
  return optimal_snr


def snr_sv_comparison(stellar_rank, max_temp):  # since we are studying bipartite entanglement, it is sufficient to consider 2 modes
   #the ergotropy constraint is given by temperature and max_stellar_rank

  def is_outlier(value, left, right, threshold=0.01):
    """
    Determines if a value is an outlier by comparing it to the average of neighboring values.
    """
    avg_neighbor = (left + right) / 2
    return abs(value - avg_neighbor) > threshold * avg_neighbor

  def replace_outliers_with_interpolation(data, threshold=0.01):
      """
      Detects and replaces outliers in a 2D list by interpolating neighboring values.
      """
      rows, cols = np.shape(data)
      for i in range(rows):
          for j in range(1, cols - 1):  # Avoid edges for simplicity
              value, left, right = data[i][j], data[i][j - 1], data[i][j + 1]
              if is_outlier(value, left, right, threshold):
                  data[i][j] = (left + right) / 2
      return data


  fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12, 6))
  t_vec = np.linspace(0.1,max_temp,100)
  x_vec=np.linspace(0,2*np.pi,100)
  X=x_vec
  Y=t_vec
  X_grid, Y_grid =np.meshgrid(X,Y)
  #create array for optimal SNR and SV of the corresponding state
  optimal_snr = []
  SV = []
  z=0.5
  for t in t_vec:
    optimal_snr += [[]]
    SV += [[]]
    nu = 1/np.tanh(1/(2* t))
    print('nu=', nu)
    for x in x_vec:
      optimal_state= State(2,[z,1/z],[x],[0,0],disp=[0,0,0,0], temp=[t,t],nongaussian_ops=[1]*stellar_rank, format='number')
      optimal_snr[np.where(t_vec == t)[0][0]]+= [np.float64(optimal_state.SNR_extr())]
      SV[np.where(t_vec == t)[0][0]]+= [-np.float64(np.real(optimal_state.SV()))]
  vmin_snr, vmax_snr = np.min(optimal_snr), np.max(optimal_snr)
  vmin_sv, vmax_sv = np.min(SV), np.max(SV)
  c1=ax1.pcolormesh(X_grid,Y_grid,optimal_snr, cmap='jet')
  c2=ax2.pcolormesh(X_grid,Y_grid,SV,cmap='jet')
  cbar1=fig.colorbar(c1,ax=ax1)
  cbar2=fig.colorbar(c2,ax=ax2)
  ax1.set_xlim(X.min(), X.max())
  ax1.set_ylim(Y.min() , Y.max())
  ax2.set_xlim(X.min(), X.max())
  ax2.set_ylim(Y.min() , Y.max())
  ax1.set_title('SNR extr')
  ax2.set_title('- SV')
  cbar1.ax.set_yticks(ticks=[vmin_snr, (vmin_snr + vmax_snr) / 2, vmax_snr])
  cbar2.ax.set_yticks(ticks=[vmin_sv, (vmin_sv + vmax_sv) / 2, vmax_sv])
  ax1.set_xlabel(r'Beamsplitter angle $\theta$')
  ax1.set_ylabel(r'$T[K]$')
  ax2.set_xlabel(r'Beamsplitter angle $\theta$')
  ax2.set_ylabel(r'$T[K]$')
  plt.subplots_adjust(wspace=0.9)
  plt.savefig(f'snr_entanglement {stellar_rank} phadd.pdf')
  plt.show()
  
  return 


def multimode_optimization(max_stellar_rank, max_temp, max_modes):
  t_vec = np.linspace(0.5,max_temp,10)
  colors = plt.cm.viridis(t_vec)
  N_vec = np.arange(1,max_modes+1)
  fig,axes = plt.subplots(2,2)
  for rank in range(0, max_stellar_rank+1):
    for i in range(len(t_vec)):
      optimal_snr=[]
      nu = 1/np.tanh(1/(2* t_vec[i]))
      for n in N_vec:
        state = State(n,[random.random()]*n,[2*np.pi*random.random()]*(n*(n-1)//2),[random.random()]*n,disp=[random.random(),random.random()]*n, temp=[t_vec[i]]*n,nongaussian_ops=[1]*rank, format='number')
        if rank ==0:
          result=state.optimize_ratio((nu**2)*n,n)
        else:
          result= state.optimize_ratio((nu**2+rank)*n,n)
        while result.success == False:
          if rank ==0:
            result=state.optimize_ratio((nu**2)*n,n)
          else:
            result= state.optimize_ratio((nu**2+rank)*n,n)
        optimal_snr+= [log(-result.fun)]
      axes[rank//2,rank%2].plot(N_vec,optimal_snr, color=colors[i])
      axes[rank//2,rank%2].set_xlabel('N')
      axes[rank//2,rank%2].set_xticks(ticks=np.arange(1,max_modes+1))
    rank+=1
  cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
  cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), cax=cbar_ax, location='right')
  cbar.set_label(r'$T[K]$')
  plt.show()
    
def minimum_energy_state(stellar_rank, maxiter=10): 
  #this function seeks to demonstrate that the one-mode 'thermal fock' state 
  #(state that results from successive photon additions and subtractions to the thermal state) is, 
  # for every stellar rank (trivially for gaussian states with stellar rank 0) the lowest energy state of that stellar rank
  fig, ax = plt.subplots(1,1)
  temp_vec= np.linspace(0.01,1, 100)
  for t in temp_vec:
    thermal_fock = State(1, [1],[], [0],disp=[0,0], temp=[t], nongaussian_ops=[1]*stellar_rank, required_ordering='xxpp', format='number')
    y = np.real(thermal_fock.expvalN())
    print(t,y)
    ax.scatter(t,y,s=2, color='b')
    for i in range(maxiter):
      state = State(1, [np.random.random()],[], [2*np.pi*np.random.random()],disp=[0,0], temp=[t], nongaussian_ops=[1]*stellar_rank, required_ordering='xxpp', format='number')
      y= np.real(state.expvalN())
      if y < 10*stellar_rank:
        ax.scatter(t,y,s=1, color='r')
      i+=1
  plt.show()

  return

def snr_with_constraints():
  max_stellar_rank =3
  temp_vec=np.linspace(0.1,1,10)
  nu_vec = [1/np.tanh(1/(2* t)) for t in temp_vec ]
  theta_vec = np.linspace(0.5,np.real(3*(nu_vec[-1]+1)),10)
  X=theta_vec
  Y=temp_vec
  X_grid, Y_grid =np.meshgrid(X,Y)
  fig,axes = plt.subplots(2,2)
  colors=['b','r','g']
  optimal_snr =[]


  #Gaussian case
  optimal_snr += [[]]
  z_opt_vec =[]
  alpha_sq_opt_vec=[]
  for i in range(len(temp_vec)):
    t= temp_vec[i]
    nu = 1/np.tanh(1/(2* t))
    optimal_snr[0] += [[]]
    for theta in theta_vec:
      z_opt = find_optimal_gaussian(t,theta)
      gauss_alpha_sq_opt =theta- (1/4)* nu* (z_opt + 1/z_opt -2) 
      z_opt_vec +=[z_opt]
      alpha_sq_opt_vec+=[gauss_alpha_sq_opt]
      n_sq_opt = (1/8)*nu**2*(z_opt**2 + 1/z_opt**2) -1/4 + nu*z_opt*gauss_alpha_sq_opt
      optimal_snr[0][i] += [np.log(np.float64(theta/n_sq_opt))]
    print('gaussian',i)

  #Non-gaussian case
  for rank in range(1, max_stellar_rank+1):
    axes[rank//2,rank%2].plot([rank*(1+n) for n in nu_vec],temp_vec, linestyle='dashed', color= 'black', linewidth=1)
    optimal_snr += [[]]
    for i in range(len(temp_vec)):
      t=temp_vec[i]
      nu= 1/np.tanh(1/(2* t))
      optimal_snr[rank] += [[]]
      for theta in theta_vec:
        if theta < rank*(1+nu):
          optimal_snr[rank][i]+= [np.nan]
        else:
          state = State(1,[random.random()],[],[random.random()],disp=[random.random(),random.random()], temp=[t],nongaussian_ops=[-1]*rank, format='number')
          result= state.optimize_ratio(theta,1)
          while result.success == False:
            result= state.optimize_ratio(theta,1)
          print(result.x,log(-result.fun))
          optimal_snr[rank][i]+= [np.log(-result.fun)]
      i+=1
      print(rank, i)
  vmin, vmax = np.nanmin(optimal_snr), np.nanmax(optimal_snr)
  print(vmin, vmax)
  for rank in range(max_stellar_rank+1):
    c= axes[rank//2,rank%2].pcolormesh(X_grid,Y_grid,optimal_snr[rank],vmin=vmin, vmax=vmax, cmap='jet')
    axes[rank//2,rank%2].set_xlabel(r'Ergotropy constraint $\theta$')
    axes[rank//2,rank%2].set_ylabel(r'$T[K]$')
    axes[rank//2,rank%2].set_title(f'Optimal SNR extr for {rank} photon adds')
    rank+=1
  fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for colorbar on the right
  cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
  fig.colorbar(mappable=c, cax=cbar_ax)  # Only one ScalarMappable is needed for colorbar
  #hacer una sola colorbar para todooo para q se vea mas claro quien gana
  plt.show()
  return



def feasible_regions(constraint_theta, system_temp):
  temp_vec=np.linspace(0.1,1,50)
  nu_vec = [1/np.tanh(1/(2* t)) for t in temp_vec ]
  minimum_ergotropy=[]
  colors=['black','r','b','g']
  for rank in range(0,4):
    minimum_ergotropy += [[]]
    for i in range(len(temp_vec)):
      minimum_ergotropy[rank] += [rank*((nu_vec[i]-1)/2+1)]
    plt.plot(minimum_ergotropy[rank],temp_vec, color=colors[rank])
  plt.fill([0, 0, constraint_theta, constraint_theta], [0, system_temp, system_temp, 0], color = 'yellow', alpha = 0.5)
  plt.xlabel(r'Ergotropy constraint $\theta$')
  plt.ylabel(r'T[K]')
  plt.legend(['gauss', '1','2','3'])
  plt.show()

def optimal_strategy():
  t=0.5
  n_th=(1/np.tanh(1/(2* t))-1)/2
  print('n_th=',n_th)
  theta_vec0=np.linspace(0.1,25,50)
  theta_vec1=np.linspace(n_th+1.0001,25,50)
  theta_vec2=np.linspace(2*(n_th+1.0001),25,50)
  theta_vec3=np.linspace(3*(n_th+1.0001),25,50)

  state_g= State(1,[np.random.random()],[],[0],disp=[np.random.random(), np.random.random()], temp=[t], nongaussian_ops=[])
  state_1pha= State(1,[np.random.random()],[],[0],disp=[np.random.random(), np.random.random()], temp=[t], nongaussian_ops=[1])
  state_2pha= State(1,[np.random.random()],[],[0],disp=[np.random.random(), np.random.random()], temp=[t], nongaussian_ops=[1,1])
  state_3pha= State(1,[np.random.random()],[],[0],disp=[np.random.random(), np.random.random()], temp=[t], nongaussian_ops=[1,1,1])

  result= [np.log(find_optimal_gaussian(t,theta)) for theta in theta_vec0]
  result1= [np.log(-state_1pha.optimize_ratio(theta, 1).fun) for theta in theta_vec1]
  result2= [np.log(-state_2pha.optimize_ratio(theta, 1).fun) for theta in theta_vec2]
  result3= [np.log(-state_3pha.optimize_ratio(theta, 1).fun) for theta in theta_vec3]
  plt.plot(theta_vec0,result, color='black', linestyle='dashed')
  plt.plot(theta_vec1,result1, color='blue')
  plt.plot(theta_vec2,result2, color='orange')
  plt.plot(theta_vec3,result3, color='green')
  plt.plot([theta_vec1[0], theta_vec1[0]], [result[0], result1[0]], color='blue', linestyle='--')
  plt.plot([theta_vec2[0], theta_vec2[0]], [result[0], result2[0]], color='orange', linestyle='--')
  plt.plot([theta_vec3[0], theta_vec3[0]], [result[0], result3[0]], color='green', linestyle='--')
  plt.legend(['Gaussian', '1 photon addition', '2 photon additions', '3 photon additions'])
  plt.xlabel(r'Maximum ergotropy $\theta$')
  plt.ylabel(r'Optimal $SNR_{extr}$')
  plt.show()
  
  return


def optimal_strategy2():
  temp_vec=np.linspace(0.4,1,3)
  nu_vec = [1/np.tanh(1/(2* t)) for t in temp_vec ]
  theta_vec = np.linspace(0.5,20,30)
  cmap=cm.rainbow
  norm = mcolors.Normalize(vmin=temp_vec.min(),vmax=temp_vec.max())
  fig, ax = plt.subplots(1,1)
  for t in temp_vec:
    i=np.where(temp_vec==t)
    nu= 1/np.tanh(1/(2* t))
    n_th=(nu -1)/2
    gaussian_snr_bound =[]
    gaussian_snr_bound2 =  []
    one_phadd =[]
    two_phadd= []
    three_phadd = []
    #optimal_strategy=[]
    index_1_ph = np.where(theta_vec > n_th+1)[0][0]  #identify the first value of theta where 1 photon addition can start to be applied
    index_2_ph = np.where(theta_vec > 2*(n_th+1))[0][0]  #identify the first value of theta where 2 photon additions can start to be applied
    index_3_ph = np.where(theta_vec > 3*(n_th+1))[0][0]  #identify the first value of theta where 3 photon additions can start to be applied
    print(index_1_ph,index_2_ph, index_3_ph)
    for theta in theta_vec:
      z_gauss, a_sq_gauss, n2_gauss, snr_opt_gauss = find_optimal_gaussian(t, theta)
      gaussian_snr_bound +=[np.log(snr_opt_gauss)]
      if n_th+1.000001 < theta :
        state = State(1,[random.random()],[],[random.random()],disp=[random.random(),random.random()], temp=[t],nongaussian_ops=[1], format='number')
        result= state.optimize_ratio(theta,1)
        while result.success == False:
          result= state.optimize_ratio(theta,1)
        one_phadd += [np.log(-result.fun)]
      if 2*(n_th+1.000001) < theta:
        state = State(1,[random.random()],[],[random.random()],disp=[random.random(),random.random()], temp=[t],nongaussian_ops=[1,1], format='number')
        result= state.optimize_ratio(theta,1)
        while result.success == False:
          result= state.optimize_ratio(theta,1)
        two_phadd += [np.log(-result.fun)]
      if 3*(n_th+1.000001)< theta:
        state = State(1,[random.random()],[],[random.random()],disp=[random.random(),random.random()], temp=[t],nongaussian_ops=[1,1,1], format='number')
        result= state.optimize_ratio(theta,1)
        while result.success == False:
          result= state.optimize_ratio(theta,1)
        three_phadd += [np.log(-result.fun)]
      print('t, theta', t, theta)
    ax.plot(theta_vec,gaussian_snr_bound,  color= cmap(norm(temp_vec[i]))) #we plot the gaussian bound
    ax.plot(theta_vec[index_1_ph:], one_phadd, linestyle='dashed', color= cmap(norm(temp_vec[i])))
    ax.plot(theta_vec[index_2_ph:], two_phadd, linestyle='dashdot', color= cmap(norm(temp_vec[i])))
    ax.plot(theta_vec[index_3_ph:], three_phadd, linestyle='dotted', color= cmap(norm(temp_vec[i])))


  cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, location='right') 
  cbar.set_label(r'Noise $\gamma$')  
  plt.show()

  return

#optimal_strategy()
#feasible_regions(2.5,0.5)
#snr_with_constraints()
#snr_with_constraints()
#minimum_energy_state(3, maxiter=10)
#snr_vs_stellar_rank(3,1,5)
#multimode_optimization(3,1.5,4)
snr_sv_comparison(2,1)
#plot_optimal_gaussian(np.linspace(0.01,1.5,200), 1)