import numpy as np
from numpy import transpose, real, sqrt, sin, cos, linalg, cosh, sinh, log, log2, min, max
import scipy
import math
import sympy as sp
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import optimize
from scipy.optimize import minimize
from scipy.linalg import sqrtm
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
#from plots import *

params = {'axes.linewidth': 2,
         'axes.labelsize': 15,
         'axes.titlesize': 15,
         'axes.linewidth': 2,
         'lines.markeredgecolor': "black",
     	'lines.linewidth': 2,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10,
         "text.usetex": True,
         "font.serif": ["Palatino"],
         "font.family": "serif"
         }
plt.rcParams.update(params)

def mutual_information_TMSQ(): #this plots the mutual information for the specific case of the two-mode squeezed state (do it for generic state)
    r_vec=np.linspace(0,1.5,1000)
    k_vec=np.linspace(1,10,1000)
    X=r_vec
    Y=k_vec
    X_grid, Y_grid =np.meshgrid(X,Y)
    W= [[2*(((k*cosh(2*r)+1)/2)*log2((k*cosh(2*r)+1)/2)-((k*cosh(2*r)-1)/2)*log2((k*cosh(2*r)-1)/2)-((k+1)/2)*log2((k+1)/2)+((k-1)/2)*log2((k-1)/2)) for r in r_vec] for k in k_vec]
    fig,ax=plt.subplots(figsize=(10,6))
    c=ax.pcolormesh(X_grid,Y_grid,W,cmap=cm.get_cmap('viridis', 7) )
    cbar=fig.colorbar(c,ax=ax, label='Mutual information')
    ax.set_xlim(X.min(), X.max())
    #ax.set_yscale('log')
    print(Y.min() , Y.max())
    ax.set_ylim(Y.min() , Y.max())
    ax.grid(True, which='both', linestyle='--')
    ax.set_xlabel('Squeezing parameter r', fontsize=22)
    ax.set_ylabel('k', fontsize=22)
    ax.set_xticks(ticks=[0,0.2,0.4,0.6,0.8,1, 1.2,1.4], labels=['0','0.2','0.4','0.6','0.8','1','1.2','1.4'])
    ax.set_yticks(ticks=[2,3,4,5,6,7,8,9], labels=['2','3','4','5','6','7','8','9'])
    c.set_label('I(A:B)')
    plt.show()



def relative_ergotropic_gap_TMSQ():  #this is only defined for the two-mode-squeezed state, generalize it for any state!!
    r_vec=np.linspace(0.001,1.5,1000)
    k_vec=np.linspace(1.001,10,1000)
    X=r_vec
    Y=k_vec
    X_grid, Y_grid =np.meshgrid(X,Y)
    fig,axes = plt.subplots(1,2)
    W1= [[(2*k*sinh(r)**2) for r in r_vec] for k in k_vec]
    W1_arr = np.array(W1)
    W2= [[(2*k*sinh(r)**2)/(k-1) for r in r_vec] for k in k_vec]
    W2_arr = np.array(W2)
    c1=axes[0].pcolormesh(X_grid,Y_grid,W1, norm=colors.LogNorm(vmin=W1_arr.min(), vmax=W1_arr.max()),cmap=cm.get_cmap('viridis', 7) )
    cbar1=fig.colorbar(c1,ax=axes[0], label='Ergotropic gap')
    c2=axes[1].pcolormesh(X_grid,Y_grid,W2, norm=colors.LogNorm(vmin=W2_arr.min(), vmax=W2_arr.max()),cmap=cm.get_cmap('viridis', 7) )
    cbar2=fig.colorbar(c2,ax=axes[1], label='Relative ergotropic gap')
    for i in range(2):
        axes[i].set_xlim(X.min(), X.max())
        axes[i].set_ylim(Y.min() , Y.max())
        axes[i].grid(True, which='both', linestyle='--')
        axes[i].set_xlabel('Squeezing parameter r', fontsize=22)
        axes[i].set_ylabel('k', fontsize=22)
        axes[i].set_xticks(ticks=[0,0.2,0.4,0.6,0.8,1, 1.2,1.4], labels=['0','0.2','0.4','0.6','0.8','1','1.2','1.4'])
        axes[i].set_yticks(ticks=[2,3,4,5,6,7,8,9], labels=['2','3','4','5','6','7','8','9'])
    
    plt.show()

def pure_ergotropic_gap(state): #returns the ergotropic gap of a 2-mode gaussian pure states with parameters z1,z2 and theta
    z1=state.squeezing[0]
    z2=state.squeezing[1]
    theta = state.bs[0]
    return sqrt((z2*cos(theta)**2 + z1*sin(theta)**2)*(z1*cos(theta)**2 + z2*sin(theta)**2)/(z1*z2))-1

def pure_mutual_info(state): #returns the ergotropic gap of a 2-mode gaussian pure states with parameters z1,z2 and theta
    z1=state.squeezing[0]
    z2=state.squeezing[1]
    theta = state.bs[0]
    argument=(z1**2 + 6*z1*z2 + z2**2 - (z1 - z2)**2*cos(4*theta))/(z1*z2)
    #print(argument, type(argument))
    return float((1/log(16))*(-8*log(8) +
    (4 - sqrt(2) * sqrt(argument)) * log(float(-4 + sqrt(2) * sqrt(argument))) + (4 + sqrt(2) * sqrt(argument))*log(float(4 + sqrt(2) * sqrt(argument))) ))

def one_dim_plot_squeezing_pure(fixed_theta):
    z_vec = np.linspace(0.0001,0.999999,1000)
    ergotropic_gap=[]
    mutual_info = []
    for z1 in z_vec:
        r = -log(z1)/2
        z2 = 1/z1
        state = State(2,[z1,z2],[fixed_theta],[0,0])
        print(state.bs[0])
        ergotropic_gap +=[pure_ergotropic_gap(state)]
        mutual_info += [pure_mutual_info(state)]
    plt.plot(z_vec,ergotropic_gap)
    #plt.show()
    plt.plot(z_vec,mutual_info)
    #plt.legend(['log Ergotropic gap','Mutual information'])
    plt.show()

def h(w):
    w=float(w)
    x=w+1
    return (x+1)*log2((x+1)/2)-(x-1)*log2((x-1)/2)


def heatmaps_pure_state(fixed_theta):
    z1_vec = np.linspace(0.0001,0.9999,100)
    z2_vec = np.linspace(0.0001,0.9999,100)
    X=z1_vec
    Y=z2_vec
    X_grid, Y_grid =np.meshgrid(X,Y)
    fig,axes = plt.subplots(1,2)
    W1= [[float(pure_ergotropic_gap(State(2,[z1,z2],[fixed_theta],[0,0]))) if z1 != z2 else 0 for z1 in z1_vec] for z2 in z2_vec]
    W1_arr= np.array(W1)
    W2= [[float(pure_mutual_info(State(2,[z1,z2],[fixed_theta],[0,0]))) if z1 != z2 else 0 for z1 in z1_vec] for z2 in z2_vec]
    W2_arr = np.array(W2)
    epsilon = 1e-6
    c1=axes[0].pcolormesh(X_grid,Y_grid,W1, norm=colors.LogNorm(vmin=min(W1_arr+epsilon), vmax=W1_arr.max()),cmap=cm.get_cmap('viridis', 10) )
    cbar1=fig.colorbar(c1,ax=axes[0], label='Ergotropic gap')
    c2=axes[1].pcolormesh(X_grid,Y_grid,W2, norm=colors.LogNorm(vmin=min(W2_arr+epsilon), vmax=W2_arr.max()),cmap=cm.get_cmap('viridis', 10) )
    cbar2=fig.colorbar(c2,ax=axes[1], label='Mutual info')
    for i in range(2):
        axes[i].set_xlim(X.min(), X.max())
        axes[i].set_ylim(Y.min() , Y.max())
        axes[i].grid(True, which='both', linestyle='--')
        axes[i].set_xlabel(r'Squeezing parameter $z_1$', fontsize=22)
        axes[i].set_ylabel(r'Squeezing parameter $z_2$', fontsize=22)
        axes[i].set_xticks(ticks=[0,0.2,0.4,0.6,0.8,1], labels=['0','0.2','0.4','0.6','0.8','1'])
        #axes[i].set_yticks(ticks=[0,0.2,0.4,0.6,0.8,1], labels=['0','0.2','0.4','0.6','0.8','1'])
    
    plt.show()

heatmaps_pure_state(np.pi/4)

#one_dim_plot_squeezing_pure(np.pi/4)
#mutual_information_TMSQ()
#relative_ergotropic_gap_TMSQ()


