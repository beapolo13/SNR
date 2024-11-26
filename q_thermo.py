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

#from utils import * 
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
    #cbar.ax.set_yticks(ticks=[1,2],labels=['1','2'])
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
    c1=axes[0].pcolormesh(X_grid,Y_grid,W1, norm=colors.Normalize(vmin=W1_arr.min(), vmax=W1_arr.max()),cmap=cm.get_cmap('viridis', 7) )
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

#the following function transforms a generic 2-mode covariance matrix (i think in xpxp ordering, but check!!) into standard form (which is the form of the local passive state of our input state)
#la ha hecha chat gpt y creo q está mal (revisar todo esto)
def to_standard_form(cm):

    def symplectic_omega(): #auxiliary function
        return np.array([[0, 1], [-1, 0]])  # 2x2 symplectic matrix

    def symplectic_matrix(n_modes): #auxiliary function
        omega = symplectic_omega()
        return np.kron(np.eye(n_modes), omega)
    

    # Ensure the covariance matrix is valid
    n = cm.shape[0] // 2  # Number of modes
    symplectic = symplectic_matrix(n)
    if not np.all(np.linalg.eigvals(cm + 1j * symplectic) >= 0):
        raise ValueError("Covariance matrix is not physical!")

    # Extract blocks A, B, and C
    A = cm[:2, :2]
    B = cm[2:, 2:]
    C = cm[:2, 2:]

    # Diagonalize local blocks (A and B)
    A_diag = sqrtm(A @ A.T)
    B_diag = sqrtm(B @ B.T)

    # Construct the symplectic transformation for diagonalization
    S_A = np.linalg.inv(sqrtm(A))  # For mode 1
    S_B = np.linalg.inv(sqrtm(B))  # For mode 2
    S_local = np.block([
        [S_A, np.zeros_like(S_A)],
        [np.zeros_like(S_B), S_B],
    ])

    # Apply the symplectic transformation
    cm_transformed = S_local @ cm @ S_local.T

    # Recompute A, B, C after diagonalization
    A_new = cm_transformed[:2, :2]
    B_new = cm_transformed[2:, 2:]
    C_new = cm_transformed[:2, 2:]

    # Simplify correlations in C_new
    c1, c2 = C_new[0, 0], C_new[1, 1]  # Extract canonical correlation values

    # Construct the standard form
    standard_form = np.block([
        [A_new, np.diag([c1, c2])],
        [np.diag([c1, c2]), B_new],
    ])
    print(standard_form)
    return standard_form

#mutual_information_TMSQ()
#relative_ergotropic_gap_TMSQ()

#two_mode_squeezed_state = State()