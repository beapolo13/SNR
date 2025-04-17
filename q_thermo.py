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
import copy

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

#heatmaps_pure_state(np.pi/4)


def gaussian_mixed_bound(n_shots):
    k_vec=np.linspace(1,10,300)
    bound_vec=[]
    for k in k_vec:
        bound_vec+=[(-k+k**2/2+1/2)/(k-1)]
    plt.plot(k_vec, bound_vec, linestyle='dashed', color='black')

    entangled_state_count=0
    for i in range(n_shots):
        k1= 1+9*np.random.random()
        k2= 1+9*np.random.random()
        #k=float(max([k1,k2]))
        k=(k1+k2)/2
        x= 2 * np.pi* np.random.random()
        z1= np.random.random()
        z2= np.random.random()
        sep= z1*z2*(1+k1**2*k2**2-k1**2-k2**2)-4*cos(x)**2*sin(x)**2*(k1*k2*(z1**2+z2**2)-(k1**2+k2**2)*(z1*z2))
        erg_gap = (-(k1+k2)/2 + (1/2)*(sqrt(k2**2*cos(x)**4+k1**2*sin(x)**4+k1*k2*cos(x)**2*sin(x)**2*((z1**2+z2**2)/(z1*z2)))+sqrt(k1**2*cos(x)**4+k2**2*sin(x)**4+k1*k2*cos(x)**2*sin(x)**2*((z1**2+z2**2)/(z1*z2)))))/((k1+k2-2)/2)
        if sep < 0:
            entangled_state_count +=1 
            print(f'not separable state {entangled_state_count}')
            if erg_gap < 100: 
                plt.scatter(k,erg_gap, c='b', s=1)
        else:
            plt.scatter(k,erg_gap, c='r', s=1)
        i+=1
    plt.show()
    return

def gaussian_mixed_new_bound(n_shots):
    fig,ax = plt.subplots()
    entangled_state_count=0
    pure_count=0
    vacuum_count=0
    for i in range(n_shots):
        w = 9*np.random.random()
        alpha = 1 + 9* np.random.rand()
        t2=9*np.random.random()
        t1= t2/alpha+ 9*np.random.random()
    
        k1= 1/np.tanh((w/t1))
        k2= 1/np.tanh((w*alpha/t2))

        k=(k1+k2)/2
        if k1 < k2:
            print('not valid')
            continue
        if np.isclose(k,1) and np.isclose(k1,1) and np.isclose(k2,1):
            pure_count += 1
            #print('pure state')
            #continue
        
        gamma=(k1-k2)/2

        if np.isclose(k,1) and (np.isclose(gamma,0) or np.isclose(alpha,1)):
            print('vacuum')
            vacuum_count+=1
            continue

        x= 2 * np.pi* np.random.random()
        z1= np.random.random()
        z2= np.random.random()
        #print('t1,t2',t1,t2,'k1,k2,k:',k, 'gamma', gamma, 't2/t1',t2/t1,'alpha',alpha,'x',x,'z1,z2',z1,z2, 'w', w)
        r1= ((k1*z2*cos(x)**2+k2*z1*sin(x)**2)/((z1*z2)*(k1*z1*cos(x)**2+k2*z2*sin(x)**2)))**(1/4)
        r2= ((k2*z1*cos(x)**2+k1*z2*sin(x)**2)/((z1*z2)*(k2*z2*cos(x)**2+k1*z1*sin(x)**2)))**(1/4)

        #gp= State(2,[1,1],[0],[0,0],[w, alpha*w],[0,0,0,0],[t1,t2],None,'xxpp','number')
        #print('global passive matrix', gp.matrix)
        #state= State(2,[z1,z2],[x],[0,0],[w, alpha*w],[0,0,0,0],[t1,t2],None,'xxpp','number')
        #print('initial matrix', state.matrix)
        #print('initial energy', state.energy())
        #loc_op=np.array([[r1,0,0,0],[0, r2,0,0],[0,0,  1/r1,0],[0,0,0,1/r2]])
        #loc_passive_mat= loc_op @ state.matrix @ loc_op.T
        #lp= (1/4)*(w*(loc_passive_mat[0,0]+loc_passive_mat[2,2]-2))+(1/4)*(alpha*w*(loc_passive_mat[1,1]+loc_passive_mat[3,3]-2))
        #print('gp',gp.energy())
        #print('lp',lp)
        #gap= (lp-gp.energy())/gp.energy()
        #print('gap', gap)
        sep= z1*z2*(1+k1**2*k2**2-k1**2-k2**2)-4*cos(x)**2*sin(x)**2*(k1*k2*(z1**2+z2**2)-(k1**2+k2**2)*(z1*z2))
        #sep2= (k1**2-1)*(k2**2-1)*(4*z1*z2*(3+cos(4*x))+(z1-z2)**2*cosh(2*(w/t1+alpha*w/t2))*(-1+cos(4*x))+2*(z1+z2)**2*cosh(2*(w/t1-w*alpha/t2))*sin(2*x)**2)
        #if np.sign(sep) != np.sign(sep2):
            #print('error!')
        
       
        erg_gap = (-(k*(1+alpha)+ gamma*(1-alpha)) + sqrt((k+gamma)**2*cos(x)**4+(k-gamma)**2*sin(x)**4+(k**2-gamma**2)*cos(x)**2*sin(x)**2*((z1**2+z2**2)/(z1*z2)))+alpha*sqrt((k-gamma)**2*cos(x)**4+(k+gamma)**2*sin(x)**4+(k**2-gamma**2)*cos(x)**2*sin(x)**2*((z1**2+z2**2)/(z1*z2))))/((k-1)*(1+alpha)+ gamma*(1-alpha))
        bound= (-(k*(1+alpha)+ gamma*(1-alpha))+((1+alpha)/2)*sqrt(1+k**4-2*k**2*gamma**2+gamma**4+2*(k**2+gamma**2)+8*k*gamma))/((k-1)*(1+alpha)+ gamma*(1-alpha))
        #if np.abs(erg_gap-gap)>10**(-4):
            #print('error!')

        if sep < 0:
            entangled_state_count +=1 
            ax.scatter(i,bound-erg_gap, c='b', s=1)
        else:
            #print(f'{i}: {bound-erg_gap}')
            if bound-erg_gap <0:
                print(i, 'data:','t1,t2',t1,t2,'k:',k, 'gamma', gamma, 't2/t1',t2/t1,'alpha',alpha,'x',x,'z1,z2',z1,z2, 'w', w)
            ax.scatter(i,bound-erg_gap, c='r', s=1)
            
                
        i+=1
    print(f'number of not separable states: {entangled_state_count}')
    print(f'Pure state count {pure_count}')
    print(f'Vacuum ground state count {vacuum_count}')
    ax.axhline(y=0, xmin=0, xmax=n_shots, color='black', linestyle='-')
    ax.set_yscale('symlog')

    plt.xlabel('Number of iterations')
    plt.ylabel(r'Bound - $\Delta \epsilon_{r e l}$')
    plt.savefig('Bound_violation_separable_vs_entangled.pdf')
    plt.show()
    return


def heatmap_bound(alpha):
    k_vec=np.linspace(1.0001,10,1000)
    gamma_vec=np.linspace(0,9,1000)
    #gamma_vec =np.array([0, 0.00005])
    X=k_vec
    Y=gamma_vec
    X_grid, Y_grid =np.meshgrid(X,Y)
    W= [[np.float64((-(k*(1+alpha)+ gamma*(1-alpha))+((1+alpha)/2)*sqrt(1+k**4-2*k**2*gamma**2+gamma**4+2*(k**2+gamma**2)+8*k*gamma))/((k-1)*(1+alpha)+ gamma*(1-alpha))) for k in k_vec] for gamma in gamma_vec]
    for i in range(len(k_vec)):
        for j in range(len(gamma_vec)):
            if k_vec[i]-1 < gamma_vec[j]:
                print('yes')
                W[j][i] = np.nan
    print(np.nanmin(W), np.nanmax(W))

    fig,ax=plt.subplots(figsize=(10,6))
    c=ax.pcolormesh(X_grid,Y_grid,W,cmap=cm.get_cmap('viridis', 100),norm=mcolors.Normalize(vmin=np.nanmin(W), vmax=np.nanmax(W)))
    cbar=fig.colorbar(c,ax=ax, label=r'Bound on $\Delta \epsilon_{r e l}$ for separable states')
    #ax.set_xlim(X.min(), X.max())
    #ax.set_yscale('log')
    #ax.set_ylim(Y.min() , Y.max())
    #ax.grid(True, which='both', linestyle='--')
    ax.set_xlabel(r'Mean temperature factor $k$')
    ax.set_ylabel(r'Fluctuation gap $\gamma$')
    #ax.set_xscale('log')
    x=np.arange(k_vec[0], k_vec[-1],1).tolist()
    y =np.arange(gamma_vec[0], gamma_vec[-1],1).tolist()
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.text(0.25, 0.75, r'non-physical values \ \ $k-\gamma < 1$', fontsize=12,
        transform=ax.transAxes) 
    c.set_label(r'Bound on $\Delta \epsilon_{r e l}$ for separable states')
    plt.savefig(f'Sep_bound_alpha={alpha}.pdf')
    plt.show()
    return

def bound_violation_tms(alpha, gamma):
    z_vec=np.linspace(0.1,1,500)
    r_vec=np.array([-np.log(z)/2 for z in z_vec])
    k_vec= np.linspace(1.001+gamma/2,10,500)
    X=z_vec
    Y=k_vec
    X_grid, Y_grid =np.meshgrid(X,Y)
    x= np.pi/5
    epsilon = 1e-6

    
    #separability
    
    sep= [[np.float64((1+k**4+gamma**4-2*k**2*gamma**2-2*k**2-2*gamma**2)-4*cos(x)**2*sin(x)**2*((k**2-gamma**2)*(z**2+1/z**2)-(2*k**2+2*gamma**2))) for z in z_vec] for k in k_vec] 
    sep_arr=np.array(sep)
    diff=[[np.float64((-(k*(1+alpha)+ gamma*(1-alpha))+((1+alpha)/2)*sqrt(1+k**4-2*k**2*gamma**2+gamma**4+2*(k**2+gamma**2)+8*k*gamma))/((k-1)*(1+alpha)+ gamma*(1-alpha))-(-(k*(1+alpha)+ gamma*(1-alpha)) + sqrt((k+gamma)**2*cos(x)**4+(k-gamma)**2*sin(x)**4+(k**2-gamma**2)*cos(x)**2*sin(x)**2*((z**2+1/z**2)/(1)))+alpha*sqrt((k-gamma)**2*cos(x)**4+(k+gamma)**2*sin(x)**4+(k**2-gamma**2)*cos(x)**2*sin(x)**2*((z**2+1/z**2)/(1))))/((k-1)*(1+alpha)+ gamma*(1-alpha)))for z in z_vec]for k in k_vec]
    W_arr= np.array(diff)
    W = list(W_arr)
 
    fig,ax=plt.subplots(1,2,figsize=(10,6))
    c=ax[0].pcolormesh(X_grid,Y_grid,W,norm=colors.SymLogNorm(0.0000001,vmin=min(W_arr+epsilon), vmax=W_arr.max()),cmap=cm.get_cmap('viridis', 10))
    #c=ax[0].pcolormesh(X_grid,Y_grid,W,cmap=cm.get_cmap('viridis', 10))
    cbar=fig.colorbar(c,ax=ax[0], label=r'Bound - $\Delta \epsilon_{r e l}$ for TMS states')
    contour_levels = [0]
    contour = ax[0].contour(X_grid, Y_grid, W, levels=contour_levels, colors='black', linestyles='dashed', linewidths=1.5)
    #ax[0].clabel(contour, inline=True, fontsize=10,fmt='ERG')
    ax[0].set_xlim(X.min(), X.max())
    ax[0].set_yscale('log')
    ax[0].set_ylim(Y.min() , Y.max())
    #ax.grid(True, which='both', linestyle='--')
    ax[0].set_ylabel(r'Temperature factor $k$')
    ax[0].set_xlabel(r'Squeezing parameter $z$')
    
    c.set_label(r'Bound - $\Delta \epsilon_{r e l}$ for TMS states')

    c2=ax[1].pcolormesh(X_grid,Y_grid,sep,norm=colors.SymLogNorm(0.00001, vmin=min(sep_arr+epsilon), vmax=sep_arr.max()),cmap=cm.get_cmap('viridis', 10))

    #c2=ax[1].pcolormesh(X_grid,Y_grid,sep,cmap=cm.get_cmap('viridis', 10))
    cbar=fig.colorbar(c,ax=ax[1], label=r'2-mode Gaussian separability condition')
    contour_levels = [0]
    contour = ax[1].contour(X_grid, Y_grid, sep, levels=contour_levels, colors='black', linestyles='dashed', linewidths=1.5)
    #ax[1].clabel(contour, inline=True, fontsize=10,fmt='PPT')
    ax[1].set_xlim(X.min(), X.max())
    ax[1].set_yscale('log')
    ax[1].set_ylim(Y.min() , Y.max())
    #ax.grid(True, which='both', linestyle='--')
    ax[1].set_ylabel(r'Temperature factor $k$')
    ax[1].set_xlabel(r'Squeezing parameter $z$')

    c2.set_label(r'2-mode Gaussian separability condition')
    plt.savefig(f'Sep_and_bound_violtion_tms_alpha={alpha}, gamma={gamma}.pdf')
    plt.subplots_adjust(wspace=2)
    y=[gamma/2+1] + [(gamma/2+1)//1 + i for i in range(1,8)] + [10]
    ax[0].set_yticks(y)
    ax[1].set_yticks(y)
    plt.show()
    return
         

from qutip import wigner, Qobj, basis, states, thermal_dm, tensor, wigner, displace,squeeze


def plot_onemodegaussian():
    # Define system parameters
    N = 30  # Hilbert space truncation (higher for better precision)
    T = 0.2  # Temperature parameter (controls mixedness)

    r = 0.8  # Squeezing parameter
    alpha = 1.0j #Displacement parameter (complex)

    # Generate a thermal state (Gaussian mixed state)
    rho_thermal = thermal_dm(N, T)

    # Apply squeezing transformation
    S = squeeze(N, r)  # Squeezing operator
    rho_squeezed = S * rho_thermal * S.dag()  # Squeezed thermal state

    # Apply displacement transformation
    D = displace(N, alpha)  # Displacement operator
    rho = D * rho_squeezed * D.dag()  # Displaced squeezed thermal state
    # Define phase-space grid
    xvec = np.linspace(-5, 5, 200)
    yvec = np.linspace(-5, 5, 200)

    # Compute the Wigner function
    W = wigner(rho, xvec, yvec)

    # Create a meshgrid for 3D plotting
    X, Y = np.meshgrid(xvec, yvec)

    # Create a 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Wigner function
    ax.plot_surface(X, Y, W, cmap="viridis", edgecolor='none', alpha=0.8)

    # Labels and titles
    ax.set_title("3D Wigner Function of a Gaussian Mixed State (Thermal)")
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_zlabel("Wigner Function")

    # Show the plot
    plt.show()

def local_passive_energy(state, params): #implements a local Gaussian general operation (a minimization of this function leads to the local passive)
    r1=params[0]
    r2=params[1]
    loc_op=np.array([[r1,0,0,0],[0, r2,0,0],[0,0,  1/r1,0],[0,0,0,1/r2]])
    loc_passive_mat= loc_op @ state.matrix @ loc_op.T
    state_copy = copy.deepcopy(state)

    state_copy.matrix = loc_passive_mat
    lp_energy= np.real(state_copy.expvalE())
    #print('initial state energy', state.expvalE(), 'new state energy', state_copy.expvalE())
    #lp_energy = (1/4)*(state.omega[0]*(loc_passive_mat[0,0]+loc_passive_mat[2,2]-2))+(1/4)*(state.omega[1]*(loc_passive_mat[1,1]+loc_passive_mat[3,3]-2))
    return lp_energy
    
def global_passive_energy(state, params): #implements a global Gaussian general operation (a minimization of this function leads to the global passive)
      
    r1=params[0]
    r2=params[1]
    #theta=params[2]
    angle= params[2]
    loc_op=np.array([[r1,0,0,0],[0, r2,0,0],[0,0,  1/r1,0],[0,0,0,1/r2]])
    glob_op= np.array([[cos(angle),sin(angle),0, 0],[-sin(angle), cos(angle),0,0,],[0,0, cos(angle),sin(angle)],[0,0,-sin(angle),cos(angle)]])
    passive_mat1= glob_op.T @ state.matrix @ glob_op
    
    passive_mat= loc_op.T @ passive_mat1 @ loc_op
    state_copy = copy.deepcopy(state)
    state_copy.matrix = passive_mat
    gp_energy= np.real(state_copy.expvalE())
    #print('initial state energy', state.expvalE(), 'new state energy', state_copy.expvalE())
    #gp_energy = (1/4)*(state.omega[0]*(passive_mat[0,0]+passive_mat[2,2]-2))+(1/4)*(state.omega[1]*(passive_mat[1,1]+passive_mat[3,3]-2))
    return gp_energy


def experimental_optimization(n_samples,n_modes=2): 
    #samples is the amount of different random gaussian states that we'll try the optimization on
    #so far we'll do it for 2 modes and then generalize
    def generate_random_gaussian():
        w = 9*np.random.random()
        alpha = 1 + 9* np.random.rand()
        t2=9*np.random.random()
        t1= t2/alpha+ 9*np.random.random()
        k1= 1/np.tanh((w/t1))
        k2= 1/np.tanh((w*alpha/t2))
        k=(k1+k2)/2
        gamma=(k1-k2)/2
        x= 2 * np.pi* np.random.random()
        z1= np.random.random()
        z2= np.random.random()
        #print('t1,t2',t1,t2,'k1,k2,k:',k, 'gamma', gamma, 't2/t1',t2/t1,'alpha',alpha,'x',x,'z1,z2',z1,z2, 'w', w)
        r1= ((k1*z2*cos(x)**2+k2*z1*sin(x)**2)/((z1*z2)*(k1*z1*cos(x)**2+k2*z2*sin(x)**2)))**(1/4)
        r2= ((k2*z1*cos(x)**2+k1*z2*sin(x)**2)/((z1*z2)*(k2*z2*cos(x)**2+k1*z1*sin(x)**2)))**(1/4)
        
        return State(2,[z1,z2],[x],[0,0],[w, alpha*w],[0,0,0,0],[t1,t2],None,'xxpp','number')
    

    for s in range(n_samples):
        copies= 0
        success = False
        state= generate_random_gaussian()
        k1,k2,z1,z2,x = 1/np.tanh((state.omega[0]/state.temp[0])), 1/np.tanh((state.omega[1]/state.temp[1])), state.squeezing[0], state.squeezing[1], state.bs[0]
        

        print('State parameters z1,z2,x', z1,z2,x)
        print('Initial energy', state.energy())

        print('Local passive search')
        true_r1= ((k1*z2*cos(x)**2+k2*z1*sin(x)**2)/((z1*z2)*(k1*z1*cos(x)**2+k2*z2*sin(x)**2)))**(1/4)
        true_r2= ((k2*z1*cos(x)**2+k1*z2*sin(x)**2)/((z1*z2)*(k2*z2*cos(x)**2+k1*z1*sin(x)**2)))**(1/4)
        print('True squeezing parameters', true_r1, true_r2 )
        iteration_count = [0]  # Store iteration count (as list to modify in callback)
        def callback(params):
            """Counts the number of iterations."""
            iteration_count[0] += 1
        opti_lp= minimize(lambda params: local_passive_energy(state, params), x0=(1.0,1.0), bounds=[(1e-6, None), (1e-6, None)], method='Nelder-Mead', callback=callback)
        print('Optimization parameters:',opti_lp.x)
        if np.isclose(true_r1, opti_lp.x[0], 0.05) and np.isclose(true_r2, opti_lp.x[1], 0.05):
            print(f'Local search succeeded in {iteration_count[0]} copies')
            print('Local passive energy', opti_lp.fun)
            copies += iteration_count[0]
        else: 
            print('Failed local search')
            continue

        print('Global passive search')
        true_gp_energy= State(2,[1,1],[0],[0,0],state.omega,[0,0,0,0],state.temp,None,'xxpp','number').energy()
        print('True global passive energy:', true_gp_energy)
        iteration_count = [0]  # Store iteration count (as list to modify in callback)
        def callback(params):
            """Counts the number of iterations."""
            iteration_count[0] += 1
        opti_gp= minimize(lambda params: global_passive_energy(state, params), x0=(1.0,1.0), bounds=[(1e-6, None), (1e-6, None)], method='Nelder-Mead', callback=callback)
        print('Optimization result (energy):',opti_gp.fun)
        if np.isclose(true_gp_energy, opti_gp.fun,0.1):
            print(f'Global search succeeded in {iteration_count[0]} copies')
            copies += iteration_count[0]
        else: 
            print('Failed global search')
            continue
        return
    
def local_passive_phadd(state, params): #implements a local Gaussian general operation (a minimization of this function leads to the local passive) on a photon added state on mode 1
    r1=params[0]
    r2=params[1]
    loc_op=np.array([[r1,0,0,0],[0, r2,0,0],[0,0,  1/r1,0],[0,0,0,1/r2]])
    loc_passive_mat= loc_op @ state.matrix @ loc_op.T
    state_copy = copy.deepcopy(state)
    state_copy.matrix = loc_passive_mat
    #print(state_copy.matrix)
    K= state_copy.expectationvalue([],[])
    N_a= state_copy.expectationvalue(['adag','a'], [1,1])
    N_b= state_copy.expectationvalue(['adag','a'], [2,2])
    lp_energy= np.real((N_a+N_b)/K)
    #print('initial state energy', state.expvalE(), 'new state energy', state_copy.expvalE())
    #return N_b
    return lp_energy

def local_passive_phadd_analytical(z,k,grid_size): #I have checked that this function is absolutely equivalent to the non-analytical one
    r1_vec=np.linspace(0.01,1,grid_size)
    r2_vec=np.linspace(0.01,1,grid_size)
    def lp_energy_phadd(z,k,r1,r2):
        N_a = (k/(8*z))**2*(1+z**4+2*z**2)*(3*r1**4+3/r1**4+2)+(k/(8*z))*(1+z**2)*(r1**2+1/r1**2)
        N_b= 2*(k/(8*z))**2*(1-z**2)**2*(r1**2*r2**2+1/(r1**2*r2**2)) + (k/(8*z))**2*(1+z**2)**2*(r1**2*r2**2+1/(r1**2*r2**2)+(r1/r2)**2+(r2/r1)**2 ) + (k/(16*z))*(1+z**2)*(r2**2+1/r2**2-r1**2-1/r1**2)-0.25
        K= (k/(8*z))*(1+z**2)*(r1**2+1/r1**2)+0.5
        return np.real((N_a+N_b)/K)
    min_lp_energy = np.min([[lp_energy_phadd(z,k,r1,r2) for r1 in r1_vec ]for r2 in r2_vec])
    return min_lp_energy
    

    
def global_passive_phadd(state,grid_size): #insert a Gaussian (no photon additions) state as input variable!!!
    #implements a global Gaussian general operation (a minimization of this function leads to the global passive)
    r1_vec=np.linspace(0.01,1,grid_size)
    r2_vec=np.linspace(0.01,1,grid_size)
    angle_vec=np.linspace(0.01,2*np.pi,grid_size)  
    def gp_energy_phadd(state,r1,r2,angle):
        loc_op=np.array([[r1,0,0,0],[0, r2,0,0],[0,0,  1/r1,0],[0,0,0,1/r2]])
        glob_op= np.array([[cos(angle),sin(angle),0, 0],[-sin(angle), cos(angle),0,0,],[0,0, cos(angle),sin(angle)],[0,0,-sin(angle),cos(angle)]])
        G= loc_op @ glob_op
        passive_mat= G @ state.matrix @ G.T
        
        state_copy = copy.deepcopy(state)
        state_copy.matrix = passive_mat

        gp_energy= G[0,0]**2 * state_copy.expectationvalue(['a','adag','a','adag'], [1,1,1,1])+ G[0,0]*G[0,1] * state_copy.expectationvalue(['a','adag','a','adag'], [2,1,1,1]) + G[0,1]*G[0,0] * state_copy.expectationvalue(['a','adag','a','adag'], [1,1,1,2]) +  G[0,1]**2 * state_copy.expectationvalue(['a','adag','a','adag'], [2,1,1,2])
        gp_energy+= G[0,0]**2 * state_copy.expectationvalue(['a','adag','a','adag'], [1,2,2,1])+ G[0,0]*G[0,1] * state_copy.expectationvalue(['a','adag','a','adag'], [2,2,2,1]) + G[0,1]*G[0,0] * state_copy.expectationvalue(['a','adag','a','adag'], [1,2,2,2]) +  G[0,1]**2 * state_copy.expectationvalue(['a','adag','a','adag'], [2,2,2,2])
        K= G[0,0]**2 * state_copy.expectationvalue(['a','adag'], [1,1])+ G[0,0]*G[0,1] * state_copy.expectationvalue(['a','adag'], [2,1]) + G[0,1]*G[0,0] * state_copy.expectationvalue(['a','adag'], [1,2]) +  G[0,1]**2 * state_copy.expectationvalue(['a','adag'], [2,2])
        gp_energy= np.real(gp_energy/K)
        return gp_energy
    min_gp_energy = np.min([[[gp_energy_phadd(state,r1,r2,angle) for r1 in r1_vec ]for r2 in r2_vec]for angle in angle_vec])
    return min_gp_energy
    
def find_ergotropic_gap_phadd(state, grid_size=None):
    print('Initial energy', state.expvalE())
    print('Local passive search')
    def lp_energy_phadd(state,params):
        z=state.squeezing[0]
        t=state.temp[0]
        k= 1/np.tanh(1/(2*t))
        (r1, r2) = params
        N_a = (k/(8*z))**2*(1+z**4+2*z**2)*(3*r1**4+3/r1**4+2)+(k/(8*z))*(1+z**2)*(r1**2+1/r1**2)
        N_b= 2*(k/(8*z))**2*(1-z**2)**2*(r1**2*r2**2+1/(r1**2*r2**2)) + (k/(8*z))**2*(1+z**2)**2*(r1**2*r2**2+1/(r1**2*r2**2)+(r1/r2)**2+(r2/r1)**2 ) + (k/(16*z))*(1+z**2)*(r2**2+1/r2**2-r1**2-1/r1**2)-0.25
        K= (k/(8*z))*(1+z**2)*(r1**2+1/r1**2)+0.5
        return np.real((N_a+N_b)/K)
    
    opti_lp= minimize(lambda params: lp_energy_phadd(state, params), x0=(0.5,0.5), bounds=[(1e-6, None), (1e-6, None)], method='COBYLA')
    print('Optimization parameters:',opti_lp.x)
    print('Local passive energy', opti_lp.fun)

    print('Global passive search')
    def gp_energy_phadd(state,params):
        (r1,r2,angle) =params
        loc_op=np.array([[r1,0,0,0],[0, r2,0,0],[0,0,  1/r1,0],[0,0,0,1/r2]])
        glob_op= np.array([[cos(angle),sin(angle),0, 0],[-sin(angle), cos(angle),0,0,],[0,0, cos(angle),sin(angle)],[0,0,-sin(angle),cos(angle)]])
        G= loc_op @ glob_op
        passive_mat= G @ state.matrix @ G.T
        
        state_copy = copy.deepcopy(state)
        state_copy.matrix = passive_mat

        gp_energy= G[0,0]**2 * state_copy.expectationvalue(['a','adag','a','adag'], [1,1,1,1])+ G[0,0]*G[0,1] * state_copy.expectationvalue(['a','adag','a','adag'], [2,1,1,1]) + G[0,1]*G[0,0] * state_copy.expectationvalue(['a','adag','a','adag'], [1,1,1,2]) +  G[0,1]**2 * state_copy.expectationvalue(['a','adag','a','adag'], [2,1,1,2])
        gp_energy+= G[0,0]**2 * state_copy.expectationvalue(['a','adag','a','adag'], [1,2,2,1])+ G[0,0]*G[0,1] * state_copy.expectationvalue(['a','adag','a','adag'], [2,2,2,1]) + G[0,1]*G[0,0] * state_copy.expectationvalue(['a','adag','a','adag'], [1,2,2,2]) +  G[0,1]**2 * state_copy.expectationvalue(['a','adag','a','adag'], [2,2,2,2])
        K= G[0,0]**2 * state_copy.expectationvalue(['a','adag'], [1,1])+ G[0,0]*G[0,1] * state_copy.expectationvalue(['a','adag'], [2,1]) + G[0,1]*G[0,0] * state_copy.expectationvalue(['a','adag'], [1,2]) +  G[0,1]**2 * state_copy.expectationvalue(['a','adag'], [2,2])
        gp_energy= np.real(gp_energy/K)
        return gp_energy
    opti_gp= minimize(lambda params: gp_energy_phadd(state, params), x0=(0.5,0.5,np.pi/4), bounds=[(1e-6, None), (1e-6, None), (0, 2*np.pi)], method='Nelder-Mead')
    print('Optimization parameters:',opti_gp.x)
    print('Optimization result (energy):',opti_gp.fun)
    print('')
    final_result = np.real((opti_lp.fun-opti_gp.fun)/opti_gp.fun)
    print('FINAL RESULT', final_result )
    print('')
    return np.real(final_result)
    


def find_ergotropic_gap_phadd_version2(state, grid_size):
    z=state.squeezing[0]
    t=state.temp[0]
    k=1/np.tanh(1/(2*t))
    lp= local_passive_phadd_analytical(z,k, grid_size)
    gp=global_passive_phadd(state,grid_size)
    relative_gap=(lp-gp)/gp 
    print(lp,gp, 'relative gap=', relative_gap )
    return relative_gap


    
def nongaussian_erg_gap(nongaussian_ops, gaussian_parameters=None):
    def create_state(nongaussian_ops, gaussian_parameters=None):
        if gaussian_parameters is not None:
            z1, z2, x, w, alpha, t1, t2 = gaussian_parameters
        else:
            w = 9*np.random.random()
            alpha = 1 + 9* np.random.rand()
            t2=9*np.random.random()
            t1= t2/alpha+ 9*np.random.random()
            x= 2 * np.pi* np.random.random()
            z1= np.random.random()
            z2= np.random.random()
        k1= 1/np.tanh((w/t1))
        k2= 1/np.tanh((w*alpha/t2))
        k=(k1+k2)/2
        gamma=(k1-k2)/2
        state=  State(2,[z1,z2],[x],[0,0],[w, alpha*w],[0,0,0,0],[t1,t2],nongaussian_ops,'xxpp','number')
        print('Initial state energy', state.expvalE())
        return state
    
    def global_passive_energy(state,params):
        r1=params[0]
        r2=params[1]
        theta=params[2]
        theta= state.bs[0]
        initial_matrix=state.matrix
        #print('Initial energy', state.expvalE())
        loc_op=np.array([[r1,0,0,0],[0, r2,0,0],[0,0,  1/r1,0],[0,0,0,1/r2]])
        glob_op= np.array([[cos(theta),sin(theta),0, 0],[-sin(theta), cos(theta),0,0,],[0,0, cos(theta),sin(theta)],[0,0,-sin(theta),cos(theta)]])
        passive_mat1= glob_op.T @ initial_matrix @ glob_op
        passive_mat= loc_op.T @ passive_mat1 @ loc_op
        state.matrix = passive_mat
        gp_energy = state.expvalE()
        #print('Global passive energy', gp_energy)
        state.matrix = initial_matrix
        return gp_energy
    
    def local_passive_energy(state, params):
        r1=params[0]
        r2=params[1]
        initial_matrix=state.matrix
        #print('Initial energy', state.expvalE())
        loc_op=np.array([[r1,0,0,0],[0, r2,0,0],[0,0,  1/r1,0],[0,0,0,1/r2]])
        loc_passive_mat= loc_op @ state.matrix @ loc_op.T
        state.matrix = loc_passive_mat
        lp_energy = state.expvalE()
        #print('Local passive energy', lp_energy)
        state.matrix = initial_matrix
        return lp_energy 
    
    state= create_state(nongaussian_ops, gaussian_parameters)
    opti_lp= minimize(lambda params: local_passive_energy(state, params), x0=(1.0,1.0), bounds=[(1e-6, None), (1e-6, None)], method='COBYLA')
    print('Optimization result (local):',opti_lp.fun)
    opti_gp= minimize(lambda params: global_passive_energy(state, params), x0=(1.0,1.0, 0.0), bounds=[(1e-6, None), (1e-6, None), (0, 2*np.pi)], method='Nelder-Mead')
    print('Optimization result (energy):',opti_gp.fun)
    ergotropic_gap= opti_lp.fun - opti_gp.fun
    print('Ergotropic gap', ergotropic_gap, 'Relative erg gap', ergotropic_gap/opti_gp.fun )
    return ergotropic_gap/opti_gp.fun

def plot_nongaussian_erg_gap(nongaussian_ops, gamma,alpha):
    z_vec=np.linspace(0.1,1,100)
    k_vec= np.linspace(1.001+gamma/2,10,100)
    X=z_vec
    Y=k_vec
    X_grid, Y_grid =np.meshgrid(X,Y)
    x= np.pi/4
    epsilon = 1e-6

    
    erggap= [[ nongaussian_erg_gap(nongaussian_ops,(z,1/z,x,1,1*alpha,k,k)) for z in z_vec] for k in k_vec] 
    W_arr= np.array(erggap)
    W = list(W_arr)
 
    fig,ax=plt.subplots(1,2,figsize=(10,6))
    c=ax[0].pcolormesh(X_grid,Y_grid,W,norm=colors.SymLogNorm(0.0000001,vmin=min(W_arr+epsilon), vmax=W_arr.max()),cmap=cm.get_cmap('viridis', 10))
    print(min(W_arr+epsilon), W_arr.max())
    #c=ax[0].pcolormesh(X_grid,Y_grid,W,cmap=cm.get_cmap('viridis', 10))
    cbar=fig.colorbar(c,ax=ax[0], label=r'Ergotorpic gap for nongaussian states')
    #ax[0].clabel(contour, inline=True, fontsize=10,fmt='ERG')
    ax[0].set_xlim(X.min(), X.max())
    ax[0].set_yscale('log')
    ax[0].set_ylim(Y.min() , Y.max())
    #ax.grid(True, which='both', linestyle='--')
    ax[0].set_ylabel(r'Temperature factor $k$')
    ax[0].set_xlabel(r'Squeezing parameter $z$')
    
    c.set_label(r'Bound - $\Delta \epsilon_{r e l}$ for TMS states')

    c2=ax[1].pcolormesh(X_grid,Y_grid,W,norm=colors.SymLogNorm(0.00001, vmin=min(W_arr+epsilon), vmax=W_arr.max()),cmap=cm.get_cmap('viridis', 10))

    #c2=ax[1].pcolormesh(X_grid,Y_grid,sep,cmap=cm.get_cmap('viridis', 10))
    cbar=fig.colorbar(c,ax=ax[1], label=r'2-mode Gaussian separability condition')
    #ax[1].clabel(contour, inline=True, fontsize=10,fmt='PPT')
    ax[1].set_xlim(X.min(), X.max())
    ax[1].set_yscale('log')
    ax[1].set_ylim(Y.min() , Y.max())
    #ax.grid(True, which='both', linestyle='--')
    ax[1].set_ylabel(r'Temperature factor $k$')
    ax[1].set_xlabel(r'Squeezing parameter $z$')

    c2.set_label(r'SV criterio')
    #plt.savefig()
    plt.subplots_adjust(wspace=2)
    y=[gamma/2+1] + [(gamma/2+1)//1 + i for i in range(1,8)] + [10]
    ax[0].set_yticks(y)
    ax[1].set_yticks(y)
    beep()
    plt.show()
    return

    



def photon_add_tms():
    z_vec=np.linspace(0.1,1,50)
    
    r_vec=np.array([-np.log(z)/2 for z in z_vec])
    t_vec = np.linspace(0.1,10,50)
    print(z_vec, t_vec)
    
    k_vec= np.array([1/np.tanh((1/(2*t))) for t in t_vec])
    X=z_vec
    Y=k_vec
    X_grid, Y_grid =np.meshgrid(X,Y)
    x= np.pi/4
    epsilon = 1e-6

    
    sv = [[np.float64(State(2, [z,1/z],[x],[0,0],None, None, [t_vec[j],t_vec[j]],[1]).SV()) for z in z_vec] for j in range(len(k_vec))]
    sv_arr = np.array(sv)
    sv_simp = [[(k**3*(z+1/z)-k**2*(z**2-2+1/z**2)-k*(z+1/z))/(2*k*(z+1/z)+4) for z in z_vec] for k in k_vec]
    sv_simp = np.array(sv_simp)
    print('array difference',np.argmax(np.abs(sv-sv_simp)), np.max(np.abs(sv-sv_simp)))
    print('min sv', np.min(sv_arr), np.argmin(sv_arr), 'max sv', np.max(sv_arr))
    
    W = [[np.float64(find_ergotropic_gap_phadd(State(2, [z,1/z],[x],[0,0],None, None, [t_vec[j],t_vec[j]]),15)) for z in z_vec] for j in range(len(k_vec))]
    W_arr= np.array(W)
    print(W)
    data = [sv, sv_simp,W]
    with open('entanglement_ergogap.pkl', 'wb') as file:
      pickle.dump(data,file)


    fig,ax=plt.subplots(1,2,figsize=(10,6))
    c=ax[0].pcolormesh(X_grid,Y_grid,sv_simp,norm=colors.SymLogNorm(0.0000001,vmin=min(sv_simp+epsilon), vmax=sv_simp.max()),cmap=cm.get_cmap('viridis', 20))
    #c=ax[0].pcolormesh(X_grid,Y_grid,W,cmap=cm.get_cmap('viridis', 10))
    cbar=fig.colorbar(c,ax=ax[0], label=r'2-mode SV separability condition')
    contour_levels = [0]
    contour = ax[0].contour(X_grid, Y_grid, sv_simp, levels=contour_levels, colors='black', linestyles='dashed', linewidths=1.5)
    #ax[0].clabel(contour, inline=True, fontsize=10,fmt='ERG')
    ax[0].set_xlim(X.min(), X.max())
    ax[0].set_yscale('log')
    ax[0].set_ylim(Y.min() , Y.max())
    #ax.grid(True, which='both', linestyle='--')
    ax[0].set_ylabel(r'Temperature factor $k$')
    ax[0].set_xlabel(r'Squeezing parameter $z$')
    
    c.set_label(r'2-mode SV separability condition')

    c2=ax[1].pcolormesh(X_grid,Y_grid,W,cmap=cm.get_cmap('viridis', 30))

    #c2=ax[1].pcolormesh(X_grid,Y_grid,sep,cmap=cm.get_cmap('viridis', 10))
    cbar=fig.colorbar(c2,ax=ax[1], label=r'Ergotropic gap for TMS photon-added states')
    contour_levels = [0]
    contour = ax[1].contour(X_grid, Y_grid, W,levels=contour_levels, colors='black', linestyles='dashed', linewidths=1.5)
    #ax[1].clabel(contour, inline=True, fontsize=10,fmt='PPT')
    ax[1].set_xlim(X.min(), X.max())
    ax[1].set_yscale('log')
    ax[1].set_ylim(Y.min() , Y.max())
    #ax.grid(True, which='both', linestyle='--')
    ax[1].set_ylabel(r'Temperature factor $k$')
    ax[1].set_xlabel(r'Squeezing parameter $z$')

    c2.set_label(r'Ergotropic gap for TMS photon-added states')
    plt.subplots_adjust(wspace=2)
    y=[1] + [1 + i for i in range(1,8)] + [10]
    ax[0].set_yticks(y)
    ax[1].set_yticks(y)
    beep()
    #plt.savefig(f'Photon-added TMS sep condition vs rel ergotropic gap.pdf')
    plt.show()
    return

photon_add_tms()
#plot_onemodegaussian()
#bound_violation_tms(1,1)
#heatmap_bound(1)

#gaussian_mixed_new_bound(10000)
#one_dim_plot_squeezing_pure(np.pi/4)
#mutual_information_TMSQ()
#relative_ergotropic_gap_TMSQ()

#experimental_optimization(1)





