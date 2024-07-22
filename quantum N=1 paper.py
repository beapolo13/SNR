#Checks that the functions in the Friis & Huber paper can be reproduced with our code
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
import matplotlib.colors as mcolors

from utils import *
from expectation_values import *


def gaussian_displacement(function):
    z_vec=np.linspace(0.01,0.95,50)
    displacement=np.linspace(0,2,25)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    cmap=cm.rainbow
    norm = mcolors.Normalize(vmin=z_vec.min(),vmax=z_vec.max())
    for i in range(len(z_vec)):
        yvec= [function(0,0,alpha,0,z_vec[i]) for alpha in displacement]
        ax.plot(displacement,yvec, color=cmap(norm(z_vec[i])))
    ax.plot(displacement, [1]*len(displacement), linestyle='dashed')
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, location='right') 
    cbar.set_label('Squeezing parameter')
    ax.set_xlabel(r'Displacement $\alpha$')
    ax.set_ylabel(f'{function}')
    plt.savefig('snr div by delta gaussian with displacement.pdf')
    plt.show()
    return

def z(r):
    return np.exp(-2*r)

def r(z):
    return -np.ln(z)/2

def ergotropy_gaussian(sigma,sigma0,displacement):  
    N=len(sigma)//2
    return expvalN(sigma,displacement)-expvalN(sigma0,[0]*(2*N))
                                                                  
def std_dev_gaussian(sigma,sigma0,displacement):
    N=len(sigma)//2
    return varianceN(sigma,displacement)- varianceN(sigma0,[0,0]) 

def figure_of_merit_gaussian(sigma,sigma0,displacement):
    return ergotropy_gaussian(sigma,sigma0,displacement)/std_dev_gaussian(sigma,sigma0,displacement)**2

def analytical_ergotropy_gaussian(nu,r,displacement): #only for N=1
    return 0.5*(nu*(np.cosh(2*r)-1)+2*(displacement[0]**2+displacement[1]**2))

def analytical_variance_gaussian(nu,r,displacement):  #only for N=1
    return 0.25*((nu**2*np.cosh(4*r))-1+4*nu*(displacement[0]**2*np.exp(-2*r)+displacement[1]**2*np.exp(2*r)))

def analytical_std_dev_gaussian(nu,r,displacement): #only for N=1
    return np.sqrt(analytical_variance_gaussian(nu,r,displacement))-np.sqrt(analytical_variance_gaussian(nu,0,[0,0]))

def figure5():
    nu=1

    #squeezing only
    r_vec=np.linspace(0,1.5,100)
    displacement=[0,0]
    x1=[analytical_ergotropy_gaussian(nu,r,displacement) for r in r_vec]
    y1=[analytical_std_dev_gaussian(nu,r,displacement) for r in r_vec]
    sigma=[V_thermal(nu,[z(r)],[0],[0],params=None) for r in r_vec]
    sigma0=V_thermal(nu,[z(0)],[0],[0],params=None)
    x1_num=[ergotropy_gaussian(sigma[i],sigma0,displacement) for i in range(len(sigma))]
    y1_num=[std_dev_gaussian(sigma[i],sigma0,displacement) for i in range(len(sigma))]
    plt.plot(x1,y1)
    plt.plot(x1_num,y1_num)

    #displacement only
    disp_vec=np.linspace(0,40,100)
    r=0
    x2=[analytical_ergotropy_gaussian(nu,r,[d,0]) for d in disp_vec]
    y2=[analytical_std_dev_gaussian(nu,r,[d,0]) for d in disp_vec]
    sigma=V_thermal(nu,[z(0)],[0],[0],params=None)
    sigma0=V_thermal(nu,[z(0)],[0],[0],params=None)
    x2_num=[ergotropy_gaussian(sigma,sigma0,[d,0]) for d in disp_vec]
    y2_num=[std_dev_gaussian(sigma,sigma0,[d,0]) for d in disp_vec]
    plt.plot(x2,y2)
    plt.plot(x2_num,y2_num)

    #now we plot the optimal case. the way to do it is: find the r that yields the minimum variance for each displacement, and then with that r calculate ergotropy
    y3=[]
    x3=[]
    y3_num=[]
    x3_num=[]
    snr=[]
    for d in disp_vec:
        array=[analytical_std_dev_gaussian(nu,r,[d,0]) for r in r_vec]
        sigma=[V_thermal(nu,[z(r)],[0],[0],params=None) for r in r_vec]
        array_num=[std_dev_gaussian(sigma[i],sigma0,[d,0]) for i in range(len(sigma))]
        lowest_std_dev=min(array)
        lowest_std_dev_num=min(array_num)
        index=array.index(lowest_std_dev)
        index_num=array_num.index(lowest_std_dev_num)
        y3+=[lowest_std_dev]
        x3+=[analytical_ergotropy_gaussian(nu,r_vec[index],[d,0])]
        y3_num+=[lowest_std_dev_num]
        x3_num+=[ergotropy_gaussian(sigma[index_num],sigma0,[d,0])]
        snr+=[figure_of_merit_gaussian(sigma[index_num],sigma0,[d,0])]

    plt.plot(x3,y3, linestyle='dashed')
    plt.plot(x3_num,y3_num, linestyle='dashed')
    plt.legend(['Squeezing only analytical', 'Squeezing only','Displacement only analytical','Displacement only','Minimum analytical' ,'Minimum'])
    plt.show()

    plt.plot(disp_vec,snr)
    plt.show()
    #from this last graph we have observed that if we plot snr (the one we have defined now, which is which the variance squared) against the displacement for a VERY large range (eg from 0 to 400), we obtain the curve flattens>>
    #snr is bounded for gaussian states, and so we cannot go any further
figure5()

def gaussian_displacement_ergotropy(temp):  #same function as above but using loop haffnian
    displacement=np.linspace(0,2,25)
    z_vec=np.linspace(0.1,0.95,25)
    sigma0=V_thermal(temp,[1],[0],[0],params=None)
    plt.rc('font', family='serif')
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(30, 15))
    cmap=cm.rainbow
    norm = mcolors.Normalize(vmin=z_vec.min(),vmax=z_vec.max())
    for i in range(len(z_vec)):
        sigma=V_thermal(temp,[z_vec[i]],[0],[0],params=None)
        yvec= [ergotropy(sigma,sigma0,[d,0]) for d in displacement]
        yvec2= [0.5*(temp*(cosh(-log(z_vec[i]))-1)+d**2) for d in displacement]
        ax1.plot(displacement,yvec, color=cmap(norm(z_vec[i])))
        ax2.plot(displacement,yvec2, color=cmap(norm(z_vec[i])))
    ax1.plot(displacement, [1]*len(displacement), linestyle='dashed')
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=(ax1,ax2), location='right') 
    cbar.set_label('Squeezing parameter')
    ax1.set_xlabel(r'Displacement $\alpha$')
    ax1.set_ylabel('ergotropy')
    ax2.set_xlabel(r'Displacement $\alpha$')
    ax2.set_ylabel('analytical ergotropy')
    plt.show()
    return

def gaussian_displacement_variance(temp):  #same function as above but using loop haffnian  #1 MODE ONLY
    displacement=np.linspace(0,2,25)
    z_vec=np.linspace(0.1,0.95,25)
    sigma0=V_thermal(temp,[1],[0],[0],params=None)
    plt.rc('font', family='serif')
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(30, 15))
    cmap=cm.rainbow
    norm = mcolors.Normalize(vmin=z_vec.min(),vmax=z_vec.max())
    for i in range(len(z_vec)):
        sigma=V_thermal(temp,[z_vec[i]],[0],[0],params=None)
        yvec= [(varianceN(sigma,[d,0]))**2 for d in displacement]
        yvec2= [0.25*(temp**2*cosh(-2*log(z_vec[i]))-1+2*temp*(z_vec[i]*d**2)) for d in displacement]
        ax1.plot(displacement,yvec, color=cmap(norm(z_vec[i])))
        ax2.plot(displacement,yvec2, color=cmap(norm(z_vec[i])))
    ax1.plot(displacement, [1]*len(displacement), linestyle='dashed')
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=(ax1,ax2), location='right') 
    cbar.set_label('Squeezing parameter')
    ax1.set_xlabel(r'Displacement $\alpha$')
    ax1.set_ylabel('variance')
    ax2.set_xlabel(r'Displacement $\alpha$')
    ax2.set_ylabel('analytical variance')
    plt.show()
    return

def snr_with_displacement(temp):  #same function as above but using loop haffnian  #1 MODE ONLY
    displacement=np.linspace(0,100,25)
    z_vec=np.linspace(0.01,0.95,25)
    sigma0=V_thermal(temp,[1],[0],[0],params=None)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    cmap=cm.rainbow
    norm = mcolors.Normalize(vmin=z_vec.min(),vmax=z_vec.max())
    for i in range(len(z_vec)):
        sigma=V_thermal(temp,[z_vec[i]],[0],[0],params=None)
        yvec= [SNR_gaussian_extr(sigma,sigma0,[d,0]) for d in displacement]
        ax.plot(displacement,yvec, color=cmap(norm(z_vec[i])))
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, location='right') 
    cbar.set_label('Squeezing parameter')
    ax.set_xlabel(r'Displacement $\alpha$')
    ax.set_ylabel('snr_ext')
    plt.show()
    return

#snr_with_displacement(6)
# gaussian_displacement_ergotropy(1)
# gaussian_displacement_variance(1)