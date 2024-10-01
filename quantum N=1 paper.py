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

def z(r):
    return np.exp(-2*r)

def r(z):
    return -np.log(z)/2

# Now we define the functions described in the paper analytically     
                                                         
def analytical_ergotropy_gaussian(nu,r,displacement): #only for N=1
    return 0.5*(nu*(np.cosh(2*r)-1)+2*(displacement[0]**2+displacement[1]**2))

def analytical_variance_gaussian(nu,r,displacement):  #only for N=1
    return 0.25*((nu**2*np.cosh(4*r))-1+4*nu*(displacement[0]**2*np.exp(-2*r)+displacement[1]**2*np.exp(2*r)))

def analytical_std_dev_gaussian(nu,r,displacement): #only for N=1
    return np.sqrt(analytical_variance_gaussian(nu,r,displacement))-np.sqrt(analytical_variance_gaussian(nu,0,[0,0]))


#We'll define a function to check that the analytical functions from the paper coincide with our numerical ones
#Everything is calculated for N=1
def paper_sanity_check(temp):  
    displacement=np.linspace(0,4,30)
    z_vec=np.linspace(0.1,0.95,25)
    sigma0=V_thermal(temp,[1],[0],[0],params=None)
    plt.rc('font', family='serif')
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(30, 15))
    cmap=cm.rainbow
    norm = mcolors.Normalize(vmin=z_vec.min(),vmax=z_vec.max())
    for i in range(len(z_vec)):
        sigma=V_thermal(temp,[z_vec[i]],[0],[0],params=None)
        y1= [ergotropy(sigma,sigma0,[d,0]) for d in displacement]
        y1_analytical= [analytical_ergotropy_gaussian(temp,r(z_vec[i]),[d,0]) for d in displacement]
        ax1.plot(displacement,y1, color=cmap(norm(z_vec[i])))
        ax1.plot(displacement,y1_analytical, color=cmap(norm(z_vec[i])), linestyle='dashed')
        y2= [varianceN(sigma,[d,0])**2 for d in displacement]
        y2_analytical= [analytical_variance_gaussian(temp,r(z_vec[i]),[d,0]) for d in displacement]
        ax2.plot(displacement,y2, color=cmap(norm(z_vec[i])))
        ax2.plot(displacement,y2_analytical, color=cmap(norm(z_vec[i])), linestyle='dashed')
        y3= [std_dev(sigma,sigma0,[d,0]) for d in displacement]
        y3_analytical= [analytical_std_dev_gaussian(temp,r(z_vec[i]),[d,0]) for d in displacement]
        ax3.plot(displacement,y3, color=cmap(norm(z_vec[i])))
        ax3.plot(displacement,y3_analytical, color=cmap(norm(z_vec[i])), linestyle='dashed')
    
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=(ax1,ax2,ax3), location='right') 
    cbar.set_label('Squeezing parameter')
    ax1.set_xlabel(r'Displacement $\alpha$')
    ax1.set_ylabel('Ergotropy')
    ax2.set_xlabel(r'Displacement $\alpha$')
    ax2.set_ylabel('Variance')
    ax3.set_xlabel(r'Displacement $\alpha$')
    ax3.set_ylabel('Standard deviation')
    plt.show()
    return




#finally we reproduce their results using our functions (figure 5, more specifically)
def figure5(nu):
    N=1
    #squeezing only
    r_vec=np.linspace(0.1,1.4,100)
    displacement=[0,0]
    x1=[analytical_ergotropy_gaussian(nu,r,displacement) for r in r_vec]
    y1=[analytical_std_dev_gaussian(nu,r,displacement) for r in r_vec]
    sigma=[V_thermal(nu,[z(r)],[0],[0],params=None) for r in r_vec]
    sigma0=V_thermal(nu,[z(0)],[0],[0],params=None)
    x1_num=[ergotropy(sigma[i],sigma0,displacement) for i in range(len(sigma))]
    y1_num=[std_dev(sigma[i],sigma0,displacement) for i in range(len(sigma))]
    plt.plot(x1,y1)
    plt.plot(x1_num,y1_num)

    #displacement only
    disp_vec=np.linspace(0,2.1,100)
    r=0
    x2=[analytical_ergotropy_gaussian(nu,r,[d,0]) for d in disp_vec]
    y2=[analytical_std_dev_gaussian(nu,r,[d,0]) for d in disp_vec]
    sigma=V_thermal(nu,[z(0)],[0],[0],params=None)
    sigma0=V_thermal(nu,[z(0)],[0],[0],params=None)
    x2_num=[ergotropy(sigma,sigma0,[d,0]) for d in disp_vec]
    y2_num=[std_dev(sigma,sigma0,[d,0]) for d in disp_vec]
    plt.plot(x2,y2)
    plt.plot(x2_num,y2_num)

    #now we plot the optimal case. the way to do it is: find the r that yields the minimum variance for each displacement, and then with that r calculate ergotropy
    y3=[]
    x3=[]
    y3_num=[]
    x3_num=[]
    for d in disp_vec:
        stdev=[analytical_std_dev_gaussian(nu,r,[d,0]) for r in r_vec]
        sigma=[V_thermal(nu,[z(r)],[0],[0],params=None) for r in r_vec]
        stdev_num=[std_dev(sigma[i],sigma0,[d,0]) for i in range(len(sigma))]
        lowest_std_dev=min(stdev)
        lowest_std_dev_num=min(stdev_num)
        index=stdev.index(lowest_std_dev)
        index_num=stdev_num.index(lowest_std_dev_num)
        y3+=[lowest_std_dev]
        x3+=[analytical_ergotropy_gaussian(nu,r_vec[index],[d,0])]
        y3_num+=[lowest_std_dev_num]
        x3_num+=[ergotropy(sigma[index_num],sigma0,[d,0])]


    plt.plot(x3,y3,linestyle='dashed')
    plt.plot(x3_num,y3_num,linestyle='dashed')
    plt.legend(['Squeezing only analytical', 'Squeezing only','Displacement only analytical','Displacement only','Gaussian Minimum analytical' ,'Gaussian Minimum'])
    plt.show()




#now we're just going to focus on the gaussian minimum (optimal situation that can be achieved with gaussian states given a certain temperature, defined by nu) N=1
#we'll also compare it with some other nongaussian cases
def snr_vs_energy_comparison(nu):
    N=1
    r_vec=np.linspace(0.0,4,100)
    disp_vec=np.linspace(0,1.7,100)
    nongaussian_ops_vec=[[1]]
    sigma0=V_thermal(nu,[z(0)],[0],[0],params=None)
    #find the r that yields the minimum variance for each displacement, and then with that r calculate all quantities
    e_gauss=[]
    snr_gauss=[]
    g=[]
    #gaussian
    for d in disp_vec:
        sigma=[V_thermal(nu,[z(r)],[0],[0],params=None) for r in r_vec]
        variance=[varianceN(sigma[i],[d,0]) for i in range(len(sigma))]
        g_array_gauss=[antibunching_one_mode(sigma[i],[d,0],[]) for i in range(len(sigma))]
        snr_array=[SNR_gaussian_extr(sigma[i],sigma0,[d,0]) for i in range(len(sigma))]

        #find optimal values of r for each displacement
        lowest_var=min(variance)
        lowest_g=min(g_array_gauss)
        best_snr=max(snr_array)

        #find the indices of those optimal values
        index_var=variance.index(lowest_var)
        index_g=g_array_gauss.index(lowest_g)
        index_snr=snr_array.index(best_snr)
       

        e_gauss+=[ergotropy(sigma[index_var],sigma0,[d,0])]
        snr_gauss+=[best_snr]
        g+=[lowest_g]
    
    plt.plot(disp_vec,snr_gauss,linestyle='dashed')

    #non-gaussian
    for nongaussian_ops in nongaussian_ops_vec:
        e_ng=[]
        snr_ng=[]
        g=[]
        for d in disp_vec:
            sigma=[V_thermal(nu,[z(r)],[0],[0],params=None) for r in r_vec]
            variance_ng=[varianceN_ng(sigma[i],[d,0],nongaussian_ops) for i in range(len(sigma))]
            g_array_ng=[antibunching_one_mode(sigma[i],[d,0],nongaussian_ops) for i in range(len(sigma))]
            snr_ng_array=[SNR_ng_extr(sigma[i],[d,0],nongaussian_ops,sigma0) for i in range(len(sigma))]

            #find optimal values of r for each displacement
            lowest_var_ng=min(variance_ng)
            lowest_g_ng=min(g_array_ng)
            best_snr_ng=max(snr_ng_array)

            #find the indices of those optimal values
            index_var_ng=variance_ng.index(lowest_var_ng)
            index_g_ng=g_array_ng.index(lowest_g_ng)
            index_snr_ng=snr_ng_array.index(best_snr_ng)
        

            e_ng+=[ergotropy_ng(sigma[index_snr_ng],sigma0,[d,0],nongaussian_ops)]
            snr_gauss+=[best_snr]
            snr_ng+=[best_snr_ng]
            g+=[lowest_g_ng]
        plt.plot(disp_vec,snr_ng)

    plt.legend(['SNR Gauss']+[f'Optimal SNR for {len(item)} photonadd' for item in nongaussian_ops_vec])
    plt.xlabel(r'displacement $|\alpha|$')
    plt.ylabel('SNR ext')
    #plt.savefig('max SNR attainable (nu=1.5).pdf')
    plt.show()



def density_plot_gaussian(): 
  N=1
  r_vec=np.linspace(0.01,4,100)
  nu_vec=np.linspace(1.1,5,100)
  disp_vec=np.linspace(0,30,200)
  X=disp_vec
  Y=nu_vec
  X_grid, Y_grid =np.meshgrid(X,Y)
  grid= np.vstack([X_grid.ravel(),Y_grid.ravel()]).T 
  sigma0=[V_thermal(nu,[z(0)],[0],[0],params=None) for nu in nu_vec]
  snr_gauss=np.empty(shape=(len(Y),len(X)))
  for j in range(len(nu_vec)):
    for i in range(len(disp_vec)):
        sigma=[V_thermal(nu_vec[j],[z(r)],[0],[0,0],params=None) for r in r_vec]
        snr_array=[SNR_gaussian_extr(sigma[k],sigma0[j],[disp_vec[i],0]) for k in range(len(sigma))]
        #find optimal values of r for each displacement
        best_snr=max(snr_array)
        snr_gauss[j][i]=np.real(best_snr)
  W= snr_gauss
  print(np.shape(W))
  fig,ax=plt.subplots(figsize=(10,6))
  c=ax.pcolormesh(X_grid,Y_grid,W,norm=mcolors.LogNorm(vmin=np.min(W), vmax=np.max(W)),cmap='jet')
  cbar=fig.colorbar(c,ax=ax, label='SNR extr')
  ax.set_xlim(X.min(), X.max())
  print(Y.min() , Y.max())
  ax.set_ylim(Y.min() , Y.max())
  ax.grid(True, which='both', linestyle='--')
  ax.set_xlabel(r'Displacement parameter $\alpha$', fontsize=13)
  ax.set_ylabel(r'Noise $\gamma$ (temperature)', fontsize=13)
  c.set_label('SNR extr')
  cbar.ax.set_yticks(ticks=[0.1,1,2,3,4,5,6],labels=['0.1','1','2','3','4','5','6'])

  contour_levels = [1]
  contour = ax.contour(X_grid, Y_grid, W, levels=contour_levels, colors='black', linestyles='dashed', linewidths=1.5)
  ax.clabel(contour, inline=True, fontsize=10,fmt='Classical limit')
  plt.savefig('density plot noise-displacement 2.pdf')
  plt.show()


def test_snr(sigma,sigma0,displacement,nongaussian_ops):
    return 1/((antibunching_one_mode(sigma,displacement,nongaussian_ops)-1)*ergotropy(sigma,sigma0,displacement)+1)

def equation_fit(nu):
    N=1
    r_vec=np.linspace(0.0,4,100)
    disp_vec=np.linspace(1.2,10,100)
    sigma0=V_thermal(nu,[z(0)],[0],[0],params=None)
    #find the r that yields the minimum g for each displacement
    g=[]
    e=[]
    snr_gauss=[]
    snr_test=[]
    #gaussian
    for d in disp_vec:
        sigma=[V_thermal(nu,[z(r)],[0],[0],params=None) for r in r_vec]
        g_array_gauss=[antibunching_one_mode(sigma[i],[d,0],[]) for i in range(len(sigma))]

        #find optimal values of r for each displacement
        lowest_g=min(g_array_gauss)

        #find the indices of those optimal values
        index_g=g_array_gauss.index(lowest_g)
        #print(z(r_vec[index_g]))
        e+=[ergotropy(sigma[index_g],sigma0,[d,0])]
        snr_gauss+=[SNR_gaussian_extr(sigma[index_g],sigma0,[d,0])]
        snr_test+=[test_snr(sigma[index_g],sigma0,[d,0],[])]
        g+=[lowest_g]

    print(np.real(g))    
    
    plt.plot(disp_vec,g,linestyle='dashed')
    plt.plot(disp_vec,1/(1 + np.exp(-0.79 * (disp_vec +0.74))))
    plt.show()
    plt.plot(disp_vec,e,linestyle='dashed')
    plt.plot(disp_vec,disp_vec**2)
    plt.show()
    plt.plot(disp_vec,snr_gauss,'b',linestyle= 'dashed')
    plt.plot(disp_vec,snr_test,'y',linestyle='dashed')
    plt.plot(disp_vec,1/((np.exp(-0.79 * (disp_vec +0.74))/(1 + np.exp(-0.79 * (disp_vec +0.74))))*disp_vec**2+1),'b') 
    plt.show()

    

# paper_sanity_check(1)
# figure5(1)
snr_vs_energy_comparison(1.5)
#density_plot_gaussian()
#equation_fit(1)
