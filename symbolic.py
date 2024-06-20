import sympy as sym
from sympy import symbols, Matrix, simplify, exp, sqrt, tanh
import numpy as np
from numpy import transpose, real,linalg, cosh, sinh
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


def V_tms_sym(z,x, phi, params, ordering='xxpp'): 
    N=len(z)
    #beamsplitter
    B_total=Matrix.eye(2*N)
    #print('x:',cos(x),sin(x))
    index=0
    for i in range(N):
        for j in range(i+1,N):
            B=Matrix.eye(2*N)
            B[i,i]=B[j,j]=B[N+i,N+i]=B[N+j,N+j]= sym.cos(x[index])
            B[i,j]=B[N+i,N+j]= sym.sin(x[index])
            B[j,i]=B[N+j,N+i]= -sym.sin(x[index])
            index+=1
            B_total=B_total@B
    #print('B',np.round(B_total,2))
    #dephasing
    P= Matrix.eye(2*N)
    for i in range(N):
      P[i,i]=P[i+N,i+N]= sym.cos(phi[i])
      P[i,i+N]=sym.sin(phi[i])
      P[i+N,i]=-sym.sin(phi[i])
    #print('P',P)
    S=sq(z)
    #print('S',S)
    if params is not None:
      O1= Orth(params)
      result= P @ B_total @ O1 @ S @ transpose(O1) @ transpose(B_total)  @ transpose(P)
    else:
      result= P @ B_total @ S @ transpose(B_total) @transpose(P)

    if ordering == 'xpxp': 
      return convention_switch(simplify(result),'xxpp',format='String')
    else:
      return simplify(result)
    
def V_thermal_sym(nu,z,x,phi,params=None,ordering='xxpp'):  #ordering and params are optional parameters
  #first we change the inverse temperature to the mean photon number \bar{n} 
  #then we calculate 2\bar{n} +1, which is the prefactor nu by which we have to multiply the covariance matrix of non thermal
  #states to get thermal ones
  #factor = 1+ 2/(exp(1/(temp))-1)
  #return factor * V_tms_sym(z,x,phi,params,ordering)
  #just for now we are going to directly multiply my nu to get simpler symbolic expressions
  return nu * V_tms_sym(z,x,phi,params,ordering)

  
  

def id1_sym(sigma,l,k,ordering='xxpp'): #function to compute Tr(a^dag_l a^dag_k rho)
    N= int(np.sqrt(len(sigma))//2)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    return (1/4)*simplify(sigma[l-1,k-1]-sigma[l+N-1,k+N-1]-1j*(sigma[l-1,k+N-1]+sigma[l+N-1,k-1]))

def id2_sym(sigma,l,k,ordering='xxpp'): #function to compute Tr(a_l a_k rho)
    N= int(np.sqrt(len(sigma))//2)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    return np.conjugate(id1_sym(sigma,l,k))

def id3_sym(sigma,l,k,ordering='xxpp'): #function to compute Tr(a^dag_l a_k rho)
    N= int(np.sqrt(len(sigma))//2)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    delta=0
    if l==k:
        delta+=1
    return (1/4)*simplify(sigma[l-1,k-1]+sigma[l+N-1,k+N-1]+1j*(sigma[l-1,k+N-1]-sigma[l+N-1,k-1])-2*delta)


def id4_sym(sigma,l,k,ordering='xxpp'):  #function to compute Tr(a^_l a^dag_k rho)
    N= int(np.sqrt(len(sigma))//2)
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    delta2=0
    if l==k:
        delta2+=1
    return np.conjugate(id3_sym(sigma,l,k))+delta2

#function to compute traces (defined in the paper)
def trace_func_sym(sigma,l,k,case):
    if case==1:
        #print('id1',l,k)
        return id1_sym(sigma,l,k)
    elif case==2:
        #print('id2',l,k)
        return id2_sym(sigma,l,k)
    elif case==3:
        #print('id3',l,k)
        return id3_sym(sigma,l,k)
    elif case==4:
        #print('id4',l,k)
        return id4_sym(sigma,l,k)

def expectationvalue_sym(covmat,operatorlist,modeslist):
    indices=[i for i in range(len(operatorlist))]
    trace=0
    #print('Perfect matchings',perfect_matchings(indices))
    for matching in p_matchings(indices):
        #print('matching:',matching)
        factor=1
        for pair in matching:
            #print(pair)
            l,k= modeslist[pair[0]],modeslist[pair[1]]
            #print('l,k:',l,k)
            if operatorlist[pair[0]]=='adag' and operatorlist[pair[1]]=='adag':
                case=1
            elif operatorlist[pair[0]]=='a' and operatorlist[pair[1]]=='a':
                case=2
            elif operatorlist[pair[0]]=='adag' and operatorlist[pair[1]]=='a':
                case=3
            elif operatorlist[pair[0]]=='a' and operatorlist[pair[1]]=='adag':
                case=4
            #print('case',case)
            factor*=trace_func_sym(covmat,l,k,case)
        trace+=factor
    return trace


#GAUSSIAN STATE (symbolic/analytical expression)


#Expectation value of N
def expvalN_sym(sigma): #input a 2N x 2N np.array of parameters for M
    N = int(np.sqrt(len(sigma))//2)
    #print(N)
    #print('sigma',sigma)
    #now let's calculate the tr(prod(a's)rho). The amount of ladder operators is twice the number of modes (2N)
    #the amount of destruction operators is N, and the amount of creation is also N
    sum=0
    for i in range(1,N+1):
      ops=['adag','a']
      modes=[i,i]
      sum+=expectationvalue_sym(sigma,ops,modes)
    return simplify(sum) 

#Expectation value of N^2

def N2_sym(sigma): #dispersion of number operator on gaussian state (rho0)
    N = int(np.sqrt(len(sigma))//2)
    #We now compute exp(N^2):
    sum=0
    for i in range(1,N+1):
      ops= ['adag','a','adag','a']
      modes=[i,i,i,i]
      sum+=expectationvalue_sym(sigma,ops,modes)
    for i in range(1,N+1):
      for j in range(i+1,N+1):
        ops= ['adag','a','adag','a']
        modes=[i,i,j,j]
        sum+=2*expectationvalue_sym(sigma,ops,modes)
    return simplify(sum)


def varianceN_sym(sigma):
    return  simplify(sym.sqrt(N2_sym(sigma) - (expvalN_sym(sigma))**2))

def SNR_gaussian_sym(sigma):
  return simplify((expvalN_sym(sigma))/varianceN_sym(sigma))

def SNR_gaussian_extr_sym(sigma,sigma0):
  a=sym.symbols('a')
  return simplify((expvalN_sym(sigma)-expvalN_sym(sigma0))/varianceN_sym(sigma))




#Non gaussian state SYMBOLIC expectation values
def K_ng_sym(sigma, nongaussian_ops):
    ops=['rho']
    modes=['rho']
    if nongaussian_ops==[]:
      return 1
    for item in nongaussian_ops:
      if item<0: #subtraction
        ops=['a']+ops+['adag']
      if item>0: #addition
        ops=['adag']+ops+['a']
      modes=[np.abs(item)]+ modes+[np.abs(item)]
    cut = ops.index('rho')
    ops= ops[cut+1:]+ops[:cut]
    modes= modes[cut+1:]+modes[:cut]
    #print(ops)
    #print(modes)
    return expectationvalue_sym(sigma,ops,modes)

#expectation value of N for the non-gaussian state
def expvalN_ng_sym(sigma,nongaussian_ops):
    N = int(np.sqrt(len(sigma))//2)
    #construct the trace we want to calculate ((adag, a) per mode + non-gaussian cov matrix)
    sum=0
    for i in range(1,N+1):
      ops=['rho']
      modes=['rho']
      for item in nongaussian_ops:
        if item<0: #subtraction
          ops=['a']+ops+['adag']
        if item>0: #addition
          ops=['adag']+ops+['a']
        modes=[np.abs(item)]+ modes+[np.abs(item)]
      ops=['adag','a']+ops
      modes=[i,i]+modes
      cut = ops.index('rho')
      ops= ops[cut+1:]+ops[:cut]
      modes= modes[cut+1:]+modes[:cut]
      #print(ops)
      #print(modes)
      sum+=expectationvalue_sym(sigma,ops,modes)
    return (1/K_ng_sym(sigma,nongaussian_ops))*sum


#expectation value of N^2 for the non-gaussian state
def N2_ng_sym(sigma,nongaussian_ops):
    N = int(np.sqrt(len(sigma))//2)
    sum=0
    for i in range(1,N+1):
      for j in range(1,N+1):
        ops=['rho']
        modes=['rho']
        for item in nongaussian_ops:
          if item<0: #subtraction
            ops=['a']+ops+['adag']
          if item>0: #addition
            ops=['adag']+ops+['a']
          modes=[np.abs(item)]+ modes+[np.abs(item)]
        ops=['adag','a','adag','a']+ops
        modes=[i,i,j,j]+modes
        cut = ops.index('rho')
        ops= ops[cut+1:]+ops[:cut]
        modes= modes[cut+1:]+modes[:cut]
        #print(ops)
        #print(modes)
        sum+=expectationvalue_sym(sigma,ops,modes)
    return (1/K_ng_sym(sigma,nongaussian_ops))*sum

def varianceN_ng_sym(sigma,nongaussian_ops):
    return  sym.sqrt(N2_ng_sym(sigma,nongaussian_ops) - (expvalN_ng_sym(sigma,nongaussian_ops))**2)

def SNR_ng_sym(sigma,nongaussian_ops):
  return (expvalN_ng_sym(sigma,nongaussian_ops)+1)/varianceN_ng_sym(sigma,nongaussian_ops)

def SNR_ng_extr_sym(sigma,nongaussian_ops,sigma0):
  return (expvalN_ng_sym(sigma,nongaussian_ops)-expvalN_sym(sigma0))/varianceN_ng_sym(sigma,nongaussian_ops)

#Print results of analytical calculations:

#DON'T FORGET TO indicate symbolic variables are real (real=True)



def analytical_results_gaussian(z1,z2,x1,phi1,phi2): #so far only for N =2
  z_values = [z1,z2]  #1:N+1
  theta_values = [x1] #N*(N-1)//2 +1
  phi_values = [phi1,phi2]  #1:N+1
  covmat= V_tms_sym(z_values,theta_values,phi_values, params=None)
  print(covmat)
  print(len(covmat))
  print(np.shape(covmat))
  N_gauss=simplify(expvalN_sym(covmat))
  print(N_gauss)
  N2_gauss=simplify(N2_sym(covmat))
  delta_gauss=simplify(varianceN_sym(covmat))
  ratio=simplify(SNR_gaussian_sym(covmat))
  z1_left= np.arange(0.0001,1.0,0.0001)
  z1_right=np.arange(1.0001,2.0,0.0001)
  z1_right2=np.arange(2.0001,3.0,0.0001)
  z1_right3=np.arange(3.0001,4.0,0.0001)
  z1_values=np.vstack((z1_left,z1_right,z1_right2,z1_right3)).flatten()
  ratio_values=[ratio.evalf(subs={z1: z}) for z in z1_values]
  fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8, 10))
  ax1.scatter(z1_values, ratio_values, s=7)
  ax1.set_title('Generic BS and PS values')
  ax1.set_xlabel('z (squeezing)')
  ax1.set_ylabel('SNR')
  covmat2= V_tms_sym(z_values,[0],[0,0], params=None)
  ratio2=SNR_gaussian_sym(covmat2)
  ratio_values2=[ratio2.evalf(subs={z1: z}) for z in z1_values]
  ax2.scatter(z1_values, ratio_values2, s=4)
  ax2.set_title('BS=PS=0')
  ax2.set_xlabel('z (squeezing)')
  ax2.set_ylabel('SNR')
  diff_z1 = ratio.diff(z1)
  diff_z1_values=[diff_z1.evalf(subs={z1: z}) for z in z1_values]
  diff_z1_2 = ratio2.diff(z1)
  diff_z1_values2=[diff_z1_2.evalf(subs={z1: z}) for z in z1_values]
  ax3.plot(z1_values,diff_z1_values)
  ax4.plot(z1_values, diff_z1_values2)
  ax3.plot(z1_values,[0]*len(z1_values), '-')
  ax4.plot(z1_values, [0]*len(z1_values),'-')
  ax3.set_title('Derivative of SNR(z)')
  ax4.set_title('Derivative of SNR(z)')
  plt.show()
  print('covariance matrix:',covmat)
  print('')
  print('N=',N_gauss)
  print('')
  print('N2=',N2_gauss)
  print('')
  print('deltaN=',delta_gauss)
  print('')
  print('SNR=',ratio)
  print('')
  print('diff z1=',diff_z1)
  print('')
  #diff_z2 = ratio.diff(z2)
  #print('diff z2=',diff_z2)
  print('')
  diff_x1 = ratio.diff(x1)
  print('diff x1=',diff_x1)
  print('')
  diff_phi1 = ratio.diff(phi1)
  print('diff phi1=',diff_phi1)
  print('')
  diff_phi2 = ratio.diff(phi2)
  print('diff phi2=',diff_phi2)
  return 

   # Solve the expression for one variable in terms of the other 
  #solutions = sym.solve(diff_z1, z1)

# Plot one variable as a function of the other
# Let's say we plot x1 as a function of z1
  #x1_values = np.arange(0, 2*np.pi)  # Adjust the range as needed
  #for s in solutions:
    #z1_values = [s.evalf(subs={x1: x}) for x in x1_values]
    #plt.plot(x1_values, z1_values)
  #plt.xlabel('x1')
  #plt.ylabel('z1')
  #plt.title('Plot of z1 as a function of x1')
  #plt.grid(True)
  #plt.show()


def analytical_results_nongaussian(z1,z2,x1,phi1,phi2,nongaussian_ops): #so far only for N =2
  N=2
  z_values = [z1,z2]  #1:N+1
  theta_values = [x1] #N*(N-1)//2 +1
  phi_values = [phi1,phi2]  #1:N+1
  covmat= V_tms_sym(z_values,theta_values,phi_values, params=None)
  N_ng=simplify(expvalN_ng_sym(covmat,nongaussian_ops))
  print('N=',N_ng)
  print('K',K_ng_sym(covmat, nongaussian_ops))
  #N2_ng=simplify(N2_ng_sym(covmat,nongaussian_ops))
  #print('')
  #print('N2=',N2_ng)
  delta_ng=simplify(varianceN_ng_sym(covmat,nongaussian_ops))
  #print('')
  print('deltaN=',delta_ng)
  ratio_ng=simplify(SNR_ng_sym(covmat,nongaussian_ops))
  print('')
  print('SNR=',ratio_ng)
  print('')
  diff_z1_ng = ratio_ng.diff(z1)
  print('diff z1=',diff_z1_ng)
  print('')
  #diff_z2_ng = ratio_ng.diff(z2)
  #print('diff z2=',diff_z2_ng)
  print('')
  diff_x1_ng = ratio_ng.diff(x1)
  print('diff x1=',diff_x1_ng)
  print('')
  diff_phi1_ng = ratio_ng.diff(phi1)
  print('diff phi1=',diff_phi1_ng)
  print('')
  diff_phi2_ng = ratio_ng.diff(phi2)
  print('diff phi2=',diff_phi2_ng)

  # Solve the expression for one variable in terms of the other
  #solutions = sym.solve(diff_z1_ng, z1)

# Plot one variable as a function of the other
# Let's say we plot x1 as a function of z1
  #x1_values = np.arange(0, 2*np.pi)  # Adjust the range as needed
  #for s in solutions:
    #z1_values = [s.evalf(subs={x1: x}) for x in x1_values]
    #plt.plot(x1_values, z1_values)
  #plt.xlabel('x1')
  #plt.ylabel('z1')
  #plt.title('Plot of z1 as a function of x1')
  #plt.grid(True)
  #plt.show()
  return


N=2
nu,z,x =symbols('nu,z,x',real=True, RealNumber=True, zero=False)
phi1,phi2 = symbols('phi1,phi2',real=True)
z_values = [z,1/z]  #1:N+1
theta_values = [x] #N*(N-1)//2 +1
phi_values = [phi1,phi2]  #1:N+1
covmat=V_tms_sym(z_values,theta_values,phi_values, params=None)
sigma0=V_tms_sym([1,1],[0],[0,0], params=None)
thermal_sigma0=V_thermal_sym(nu,[1,1],[0],[0,0],params=None)
thermal_covmat=V_thermal_sym(nu,z_values,theta_values,phi_values,params=None)
#print(simplify(thermal_covmat),thermal_sigma0)

# print('gaussian','N', expvalN_sym(covmat), 'variance', varianceN_sym(covmat),'snr',SNR_gaussian_sym(covmat), 'snr ext', SNR_gaussian_extr_sym(covmat,sigma0) )
# print('gaussian thermal','N', expvalN_sym(thermal_covmat), 'variance', varianceN_sym(thermal_covmat),'snr',SNR_gaussian_sym(thermal_covmat),'snr ext', SNR_gaussian_extr_sym(thermal_covmat,thermal_sigma0) )
print('1 subtraction','N',simplify(expvalN_ng_sym(covmat,[-1])), 'delta', simplify(varianceN_ng_sym(covmat,[-1])), 'snr_ext', simplify(SNR_ng_extr_sym(covmat,[-1],sigma0)))
print('1 subtraction thermal','N',simplify(expvalN_ng_sym(thermal_covmat,[-1])), 'delta', simplify(varianceN_ng_sym(thermal_covmat,[-1])),'snr_ext', simplify(SNR_ng_extr_sym(thermal_covmat,[-1],thermal_sigma0)))
print('1 addition','N', simplify(expvalN_ng_sym(covmat,[+1])), 'delta', simplify(varianceN_ng_sym(covmat,[+1])), 'snr_ext', simplify(SNR_ng_extr_sym(covmat,[+1],sigma0)))
print('1 addition thermal',simplify(expvalN_ng_sym(thermal_covmat,[+1])), 'delta', simplify(varianceN_ng_sym(thermal_covmat,[+1])),'snr_ext', simplify(SNR_ng_extr_sym(thermal_covmat,[+1],thermal_sigma0)))
print('2 subtraction','N',simplify(expvalN_ng_sym(covmat,[-1,-1])), 'delta', simplify(varianceN_ng_sym(covmat,[-1,-1])), 'snr_ext', simplify(SNR_ng_extr_sym(covmat,[-1,-1],sigma0)))
print('2 subtraction thermal','N',simplify(expvalN_ng_sym(thermal_covmat,[-1,-1])), 'delta', simplify(varianceN_ng_sym(thermal_covmat,[-1,-1])),'snr_ext', simplify(SNR_ng_extr_sym(thermal_covmat,[-1,-1],thermal_sigma0)))
print('2 addition','N', simplify(expvalN_ng_sym(covmat,[+1,+1])), 'delta', simplify(varianceN_ng_sym(covmat,[+1,+1])), 'snr_ext', simplify(SNR_ng_extr_sym(covmat,[+1,+1],sigma0)))
print('2 addition thermal',simplify(expvalN_ng_sym(thermal_covmat,[+1,+1])), 'delta', simplify(varianceN_ng_sym(thermal_covmat,[+1,+1])),'snr_ext', simplify(SNR_ng_extr_sym(thermal_covmat,[+1,+1],thermal_sigma0)))

#print(analytical_results_gaussian(z1,z2,x1,phi1,phi2))
#print(analytical_results_nongaussian(z1,z2,x1,phi1,phi2,[-1]))

#nongaussian_ops=[+1]
#for i in range(3):
  #print(nongaussian_ops)
  #print(analytical_results_nongaussian(z1,z2,x1,phi1,phi2,nongaussian_ops))
  #beep()
  #nongaussian_ops+=[-1]

beep()


def separability_check():  #separability check using serafini's criterion for N=2
  sigmatest= create_test_matrix(2,'xxpp')
  print('test covariance matrix')
  print(sigmatest)
  print('reordering xpxp to apply serafinis criterion')
  print(convention_switch(sigmatest,'xxpp','string'))
  sigma=V_tms_sym(z_values,theta_values, phi_values, params=None, ordering='xxpp')
  print('initial matrix ordered xxpp')
  print(sigma)
  corr_mat= Matrix(np.array([[sigma[0,1],sigma[0,3]],[sigma[2,1],sigma[2,3]]]))
  print('correlations matrix:', corr_mat)
  print('')
  print('determinant of correlations', simplify(sym.det(corr_mat)))
  return 




#print(analytical_results_gaussian(z1,z2,x1,phi1,phi2))

#print(analytical_results_nongaussian(z1,z2,x1,phi1,phi2,nongaussian_ops))

def surfaces_symbolic():
  N=2
  z1,z2,x1 =symbols('z1,z2,x1',real=True, RealNumber=True)
  phi1,phi2 =symbols('phi1,phi2',real=True)
  z2=1/z1
  z_values = [z1,z2]  #1:N+1
  theta_values = [x1] #N*(N-1)//2 +1
  phi_values = [phi1,phi2]  #1:N+1
  covmat=V_tms_sym(z_values,theta_values,phi_values, params=None)
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  X = np.linspace(0, 2*np.pi, 100) #for angles
  Y = np.linspace(0.05,4, 100)  #for squeezing

  #gaussian case
  Z= 0.5*(Y- 1)**2/(np.sqrt(-0.25*(Y - 1)**2 + (Y**2 + 1)**2/Y)*abs(Y - 1)) 
  # Plot the surface.
  Z, X = np.meshgrid(Z, X)
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
  ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
  ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()
  beep()

  #One photon subtraction
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  Z= 2.0*(0.5*Y**2 - 0.5*Y + 0.5)/(Y*np.sqrt(0.5*Y**2 - 1 + 0.5/Y**2))
  # Plot the surface.
  Z, X = np.meshgrid(Z, X)
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
  ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
  ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()
  beep()
  return 

#surfaces_symbolic()


