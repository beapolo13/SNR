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



#COVARIANCE MATRIX BUILDING
#function that builds squeezing matrix D
def sq(z):
  N=len(z)
  if ordering == 'xpxp':
    d_vector=[]
    for item in z:
        d_vector+=[item, 1/item]
    D=d_vector*np.eye(2*N)

    #convention x1,x2,p1,p2
  elif ordering == 'xxpp':
    d_vector=[item for item in z]
    for item in z:
        d_vector+=[1/item]
    D=d_vector*np.eye(2*N)

  return D

#function that builds an orthogonal matrix out of a random one
def Orth(params):
    N=int(np.sqrt(len(params)))
    M=np.resize(params,(N,N))
    H= M + transpose(np.conjugate(M))
    U=scipy.linalg.expm(1j*H)
    unitary_check=(np.round(U @ transpose(np.conjugate(U)),4)==np.eye(N)).all()
    #print(np.round(U @ transpose(np.conjugate(U)),4))
    #print('U unitary:', unitary_check)

    #first build matrix O
    Ubar= (1/sqrt(2))*np.block([[np.eye(N), 1j*np.eye(N)],[np.eye(N), -1j*np.eye(N)] ])

    V= np.block([[np.conjugate(U),np.zeros((N,N))],[np.zeros((N,N)), U]])
    #print('V=', np.round(V,3))

    #print('V * Ubar=', np.round(V @ Ubar,3))

    O=transpose(np.conjugate(Ubar)) @ (V @ Ubar)
    orth= O @ transpose(O)
    orth_check=(np.round(orth,2)==np.eye(2*N)).all()

    #print('O is orthogonal:', orth_check)
    return O

#function that builds covariance matrix
#z is the squeezing (it has to fulfill certain properties)
#phi is a vector of dephasing angles for each  mode
#x is the beamsplitter angles between each of the modes N(N-1)//2 total combinatios 
#params is the vector of passive optics O
#no params2 since i have checked that the correct definition is with O and O transpose (same orthogonal matrix)

def V_tms(z,x, phi, params): 
    N=len(z)
    if ordering == 'xpxp': #this only works for N=2 so far
      B=np.array([[cos(x),0,sin(x),0],[0,cos(x),0,sin(x)],[-sin(x),0,cos(x),0],[0,-sin(x),0,cos(x)]])
      P1=np.array([[cos(phi1),sin(phi1)], [-sin(phi1), cos(phi1)]])
      P2=np.array([[cos(phi2),sin(phi2)], [-sin(phi2), cos(phi2)]])
      P= np.block([[P1,np.zeros((N,N))],[np.zeros((N,N)), P2]])
    elif ordering == 'xxpp':
      #beamsplitter
      B_total=np.eye(2*N)
      #print('x:',cos(x),sin(x))
      index=0
      for i in range(N):
          for j in range(i+1,N):
              B=np.eye(2*N)
              B[i,i]=B[j,j]=B[N+i,N+i]=B[N+j,N+j]= cos(x[index])
              B[i,j]=B[N+i,N+j]= sin(x[index])
              B[j,i]=B[N+j,N+i]= -sin(x[index])
              index+=1
              B_total=B_total@B
      #print('B',np.round(B_total,2))
      #dephasing
      P= np.zeros((2*N, 2*N))
      for i in range(N):
        P[i,i]=P[i+N,i+N]= cos(phi[i])
        P[i,i+N]=sin(phi[i])
        P[i+N,i]=-sin(phi[i])
      #print('P',P)
    S=sq(z)
    #print('S',S)
    if params is not None:
      O1= Orth(params)
      result= P @ B_total @ O1 @ S @ transpose(O1) @ transpose(B_total) @ transpose(P)
    else:
      result= P @ B_total @ S @ transpose(B_total) @ transpose(P)
    return result



