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
#import winsound


def beep(): #definition of sounds 
  #os.system("afplay /System/Library/Sounds/Ping.aiff")
  winsound.Beep(2000, 1000)

'''Auxiliary functions for perfect matchings'''

def perfect_matchings(num_ladder_operators):
    '''
    Finds all existing perfect matchings in a list of an even number of nodes
    referred to the even number of ladder operators indices applied to a Gaussian state.

    :param num_ladder_operators: EVEN number of ladder operators
    :return: List of lists containing all possible perfect matchings of the operators
    '''
    perf_matchings = []
    find_perf_match([i for i in range(num_ladder_operators)], [], perf_matchings)
    return perf_matchings

def find_perf_match(index_list, current_combination, perf_matchings):
    '''
    AUXILIARY RECUSRIVE FUNCTION OF perfect_matchings(num_ladder_operators) that creates
    all existing perfect matchings given an index list and stores them in
    perf_matchings parameter

    :param index_list: Number of existing indices (or nodes in a complete graph)
    :param current_combination: The perfect matching combination being filled at the moment
    :param perf_matchings: List of lists that will store all perfect matchings at the end of the recursive calls
    '''
    if len(index_list) > 0:
        v1 = index_list.pop(0)
        current_combination.append(v1)
        for i in range(len(index_list)):
            new_combination = current_combination.copy()
            new_idx_list = index_list.copy()
            v2 = new_idx_list.pop(i)
            new_combination.append(v2)
            find_perf_match(new_idx_list, new_combination, perf_matchings)
    else:
        perf_matchings.append(current_combination)


def p_matchings(elements):  #input is an array/list of numbers #output is a reshaped array that we can iterate through after
    x=len(elements)
    y=len(perfect_matchings(x))
    return np.reshape(perfect_matchings(x),(1,y,int(x/2),2)).tolist()[0]


#BUILDING OF THE COVARIANCE MATRIX OF A GAUSSIAN SQUEEZED STATE
#function that builds squeezing matrix D
def sq(z,ordering='xxpp'): #ordering is optional parameter
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

#diagonal matrix for thermal states
def Temp(temp,ordering='xxpp'): #ordering is optional parameter
  N=len(temp)
  if ordering == 'xpxp':
    temp_vector=[]
    for item in temp:
        temp_vector+=[item, item]
    t=temp_vector*np.eye(2*N)

    #convention x1,x2,p1,p2
  elif ordering == 'xxpp':
    temp_vector=[item for item in temp]+[item for item in temp]
    t=temp_vector*np.eye(2*N)
  return t

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

def V_tms(z,x, phi, params,ordering='xxpp'):  #ordering is an optional parameter
    N=len(z)
    #beamsplitter
    
    if type(x)==np.float64:
      x=[x]
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
      result= B_total @P @ S  @ transpose(P) @ transpose(B_total) 
    
    if ordering == 'xpxp': 
      return convention_switch(result,'xxpp',format='number')
    else:
      return result

#CHANGE OF CONVENTION IN THE ORDER OF THE COVARIANCE MATRIX ELEMENTS
def convention_switch(sigma,ordering,format):
  # expresses the covariance matrix in the opposite convention
  #ordering is the initial ordering of input matrix
  array=sigma
  N=len(sigma)//2
  #print('input array:')
  #pprint(sigma)
  if format=='string':
    newarray= np.empty((2*N,2*N),dtype=object)
  else:
    newarray= np.zeros((2*N,2*N))
  if ordering=='xxpp':
    for k in range(N):
      newarray[:, 2*k] = array[:, k]
      newarray[:,2*k+1]= array[:, N+k]
    if format=='string':
      newarray2= np.empty((2*N,2*N),dtype=object)
    else:
      newarray2= np.zeros((2*N,2*N))
    for k in range(N):
      newarray2[2*k,:] = newarray[k,:]
      newarray2[2*k+1,:]= newarray[N+k,:]
    #print('swapped array:',newarray2)
    return newarray2
  elif ordering=='xpxp':
    for k in range(N):
        newarray[:, k] = array[:, 2*k]
        newarray[:, N+k] = array[:, 2*k+1]
    if format=='string':
      newarray2= np.empty((2*N,2*N),dtype=object)
    else:
      newarray2= np.zeros((2*N,2*N))
    for k in range(N):
        newarray2[k, :] = newarray[2*k, :]
        newarray2[N+k, :] = newarray[2*k+1, :]
    #print('swapped array:',newarray2)
    return newarray2
  
#now we're going to construct the covariance matrix of a thermal state
#Everything is exactly the same except that instead of stating from vacuum we start from some diagonal noise matrix
def V_thermal(temp,z,x,phi,params=None,ordering='xxpp'):  #ordering and params are optional parameters
  #first we change the inverse temperature to the mean photon number \bar{n} 
  for i in range(len(temp)):
     temp[i]= 1+ 2/(np.exp(1/(temp[i]))-1)
  N=len(z)
  #beamsplitter
  if type(x)==np.float64:
    x=[x]
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
  #dephasing
  P= np.zeros((2*N, 2*N))
  for i in range(N):
    P[i,i]=P[i+N,i+N]= cos(phi[i])
    P[i,i+N]=sin(phi[i])
    P[i+N,i]=-sin(phi[i])
  #print('P',P)
  S=sq(z) #we apply the square root since we are going to apply the squeezing matrix twice 

  #print('S',S)
  if params is not None:
    O1= Orth(params)
    result= temp[0]*(P @ B_total @ O1 @ S @ transpose(O1) @ transpose(B_total) @ transpose(P))
  else:
     result= temp[0]*(B_total @P @ S  @ transpose(P) @ transpose(B_total) )
    
  if ordering == 'xpxp': 
    return convention_switch(result,'xxpp',format='number')
  else:
    return result

  
def create_test_matrix(N,ordering):   #creation of a test matrix to test the convention switch
  def generate_labels(N,ordering):
    labels=[]
    if ordering=='xxpp':
      for i in range(N):
          labels.append(f"x{i+1}")
      for i in range(N):
          labels.append(f"p{i+1}")
      return labels
    if ordering=='xpxp':
      for i in range(N):
          labels.append(f"x{i+1}")
          labels.append(f"p{i+1}")
      return labels

  matrix = np.empty((2*N,2*N),dtype=object)
  labels= generate_labels(N,ordering)
  #print('label:',labels)

  for i in range(len(labels)):
      for j in range(len(labels)):
              matrix[i][j] = labels[i] + labels[j]  # Product of x's
  return matrix


# Gaussianity check function using Robertson-Schrodinger uncertainty relation
def gauss_check(sigma, ordering='xxpp'): #checks that the input covariance matrix corresponds to a gaussian state
  N= len(sigma)//2
  print('input sigma')
  pprint(np.round(sigma,2))
  if ordering=='xxpp':
    sigma=convention_switch(sigma,'xxpp','number')
  print('sigma after reordering')
  pprint(np.round(sigma,2))

  #define Omega matrix
  omega1=[[0,1],[-1,0]]
  Omega = np.zeros((2*N, 2*N))
  # Replace diagonal blocks with omega1
  for i in range(N):
      Omega[2*i:2*(i+1), 2*i:2*(i+1)] = omega1
  #check positivity of sigma + i*Omega:
  eigenvalues, eigenvectors= np.linalg.eig(sigma+1j*Omega)
  print('eigenvalues',np.round(eigenvalues,2))
  check=0
  for item in eigenvalues:
      if np.round(np.real(item),3) < 0:
        check= check
      else:
          check+=1
  if check==2*N:
    print('state is gaussian')
  if check!=2*N:
        print('Not gaussian state')
  return


#definition of the paper's identities 
def id1(sigma,l,k,ordering='xxpp'): #function to compute Tr(a^dag_l a^dag_k rho)
    N= len(sigma)//2
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    return (1/4)*(sigma[l-1][k-1]-sigma[l+N-1][k+N-1]-1j*(sigma[l-1][k+N-1]+sigma[l+N-1][k-1]))

def id2(sigma,l,k,ordering='xxpp'): #function to compute Tr(a_l a_k rho)
    N= len(sigma)//2
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')

    return np.conjugate(id1(sigma,l,k))

def id3(sigma,l,k,ordering='xxpp'): #function to compute Tr(a^dag_l a_k rho)
    N= len(sigma)//2
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    delta=0
    if l==k:
        delta+=1
    return (1/4)*(sigma[l-1][k-1]+sigma[l+N-1][k+N-1]+1j*(sigma[l-1][k+N-1]-sigma[l+N-1][k-1])-2*delta)

def id4(sigma,l,k,ordering='xxpp'):  #function to compute Tr(a^_l a^dag_k rho)
    N= len(sigma)//2
    if ordering == 'xpxp':
      sigma=convention_switch(sigma,'xpxp','number')
    delta2=0
    if l==k:
        delta2+=1
    return np.conjugate(id3(sigma,l,k))+delta2

#function to compute traces (defined in the paper)
def trace_func(sigma,l,k,case):
    if case==1:
        return id1(sigma,l,k)
    elif case==2:
        return id2(sigma,l,k)
    elif case==3:
        return id3(sigma,l,k)
    elif case==4:
        return id4(sigma,l,k)

def expectationvalue(covmat,operatorlist,modeslist):
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
            factor*=trace_func(covmat,l,k,case)
        trace+=factor
    return trace

