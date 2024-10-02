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
import winsound



def beep(): #definition of sounds 
  #os.system("afplay /System/Library/Sounds/Ping.aiff")
  winsound.Beep(2000, 1000)
  return

'''Auxiliary functions for perfect matchings and loops'''

def perfect_matchings_and_loops(num_ladder_operators):
    '''
    Finds all existing perfect matchings in a list of an even number of nodes
    referred to the even number of ladder operators indices applied to a Gaussian state.
    Additionally, considers matchings where some nodes are left as self-loops.

    :param num_ladder_operators: EVEN number of ladder operators
    :return: List of lists containing all possible perfect matchings of the operators
    '''
    perf_matchings = []
    find_perf_match_and_loops([i for i in range(num_ladder_operators)], [], perf_matchings)
    return perf_matchings

def find_perf_match_and_loops(index_list, current_combination, perf_matchings):
    '''
    AUXILIARY RECURSIVE FUNCTION OF perfect_matchings(num_ladder_operators) that creates
    all existing perfect matchings given an index list and stores them in
    perf_matchings parameter

    :param index_list: Number of existing indices (or nodes in a complete graph)
    :param current_combination: The perfect matching combination being filled at the moment
    :param perf_matchings: List of lists that will store all perfect matchings at the end of the recursive calls
    '''
    if len(index_list) > 0:
        v1 = index_list.pop(0)
        
        # Option 1: v1 is paired with another node
        for i in range(len(index_list)):
            new_combination = current_combination.copy()
            new_idx_list = index_list.copy()
            v2 = new_idx_list.pop(i)
            new_combination.append([v1, v2])
            find_perf_match_and_loops(new_idx_list, new_combination, perf_matchings)
        
        # Option 2: v1 is left as a self-loop
        new_combination = current_combination.copy()
        new_combination.append([v1])
        find_perf_match_and_loops(index_list.copy(), new_combination, perf_matchings)

        # Also consider the case without pairing v1 at this level
        index_list.insert(0, v1)
    else:
        perf_matchings.append(current_combination)


#function that builds an orthogonal matrix out of a random one
def Orth(params):  #params is a generic vector of passive optics with N^2 parameters
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

#function that alters the convention order xxpp or xpxp
def convention_switch(N,sigma,ordering,format):
      # expresses the covariance matrix in the opposite convention
      #ordering is the initial ordering of input matrix
      array=sigma
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






class State:    #notation as in master thesis. Assume kb= 1, hbar=1 
    
  def __init__(self,N,squeezing,bs,pshift,disp=None,temp=None,nongaussian_ops=None, required_ordering='xxpp', format='number'):
    self.N=N
    if temp is not None:  #T is a vector of length N, with the temperature of each mode
      self.temp = temp #temp should indicate T in the thermodynamics formulas, i.e the degrees in kelvin
    self.temp= [0]*self.N 
    if disp is not None:
        self.disp= disp  #disp should be a vector of length 2N: first the N mean positions, then the N mean momenta
    self.disp= [0]*(2*self.N)
    self.squeezing = squeezing  #vector of length N with the squeezing parameter z (not r!!) for each mode 
    self.bs= bs         #vector of length N(N-1)//2 with the beamsplitter angles between each of the modes  
    self.pshift = pshift   #vector of length N with the dephasing angle (rads) of each mode 
    self.nongaussian_ops = nongaussian_ops
    self.required_ordering = required_ordering
    self.format = format


  @property
  def nu(self):    #we assume kB =1, unit frequency omega =1 for all modes. We construct the nu vector of length 2N to multiply by the resulting covariance matrix
    if self.temp == [0]*self.N:
       return [1]*(2*self.N)
    return [1+ 2/(np.exp(1/(self.temp[i]))-1)  for i in range(len(self.temp))]+ [1+ 2/(np.exp(1/(self.temp[i]))-1)  for i in range(len(self.temp))] 
    
  @property
  def r_factor(self):
    return [-np.log(z)/2 for z in self.squeezing]  #vector of r or lambda squeezing factor

      
  @property
  def matrix(self):  #covariance matrix of the gaussian state before nongaussian operations
    def S(self): 
      d_vector=[item for item in self.squeezing]
      for item in self.squeezing:
        d_vector+=[1/item]
      return d_vector*np.eye(2*self.N)
  
    def P(self):
      P = np.zeros((2*self.N, 2*self.N))
      for i in range(self.N):
        P[i,i]=P[i+self.N,i+self.N]= cos(self.pshift[i])
        P[i,i+self.N]=sin(self.pshift[i])
        P[i+self.N,i]=-sin(self.pshift[i])
      return P
    
    def B(self):
      B_total=np.eye(2*self.N)
      index=0
      for i in range(self.N):
          for j in range(i+1,self.N):
              B=np.eye(2*self.N)
              B[i,i]=B[j,j]=B[self.N+i,self.N+i]=B[self.N+j,self.N+j]= cos(self.bs[index])
              B[i,j]=B[self.N+i,self.N+j]= sin(self.bs[index])
              B[j,i]=B[self.N+j,self.N+i]= -sin(self.bs[index])
              index+=1
              B_total=B_total@B
      return B_total
    
    matrix= B(self) @ P(self) @ (self.nu*S(self))  @ transpose(P(self)) @ transpose(B(self))

    if self.required_ordering == 'xpxp':
      return convention_switch(self.N,matrix,'xxpp',format='number')
    return matrix
  
  @property
  def state_operators(self):
    ops = ['rho_g']
    if self.nongaussian_ops is not None:
      for item in self.nongaussian_ops:
        if item<0: #subtraction
          ops=['a']+ ops +['adag']
        elif item>0: #addition
          ops=['adag']+ ops +['a']
    return ops

  @property
  def state_operator_modes(self):
    modes = ['rho_g']
    if self.nongaussian_ops is not None:
      for item in self.nongaussian_ops:
        modes = [np.abs(item)]+ modes +[np.abs(item)] 
    return modes
     
    
  #METHODS ON CLASS INSTANCES  (i.e on the state's covariance matrix)

  def gauss_check(self): #checks that the covariance matrix corresponds to a gaussian state using the Robertson-Schrodinger uncertainty relation
    sigma=self.matrix
    if self.required_ordering=='xxpp':
      sigma=convention_switch(self.N,sigma,'xxpp','number')
    #define Omega matrix
    omega1=[[0,1],[-1,0]]
    Omega = np.zeros((2*self.N, 2*self.N))
    # Replace diagonal blocks with omega1
    for i in range(self.N):
        Omega[2*i:2*(i+1), 2*i:2*(i+1)] = omega1
    #check positivity of matrix + i*Omega:
    eigenvalues, eigenvectors= np.linalg.eig(sigma+1j*Omega)
    print('eigenvalues',np.round(eigenvalues,2))
    check=0
    for item in eigenvalues:
        if np.round(np.real(item),3) < 0:
          check= check
        else:
            check+=1
    if check==2*self.N:
      print('state is gaussian')
    if check!=2*self.N:
          print('Not gaussian state')
    return

  def create_test_matrix(self):  #creation of a test matrix to test the convention switch
    def generate_labels(self):
      labels=[]
      if self.required_ordering=='xxpp':
        for i in range(self.N):
            labels.append(f"x{i+1}")
        for i in range(self.N):
            labels.append(f"p{i+1}")
        return labels
      if self.ordering=='xpxp':
        for i in range(self.N):
            labels.append(f"x{i+1}")
            labels.append(f"p{i+1}")
        return labels

    matrix = np.empty((2*self.N,2*self.N),dtype=object)
    labels= generate_labels(self)
    for i in range(len(labels)):
        for j in range(len(labels)):
                matrix[i][j] = labels[i] + labels[j]  # Product of x's
    return matrix


  def expectationvalue(self, operatorlist, modeslist):  
    sigma=self.matrix
    if self.required_ordering == 'xpxp':
      sigma=convention_switch(self.N,sigma,'xpxp','number')

      #definition of the paper's identities 
    def id1(l,k): #function to compute Tr(a^dag_l a^dag_k rho)
      return (1/4)*(sigma[l-1][k-1]-sigma[l+self.N-1][k+self.N-1]-1j*(sigma[l-1][k+self.N-1]+sigma[l+self.N-1][k-1]))

    def id2(l,k): #function to compute Tr(a_l a_k rho)
      return np.conjugate(id1(l,k))

    def id3(l,k): #function to compute Tr(a^dag_l a_k rho)
        delta=0
        if l==k:
            delta+=1
        return (1/4)*(sigma[l-1][k-1]+sigma[l+self.N-1][k+self.N-1]+1j*(sigma[l-1][k+self.N-1]-sigma[l+self.N-1][k-1])-2*delta)

    def id4(l,k):  #function to compute Tr(a^_l a^dag_k rho)
        delta2=0
        if l==k:
            delta2+=1
        return np.conjugate(id3(l,k))+delta2
    
    full_operatorlist= [op for op in operatorlist] + [ng_op for ng_op in self.state_operators]
    full_modeslist = [mod for mod in modeslist] + [ng_mod for ng_mod in self.state_operator_modes]
    cut = full_operatorlist.index('rho_g')
    full_operatorlist= full_operatorlist[cut+1:]+full_operatorlist[:cut]
    full_modeslist= full_modeslist[cut+1:]+full_modeslist[:cut]

    indices=[i for i in range(len(full_operatorlist))]
    trace=0
    for matching in perfect_matchings_and_loops(len(indices)):
        factor=1
        for pair in matching:
            if len(pair)==1:
              l= full_modeslist[pair[0]]
              if full_operatorlist[pair[0]]=='a':
                factor*= self.disp[l-1]+1j*self.disp[l-1+self.N]   #alpha_j es (<x>+i<p>)_j
              if full_operatorlist[pair[0]]=='adag':
                factor*= np.conjugate(self.disp[l-1]+1j*self.disp[l-1+self.N])
            else:   
              l,k= full_modeslist[pair[0]],full_modeslist[pair[1]]
              if full_operatorlist[pair[0]]=='adag' and full_operatorlist[pair[1]]=='adag':
                  factor*=id1(l,k)
              elif full_operatorlist[pair[0]]=='a' and full_operatorlist[pair[1]]=='a':
                  factor*=id2(l,k)
              elif full_operatorlist[pair[0]]=='adag' and full_operatorlist[pair[1]]=='a':
                  factor*=id3(l,k)
              elif full_operatorlist[pair[0]]=='a' and full_operatorlist[pair[1]]=='adag':
                  factor*=id4(l,k)
        trace+=factor
    return trace
  
  def K(self):  #normalization of state vector (should be 1 for gaussian)
    return self.expectationvalue([],[])


  def expvalN(self): 
    sum=0
    for i in range(1,state.N+1):
      ops=['adag','a']
      modes=[i,i]
      sum+=self.expectationvalue(ops,modes)
    return sum/self.K()
  
  def expvalN2(self): 
    sum=0
    for i in range(1,self.N+1):
      for j in range(1,self.N+1):
        ops= ['adag','a','adag','a']
        modes=[i,i,j,j]
        sum+=self.expectationvalue(ops,modes)
    return sum/self.K()
  
  def passive(self):  #returns the passive vacuum state associated to our state
    return State(self.N,[1]*self.N, [0]*((self.N*self.N-1)//2), [0]*self.N, temp=self.temp)
  
  def ergotropy(self):
    return self.expvalN()-self.passive().expvalN()
  
  def varianceN(self):
    return  np.sqrt(self.expvalN2() - (self.expvalN())**2) 

  def std_dev(self):
    return self.varianceN()-self.passive().varianceN()

  def SNR(self):
    return self.expvalN()/(self.varianceN()**2)

  def SNR_extr(self):
    return self.ergotropy()/(self.varianceN()**2)
  
  def SV(self):  #works for N=2 only
    if self.N != 2: 
      print('This function only computes bipartite entanglement! N has to be 2')
      return
    sum1= self.expectationvalue(['adag','a'],[1,1])/self.K()
    sum2= self.expectationvalue(['adag','a'],[2,2])/self.K()
    sum3= self.expectationvalue(['adag','adag'],[1,2])/self.K()
    sum4= self.expectationvalue(['a','a'],[1,2])/self.K()
    return ((sum1*sum2)-(sum3*sum4))
  
  def antibunching(self): #N=1 or N=2
    if self.N==1:
      sum1= self.expectationvalue(['adag','adag','a','a'],[1,1,1,1])
      sum2= self.expectationvalue(['adag','a'],[1,1])
      return sum1/(sum2**2)
    elif self.N==2:
      sum1= self.expectationvalue(['adag','adag','a','a'],[1,1,1,1])
      sum2= self.expectationvalue(['adag','adag','a','a'],[2,2,2,2])
      sum3= self.expectationvalue(['adag','adag','a','a'],[1,2,1,2])
      return ((sum1+sum2)+2*sum3)/(self.expvalN())**2
    
    else:
      print('This function only works for N=1 or N=2! (yet)')
      return
  

  
  


state = State(2,[0.5,2],[np.pi/4],[0,0],nongaussian_ops=[-1,-1,-1])
#print(state.__dict__)
n = state.expvalN()
print(state.ergotropy())
print(state.SNR())
print(state.SNR())


# n2= Operator (...)
# delta_n= Operator(...)
# snr = state.expectationvalue(snr.operatorlist,snr.modeslist) / state.expectationvalue(snr.other_ops_list,snr.other_modes_list)

