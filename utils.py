import numpy as np
from numpy import transpose, real, sqrt, sin, cos, linalg, cosh, sinh, diag
import sympy as sp
from sympy import symbols, Matrix, simplify, exp, sqrt, tanh, diag, cos, sin, coth
import scipy
import matplotlib.pyplot as plt
import random
import itertools
from itertools import combinations
from scipy import optimize
from scipy.optimize import minimize, fsolve
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
    AUX1jL1jARY RECURS1jVE FUNCT1jON OF perfect_matchings(num_ladder_operators) that creates
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
        newarray= np.zeros((2*N,2*N), dtype=object)
      else:
        newarray= np.zeros((2*N,2*N))
      if ordering=='xxpp':
        for k in range(N):
          for i in range(2*N):
            newarray[i, 2*k] = array[i, k]
            newarray[i,2*k+1]= array[i, N+k]
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
    
  def __init__(self,N,squeezing,bs,pshift,omega=None, disp=None,temp=None,nongaussian_ops=None, required_ordering='xxpp', format='number'):
    self.N=N
    if omega is not None:  #omega is a vector of length N, with the frequencies of each mode
      self.omega = self._convert_to_symbolic(omega) 
    if omega is None:
      self.omega= self._convert_to_symbolic([1]*self.N ) #if we do not have information on omega, we assume unit frequency
    if temp is not None:  #T is a vector of length N, with the temperature of each mode
      self.temp = self._convert_to_symbolic(temp) #temp should indicate T in the thermodynamics formulas, i.e the degrees in kelvin
    if temp is None:
      self.temp= self._convert_to_symbolic([0]*self.N )
    if disp is not None:  
        self.disp= self._convert_to_symbolic(disp)  #disp should be a vector of length 2N: first the N mean positions, then the N mean momenta
    if disp is None:
      self.disp= self._convert_to_symbolic([0]*(2*self.N))
    self.squeezing = self._convert_to_symbolic(squeezing)  #vector of length N with the squeezing parameter z (not r!!) for each mode 
    self.bs= self._convert_to_symbolic(bs)         #vector of length N(N-1)//2 with the beamsplitter angles between each of the modes  
    self.pshift = self._convert_to_symbolic(pshift)   #vector of length N with the dephasing angle (rads) of each mode 
    self.nongaussian_ops = nongaussian_ops
    self.required_ordering = required_ordering
    self.format = format
    self._matrix=None

  
  def _convert_to_symbolic(self, input_value): 
    if isinstance(input_value, str):  # Single string input
      return sp.symbols(input_value)
    elif all(isinstance(item, str) for item in input_value):  # List of strings
      return [sp.symbols(item) for item in input_value]
    else:  # Assume it's a numeric list
      return input_value


  @property
  def nu(self):    #we assume kB =1. We construct the nu vector of length 2N to multiply by the resulting covariance matrix
    if self.temp == [0]*self.N and self.omega == [1]*self.N:
      return [1]*(2*self.N)
    else:
      if self.format == 'string':
        return [sp.coth(self.omega[i]/(2* self.temp[i]))  for i in range(self.N)]+ [sp.coth(self.omega[i]/(2* self.temp[i]))  for i in range(self.N)] 
      else:  
        return [np.coth(self.omega[i]/(2* self.temp[i]))  for i in range(self.N)]+ [np.coth(self.omega[i]/(2* self.temp[i]))  for i in range(self.N)] 
    
  @property
  def r_factor(self):
    if self.format == 'string':
      return [-sp.log(z)/2 for z in self.squeezing]
    else:
      return [-np.log(z)/2 for z in self.squeezing]  #vector of r or lambda squeezing factor

      
  @property
  def matrix(self):  #covariance matrix of the gaussian state before nongaussian operations
    if self._matrix is not None:
      return self._matrix
    
    def S(self): 
      d_vector=[item for item in self.squeezing]
      for item in self.squeezing:
        d_vector+=[1/item]
      if self.format == 'string':
        return sp.diag(*d_vector)
      else:
        return d_vector*np.eye(2*self.N)
  
    def P(self):
      if self.format == 'number':
        P = np.zeros((2*self.N, 2*self.N))
        for i in range(self.N):
          P[i,i]=P[i+self.N,i+self.N]= cos(self.pshift[i])
          P[i,i+self.N]=sin(self.pshift[i])
          P[i+self.N,i]=-sin(self.pshift[i])
        return P
      elif self.format == 'string':
        P = Matrix.eye(2*self.N, 2*self.N)
        for i in range(self.N):
          P[i, i] = P[i + self.N, i + self.N] = sp.cos(self.pshift[i])  
          P[i, i + self.N] = sp.sin(self.pshift[i]) 
          P[i + self.N, i] = -sp.sin(self.pshift[i])
        return P
    
    def B(self):
      if self.format == 'number':
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
      elif self.format == 'string': 
        B_total=Matrix.eye(2*self.N)
        index=0
        for i in range(self.N):
          for j in range(i+1,self.N):
            B=sp.eye(2*self.N)
            B[i,i]=B[j,j]=B[self.N+i,self.N+i]=B[self.N+j,self.N+j]= sp.cos(self.bs[index])
            B[i,j]=B[self.N+i,self.N+j]= sp.sin(self.bs[index])
            B[j,i]=B[self.N+j,self.N+i]= -sp.sin(self.bs[index])
            index+=1
            B_total=B_total@B
            print(B_total)
        return B_total
         
    if self.format =='string':
      matrix= B(self) @ P(self) @ (diag(*self.nu)@S(self))  @ transpose(P(self)) @ transpose(B(self))
    elif self.format =='number':
      matrix= B(self) @ P(self) @ (self.nu*S(self))  @ transpose(P(self)) @ transpose(B(self))

    if self.required_ordering == 'xpxp':
      return convention_switch(self.N,matrix,'xxpp',format=self.format)
    return matrix
  
  
  @matrix.setter
  def matrix(self, new_matrix):
    self._matrix = new_matrix

  def local_operation(self,theta1,r1,psi1,theta2,r2,psi2): #only for two modes
    if self.N != 2 and self.N !=1:
      print('This method cannot be applied to this number of modes')
      return
    elif self.N==1:
      if self.format == 'string':
        rotation_matrix = sp.Matrix([[r1*sp.cos(theta1)*sp.cos(psi1)-sp.sin(theta1)*sp.sin(psi1)/r1,r1*sp.cos(theta1)*sp.sin(psi1)+sp.sin(theta1)*sp.cos(psi1)/r1],
                                     [-r1*sp.sin(theta1)*sp.cos(psi1)-sp.cos(theta1)*sp.sin(psi1)/r1, -r1*sp.sin(theta1)*sp.sin(psi1)+sp.cos(theta1)*sp.cos(psi1)/r1]])
      elif self.format == 'number':
        rotation_matrix = [[r1*cos(theta1)*cos(psi1)-sin(theta1)*sin(psi1)/r1,r1*cos(theta1)*sin(psi1)+sin(theta1)*cos(psi1)/r1],
                          [-r1*sin(theta1)*cos(psi1)-cos(theta1)*sin(psi1)/r1, -r1*sin(theta1)*sin(psi1)+cos(theta1)*cos(psi1)/r1]]
    elif self.N==2:
      if self.format == 'string':
        rotation_matrix = sp.Matrix([[r1*sp.cos(theta1)*sp.cos(psi1)-sp.sin(theta1)*sp.sin(psi1)/r1,0,r1*sp.cos(theta1)*sp.sin(psi1)+sp.sin(theta1)*sp.cos(psi1)/r1,0],
                                     [0, r2*sp.cos(theta2)*sp.cos(psi2)-sp.sin(theta2)*sp.sin(psi2)/r2,0,r2*sp.cos(theta2)*sp.sin(psi2)+sp.sin(theta2)*sp.cos(psi2)/r2],
                                     [-r1*sp.sin(theta1)*sp.cos(psi1)-sp.cos(theta1)*sp.sin(psi1)/r1,0,  -r1*sp.sin(theta1)*sp.sin(psi1)+sp.cos(theta1)*sp.cos(psi1)/r1,0],
                                     [0, -r2*sp.sin(theta2)*sp.cos(psi2)-sp.cos(theta2)*sp.sin(psi2)/r2,0, -r2*sp.sin(theta2)*sp.sin(psi2)+sp.cos(theta2)*sp.cos(psi2)/r2]])
      elif self.format == 'number':
        rotation_matrix = [[r1*cos(theta1)*cos(psi1)-sin(theta1)*sin(psi1)/r1,0,r1*cos(theta1)*sin(psi1)+sin(theta1)*cos(psi1)/r1,0],
                          [0, r2*cos(theta2)*cos(psi2)-sin(theta2)*sin(psi2)/r2,0,r2*cos(theta2)*sin(psi2)+sin(theta2)*cos(psi2)/r2],
                          [-r1*sin(theta1)*cos(psi1)-cos(theta1)*sin(psi1)/r1,0,  -r1*sin(theta1)*sin(psi1)+cos(theta1)*cos(psi1)/r1,0],
                          [0, -r2*sin(theta2)*cos(psi2)-cos(theta2)*sin(psi2)/r2,0, -r2*sin(theta2)*sin(psi2)+cos(theta2)*cos(psi2)/r2]]
    print(rotation_matrix)
    self.matrix= rotation_matrix @ self.matrix @ rotation_matrix.T
    
  
  #def single_mode_squeezer(self,lambda1,lambda2):
  
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
    #only makes sense for numerically-valued matrices
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
      return (1/4)*(sigma[l-1,k-1]-sigma[l+self.N-1,k+self.N-1]-1j*(sigma[l-1,k+self.N-1]+sigma[l+self.N-1,k-1]))

    def id2(l,k): #function to compute Tr(a_l a_k rho)
      return np.conjugate(id1(l,k))

    def id3(l,k): #function to compute Tr(a^dag_l a_k rho)
        delta=0
        if l==k:
            delta+=1
        return (1/4)*(sigma[l-1,k-1]+sigma[l+self.N-1,k+self.N-1]+1j*(sigma[l-1,k+self.N-1]-sigma[l+self.N-1,k-1])-2*delta)

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

  def expvalN_gaussian(self):
    result=0
    for i in range(self.N):
      result+=self.omega[i]*(self.matrix[i,i]+self.matrix[i+self.N,i+self.N]-2)

    return 1/4*result #faltaria el displacement por sumar

  def expvalN(self): 
    sum=0
    for i in range(1,self.N+1):
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
    return State(self.N,[1]*self.N, [0]*((self.N*self.N-1)//2), [0]*self.N, temp=self.temp, format=self.format)
  
  def fock(self, exc):
    return State(self.N,[1]*self.N, [0]*((self.N*self.N-1)//2), [0]*self.N, temp=self.temp, nongaussian_ops=[1]*exc)
  
  def ergotropy(self):
    return self.expvalN()-self.passive().expvalN()
  
  def varianceN(self):
    if self.format == 'string':
      return  sp.sqrt(self.expvalN2() - (self.expvalN())**2)  
    else:
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
    
  def optimize_ratio(self, max_energy):
        # Objective function (we minimize -ratio to maximize ratio)
        def objective(attrs):
            self.disp[0], self.disp[1], self.squeezing[0] = attrs  # Update class attributes
            return -self.SNR_extr()  # Negative for maximization

        # Constraint: energy should not exceed max_energy
        def energy_constraint(attrs):
            self.disp[0], self.disp[1], self.squeezing[0]  = attrs
            return max_energy - self.ergotropy()  # Must be non-negative

        #Define bounds for parameters
        bounds = [(0, None), (0, None),(0, 1)] 
        # Define constraints dictionary
        constraints = ({'type': 'ineq', 'fun': energy_constraint})

        # Initial guess for the attributes
        initial_guess = self.disp[0], self.disp[1], self.squeezing[0] 

        # Perform optimization
        result = minimize(objective, initial_guess, constraints=constraints, bounds=bounds)

        # Update the attributes with the optimized values
        self.disp[0], self.disp[1], self.squeezing[0]  = result.x

        return result
  

#Symbolic representation
nu1, nu2, w1, w2, z1,z2,x,T,phi1,phi2,alpha1,alpha2,beta1,beta2, lambda1, lambda2,theta1,theta2,psi1,psi2,r1,r2 = symbols('nu1, nu2, w1, w2 z1,z2,x,T, phi1,phi2,alpha1,alpha2,beta1,beta2,lambda1, lambda2,theta1,theta2,psi1,psi2,r1,r2',real=True, RealNumber=True, commutative= True)
alpha = symbols('alpha')
state_sym=State(2,[z1,z2],[x],[0,0],disp=[0,0,0,0],omega=[w1,w2],temp=[T, T],nongaussian_ops=[], required_ordering='xxpp',format='string')
print('initial matrix', state_sym.matrix.subs({coth(w1/(2*T)): nu1, coth(w2/(2*T)): nu2 }))
print('initial energy',state_sym.expvalN_gaussian().subs({coth(w1/(2*T)): nu1, coth(w2/(2*T)): nu2 }))
state_sym.local_operation(0,r1,psi1,0,r2,psi2)


energy_expr=state_sym.expvalN_gaussian().subs({coth(w1/(2*T)): nu1, coth(w2/(2*T)): nu2 })
print('energy local passive', energy_expr)
variables = [r1,psi1,r2,psi2]
gradient_vector = [sp.diff(energy_expr, var) for var in variables]
print('derivative',gradient_vector)

# state_sym2=State(2,[z,z2],[0],[0,0],disp=[0,0,0,0],temp=[T,T],nongaussian_ops=[], required_ordering='xxpp',format='string')
# print('passives',state_sym.passive().matrix,state_sym.passive().expvalN_gaussian())
# print('ergotropic gap', state_sym.ergotropy()-state_sym2.ergotropy())
# pprint(simplify(state_sym.matrix))
# print('N',simplify(state_sym.expvalN()))
# print('N0',simplify((state_sym.passive()).expvalN()))
# print('SNR',state_sym.SNR_extr().factor().expand().subs({alpha1**2+alpha2**2 : alpha**2}).factor().simplify())

#Numerical representation
state_num = State(2,[0.5,2],[0],[0,0],temp=[0,0],nongaussian_ops=[], format='number')
print(state_num.matrix)
print(state_num.expvalN())
print(state_num.expvalN_gaussian())
# state_num2 = State(1,[0.001],[],[0],disp=[13,0],temp=[0.1],nongaussian_ops=[], format='number')
# print('snr', state_num2.SNR_extr())
# #state_num1 = State(1,[random.random()],[],[random.random()],disp=random.sample(range(0, 5), 2),temp=[0.4],nongaussian_ops=[1], format='number')
# #state_num2 = State(2,[random.random(),random.random()],[2*np.pi*random.random()],[random.random(),random.random()],disp=random.sample(range(0, 5), 4),temp=[0.5]*2,nongaussian_ops=[-1,-1], format='number')
# print(state_num.__dict__)
# print(state_num.fock(2).expvalN())
# print(state_num.passive().expvalN())

# optimal_vec=[]
# # optimal_vec1=[]
# # #optimal_vec2=[]
# x_axis=[]
# i=0
# while i < 20:
#   result = state_num.optimize_ratio(i)
# #   result1 = state_num1.optimize_ratio(i)
# #   #result2 = state_num2.optimize_ratio(i)
#   if result.success == True:
#     optimal_vec +=[-result.fun]
# #     optimal_vec1 +=[-result1.fun]
# #     #optimal_vec2 +=[-result2.fun]
#     x_axis+=[i]
#     i+=1
# #   #print(result)
# plt.plot(x_axis,optimal_vec)
# # plt.plot(x_axis,optimal_vec1)
# # #plt.plot(x_axis,optimal_vec2)
# # plt.legend(['gauss','1'])
# plt.show()
# # print(result.success)
# # print("Optimized disp:", state_num.disp)
# # print("Optimized squeezing:", state_num.squeezing)
# # print("Optimized bs:", state_num.bs)

# # print("Maximized ratio:", state_num.SNR_extr())
# # print("Energy after optimization:", state_num.expvalN())



