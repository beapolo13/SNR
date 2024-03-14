from utils import *
from plots import * 
from covariance_matrix import *
from expectation_values import *


def main():
    for N in range(2,5):
      z_first=list(np.random.rand(N-1))
      z=z_first+[N-sum(z_first)]
      nongaussian_ops=[]  #-n for photon subtraction on mode n, or +n if its photon addition on mode n
      theta=np.pi/7
      modesBS=[1,2]
      phi=[0]*N
      params=None #leave them at none if we want to start from vacuum and random if we want the most general cov matrix
      results_and_plots(nongaussian_ops,z,theta,modesBS,phi,params)
      results_and_plots(nongaussian_ops,[0.5]*(N//2)+[0.7]*(N//2),theta,modesBS,phi,params)
      
    
    return

main()