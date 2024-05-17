from utils import *
from plots import * 
from expectation_values import *
from numpy import tanh
x=np.linspace(0.0001,1,10)

z_vec=np.linspace(0.1,3,1000)
mat=[V_tms([z,1/z],[0],[0,0],params=None) for z in z_vec]
#mat2=[V_thermal([1]*2,[z,1/z],[np.pi/4],[0],[0,0],[0,0],params1=None,params2=None) for z in z_vec]
#plt.plot(z_vec,[expvalN(mat[i]) for i in range(len(z_vec))])
#plt.plot(z_vec,[varianceN(mat[i]) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_gaussian(mat[i]) for i in range(len(z_vec))])
#plt.plot(z_vec,[expvalN(mat2[i]) for i in range(len(z_vec))])
#plt.plot(z_vec,[SNR_gaussian(mat2[i]) for i in range(len(z_vec))])
plt.title('Evolution of SNR with squeezing factor for gaussian case' )
plt.show()
