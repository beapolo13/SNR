from utils import *
from plots import * 
from expectation_values import *
from numpy import tanh
x=np.linspace(0.0001,1,10)

z_vec=np.linspace(0.2,3,1000)
mat=[V_thermal([0.1]*2,[z,1/z],[0],[0,0],params=None) for z in z_vec]
sigma0=V_thermal([0.1]*2,[1,1],[0],[0,0],params=None)
print(sigma0)
operations=[-1]
print(K_ng(sigma0,operations))
print(expvalN_ng(sigma0,operations))
#mat2=[V_thermal([1]*2,[z,1/z],[np.pi/4],[0],[0,0],[0,0],params1=None,params2=None) for z in z_vec]
#plt.plot(z_vec,[expvalN(mat[i]) for i in range(len(z_vec))])
#plt.plot(z_vec,[varianceN(mat[i]) for i in range(len(z_vec))])
plt.plot(z_vec,[expvalN_ng(mat[i],operations)+1 for i in range(len(z_vec))])
#plt.plot(z_vec,[expvalN_ng(mat[i],operations)-expvalN_ng(sigma0,operations) for i in range(len(z_vec))])
plt.plot(z_vec,[varianceN_ng(mat[i],operations) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_ng(mat[i],operations) for i in range(len(z_vec))])
plt.plot(z_vec,[(expvalN_ng(mat[i],operations)-expvalN_ng(sigma0,operations))/varianceN_ng(mat[i],operations) for i in range(len(z_vec))])
#plt.plot(z_vec,[expvalN(mat2[i]) for i in range(len(z_vec))])
#plt.plot(z_vec,[SNR_gaussian(mat2[i]) for i in range(len(z_vec))])
plt.legend(['N','variance','SNR','ratio extracted'])
plt.title('Evolution of SNR and extractable SNR with squeezing factor' )
plt.show()

#SV_plots([[],[-1],[-1,-1]])