from utils import *
from plots import * 
from expectation_values import *
from numpy import tanh
x=np.linspace(0.00000001,1,10)
T=0.5
z_vec=np.linspace(0.1,4,1000)
mat=[V_thermal([T]*2,[z,1/z],[0],[0,0],params=None) for z in z_vec]
sigma0=V_thermal([T]*2,[1,1],[0],[0,0],params=None)
print(sigma0)
operations=[-1]
print('K',K_ng(sigma0,operations))
print('N',expvalN_ng(sigma0,operations))
fig, (ax1,ax2) = plt.subplots(1,2)
#mat2=[V_thermal([1]*2,[z,1/z],[np.pi/4],[0],[0,0],[0,0],params1=None,params2=None) for z in z_vec]
#plt.plot(z_vec,[expvalN(mat[i]) for i in range(len(z_vec))])
#plt.plot(z_vec,[varianceN(mat[i]) for i in range(len(z_vec))])
ax1.plot(z_vec,[expvalN_ng(mat[i],operations)+ 1 for i in range(len(z_vec))])
#plt.plot(z_vec,[expvalN_ng(mat[i],operations)-expvalN_ng(sigma0,operations) for i in range(len(z_vec))])
ax1.plot(z_vec,[varianceN_ng(mat[i],operations) for i in range(len(z_vec))])
#plt.plot(z_vec,[SNR_ng(mat[i],operations) for i in range(len(z_vec))])
ax2.plot(z_vec,[(expvalN_ng(mat[i],operations)-expvalN(sigma0))/varianceN_ng(mat[i],operations) for i in range(len(z_vec))])
#plt.plot(z_vec,[expvalN(mat2[i]) for i in range(len(z_vec))])
#plt.plot(z_vec,[SNR_gaussian(mat2[i]) for i in range(len(z_vec))])
ax1.legend(['E','variance'])
ax2.legend(['SNR extr'])
plt.xlabel('squeezing factor z')
#plt.title('Evolution of SNR and extractable SNR with squeezing factor' )
plt.savefig('[-1]case_with noise 0.5.png')
plt.show()


#SV_plots([[],[-1],[-1,-1]])