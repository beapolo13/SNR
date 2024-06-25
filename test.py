from utils import *
from plots import * 
from expectation_values import *
from numpy import tanh
x=np.linspace(0.00000001,np.pi,100)
nu=3
z_vec=np.linspace(0.0001,1,1000)
mat_worst=[V_thermal(nu,[z,1/z],[0],[0,0],params=None) for z in z_vec]
sigma0=V_thermal(nu,[1,1],[0],[0,0],params=None)
mat_best=[V_thermal(nu,[z,1/z],[np.pi/4],[0,0],params=None) for z in z_vec]


plt.plot(z_vec,[SNR_gaussian_extr(mat_worst[i],sigma0) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_ng_extr(mat_worst[i],[-1],sigma0) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_ng_extr(mat_worst[i],[+1],sigma0) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_ng_extr(mat_worst[i],[-1,-1],sigma0) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_ng_extr(mat_best[i],[-1,-1],sigma0) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_ng_extr(mat_worst[i],[+1,+1],sigma0) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_ng_extr(mat_best[i],[+1,+1],sigma0) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_ng_extr(mat_worst[i],[+1,+1,+1],sigma0) for i in range(len(z_vec))])
plt.plot(z_vec,[SNR_ng_extr(mat_best[i],[+1,+1,+1],sigma0) for i in range(len(z_vec))])



plt.legend(['gaussian','1 photon sub','1 photon add','worst 2 photon sub','best 2 photon sub','worst 2 photon add','best 2 photon add'])
plt.xlabel('squeezing factor z')
#plt.title('Evolution of SNR and extractable SNR with squeezing factor' )
plt.savefig('bounds')
plt.show()


