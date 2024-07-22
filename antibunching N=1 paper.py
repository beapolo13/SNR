from utils import *
from plots import * 
from expectation_values import *
from numpy import tanh

#first we reproduce the results from the paper 


disp=np.linspace(0,1.8,100)
z_vec=np.linspace(0.001,1,100)
temperatures=np.linspace(0.001,0.3,100)
r_vec=np.linspace(0,1,100)

n_eff=[0,0.01,0.1]

def nu(n_eff):
    return 2*n_eff+1
def z_opt(n_eff):
    r_opt= np.arcsinh(np.sqrt(n_eff))
    return np.exp(-2*r_opt)

def analytical_g_zero_temp(d,r):
    return 1 + np.cosh(2*r)/(d**2+np.sinh(r)**2)- (d**2*(1+np.sinh(2*r)))/(d**2 + np.sinh(r)**2)**2


#plot b
# plt.plot(temperatures, [np.arcsinh(np.sqrt(n)) for n in temperatures])
# plt.show()


#plot a 
for n in n_eff:
    if n==0:
        g=[min([antibunching_one_mode(V_thermal(1,[z],[0],[0],params=None),[d,0],[])  for z in z_vec]) for d in disp]
        g_analytical= [min([analytical_g_zero_temp(d,r) for r in r_vec]) for d in disp] 
        plt.plot(disp,g)
        plt.plot(disp,g_analytical,linestyle='dashed')
    else:
        g=[min([antibunching_one_mode(V_thermal(nu(n),[z],[0],[0],params=None),[d,0],[]) for z in z_vec])  for d in disp]
        plt.plot(disp,g)
plt.legend(['0','analytical T=0k','0.01','0.1'])
plt.show()
