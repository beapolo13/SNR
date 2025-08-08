import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
#from qutip import *
from utils import *
from qutip import coherent,negativity, create, destroy, expect, basis, displace, squeeze, commutator, thermal_dm, ket2dm, mesolve
from scipy.optimize import minimize
from scipy.linalg import expm

def r(z):
    return -np.log(z)/2

def n_th(t):
    nu = 1/np.tanh(1/(2* t))
    return (nu-1)/2 

def beamsplitter_operator(theta, cutoff):
    """
    Returns the two-mode beam splitter operator with angle theta.
    The transmissivity is T = cos^2(theta)
    """
    a1 = qt.tensor(qt.destroy(cutoff), qt.qeye(cutoff))
    a2 = qt.tensor(qt.qeye(cutoff), qt.destroy(cutoff))
    return (-theta * (a1.dag() * a2 - a1 * a2.dag())).expm()



def create_two_mode_gaussian_state(t1, t2, z1,z2, theta, cutoff):
    """
    Create a two-mode Gaussian state without displacement:
    - n_th: thermal photon number (same for both modes)
    - r: single-mode squeezing parameter (same for both modes)
    - T: beam splitter transmissivity
    - cutoff: Fock space truncation
    """
    # Thermal states
    rho_th1 = qt.thermal_dm(cutoff, n_th(t1))
    rho_th2 = qt.thermal_dm(cutoff, n_th(t2))

    # Squeezing
    S1 = qt.squeeze(cutoff, r(z1))
    S2 = qt.squeeze(cutoff, r(z2))
    rho_sq1 = S1 * rho_th1 * S1.dag()
    rho_sq2 = S2 * rho_th2 * S2.dag()

    # Combine into two-mode product state
    rho_in = qt.tensor(rho_sq1, rho_sq2)

    # Beam splitter
    BS = beamsplitter_operator(theta,cutoff)

    # Apply BS
    rho_out = BS * rho_in * BS.dag()

    return rho_out


def photon_subtract(rho, cutoff, mode=0):
    """
    Subtract one photon from `mode` (0 or 1) of the two-mode state `rho`
    """
    a = destroy(cutoff)
    if mode == 0:
        A = qt.tensor(a, qt.qeye(cutoff))
    elif mode == 1:
        A = qt.tensor(qt.qeye(cutoff), a)
    else:
        raise ValueError("Mode must be 0 or 1 for two-mode states")
    
    rho_sub = A * rho * A.dag()
    rho_sub = rho_sub / rho_sub.tr()  # Normalize

    return rho_sub

def log_negativity(rho):
    return negativity(rho, 1, method='tracenorm', logarithmic=True)

def compute_energy(rho, cutoff=10):
    """
    Compute the total energy (mean photon number) of the two-mode state
    """
    n = qt.num(cutoff)
    H = qt.tensor(n, qt.qeye(cutoff)) + qt.tensor(qt.qeye(cutoff), n)
    return qt.expect(H, rho)


def truncation_comparison(t1, t2, z1,z2, theta, max_cutoff): #we fix some random gaussian parameters and compare how the energy computation approaches the exact value as the cutoff increases
    cutoff_vector =  np.arange(3,max_cutoff,1)
    energy_vector=[]
    energy_exact = State(2,[z1,z2],[theta],[0,0],temp=[t1,t2],nongaussian_ops=[-1]).expvalN()
    for cut in cutoff_vector:
        rhogauss = create_two_mode_gaussian_state(t1, t2, z1,z2, theta, cut)
        rho_final= photon_subtract(rhogauss,cut)
        energy= compute_energy(rho_final,cut)
        energy_vector += [energy]
    print(energy_exact)
    plt.plot(cutoff_vector,energy_vector)
    plt.show()
    return

#optimization of unitary operation


def basis_set(dim):
    """Create an orthonormal basis set of Hermitian matrices (fixed for optimization)"""
    # Just once for speed; consider caching
    bases = []
    for i in range(dim):
        for j in range(i, dim):
            mat = np.zeros((dim, dim), dtype=complex)
            if i == j:
                mat[i, j] = 1
            else:
                mat[i, j] = mat[j, i] = 1
                bases.append(qt.Qobj(mat))
                mat = np.zeros((dim, dim), dtype=complex)
                mat[i, j] = -1j
                mat[j, i] = 1j
            bases.append(qt.Qobj(mat))
    return bases

def optimize_unitary(rho, cutoff):
    dim = cutoff ** 2
    bases = basis_set(dim)

    def energy_from_params(params):
        H = sum(p * B for p, B in zip(params, bases))
        H_np = H.full()  # H is a Qobj, .full() returns the numpy array
        U = expm(-1j * H_np)
        U_qobj = qt.Qobj(U, dims=[[cutoff, cutoff], [cutoff, cutoff]])
        rho_new = U_qobj * rho * U_qobj.dag()
        return compute_energy(rho_new, cutoff)

    # Initial guess
    x0 = np.random.random(len(bases))
    result = minimize(energy_from_params, x0, method='L-BFGS-B')
    min_energy = result.fun
    print(result)
    return min_energy, result

def plot_log_en():
    z_vec=np.linspace(0.1,1,30)
    
    r_vec=np.array([-np.log(z)/2 for z in z_vec])
    t_vec = np.linspace(0.1,8,30)
    print(z_vec, t_vec)
    
    k_vec= np.array([1/np.tanh((1/(2*t))) for t in t_vec])
    X=z_vec
    Y=k_vec
    X_grid, Y_grid =np.meshgrid(X,Y)
    x= np.pi/4
    epsilon = 1e-6

    en = [[np.float64(log_negativity(photon_subtract(create_two_mode_gaussian_state(t_vec[j], t_vec[j], z,1/z, np.pi/4, 20),20))) for z in z_vec] for j in range(len(k_vec))]
    print('array computed')
    W_arr= np.array(en)
 
    

    fig,ax=plt.subplots(1,1,figsize=(10,6))
    

    c2=ax.pcolormesh(X_grid,Y_grid,en,cmap=cm.get_cmap('viridis_r', 40))

    #c2=ax[1].pcolormesh(X_grid,Y_grid,sep,cmap=cm.get_cmap('viridis', 10))
    cbar=fig.colorbar(c2,ax=ax, label=r'$\Delta \epsilon_{\text{rel}}$')
    contour = ax.contour(X_grid, Y_grid, en,levels=[0], colors='black', linestyles='dashed', linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=30,fmt='log neg')
    #ax[1].clabel(contour, inline=True, fontsize=10,fmt='PPT')
    ax.set_xlim(X.min(), X.max())
    ax.set_yscale('log')
    ax.set_ylim(Y.min() , Y.max())
    #ax.grid(True, which='both', linestyle='--')
    ax.set_ylabel(r'$k$')
    ax.set_yticks(ticks=[1,10], labels=['1', '10'])
    ax.set_xlabel(r' $z$')

    plt.subplots_adjust(wspace=2)
    y=[1] + [1 + i for i in range(1,8)] + [10]
    ax.set_yticks(y)
    #beep()
    #plt.savefig()
    plt.show()
    return


plot_log_en()


#PARAMETERS
t1= 0.5  #temperature of first mode (between 0 and infty)
t2= 0.5 #temperature of second mode (between 0 and infty)
z1= 0.3  #squeezing of first mode (between 0 and 1)
z2= 0.7 #squeezing of second mode  (between 0 and 1)
theta= 0  #beam splitter angle
cutoff= 10 #dimension of hilbert space cutoff
max_cutoff=20


#truncation_comparison(t1, t2, z1,z2, theta, max_cutoff)




rhogauss = create_two_mode_gaussian_state(t1, t2, z1,z2, theta, cutoff)
min_energy_exact = State(2,[1,1],[0],[0,0],temp=[t1,t2],nongaussian_ops=[]).expvalN()
#rho= photon_subtract(rhogauss,cutoff)
print(min_energy_exact)
#optimize_unitary(rhogauss, cutoff)