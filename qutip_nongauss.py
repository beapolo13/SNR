import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
#from qutip import *
from utils import *
from q_thermo import *
from qutip import coherent,negativity, create, destroy, expect, basis, displace, squeeze, commutator, thermal_dm, ket2dm, mesolve
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from qutip.continuous_variables import covariance_matrix


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


def photon_subtract(rho, cutoff, mode=[1,0]): #the mode vector indicates the weight of the superposition of performing the subtraction on mode 0 and on mode 1 (mode[0]**2+mode[1]**2=1)
    """
    Subtract one photon from `mode` (0 or 1) of the two-mode state `rho`
    """
    a = destroy(cutoff)
    aA = tensor(destroy(cutoff), qt.qeye(cutoff))   # mode A
    aB = tensor(qt.qeye(cutoff), destroy(cutoff))   # mode B
    op = mode[0] * aA + mode[1] * aB
    
    rho_sub = op * rho * op.dag()
    rho_sub = rho_sub / rho_sub.tr()  # Normalize

    return rho_sub

def covariance_matrix_check(t,z,x,cutoff,phi):
    #this function checks if the covariance matrix of the above-created photon subtracted state is the same as that in the paper by Barral et al
    
    modes=2
    rho=photon_subtract(create_two_mode_gaussian_state(t, t, z, 1/z, x, cutoff),cutoff,[np.cos(phi),np.sin(phi)])
    a_ops = [ qt.tensor([qt.destroy(rho.dims[0][j]) if j == m else qt.qeye(rho.dims[0][j])
                   for j in range(modes)]) for m in range(modes)]
    
    # quadratures
    q_ops = [(a + a.dag())/np.sqrt(2) for a in a_ops]
    p_ops = [(a - a.dag())/(1j*np.sqrt(2)) for a in a_ops]
    basis_ops = q_ops + p_ops
    sigma_0 = np.array(State(2,[z,1/z],[x],[0,0],temp=[t,t]).matrix)
    identity4= qt.qeye(4).full()
    def P(phi):
        return np.array([
            [np.cos(phi)**2,   0.5*np.sin(2*phi),0, 0],
            [0.5*np.sin(2*phi),np.sin(phi)**2,0,0],
            [ 0,  0,np.cos(phi)**2,   0.5*np.sin(2*phi)],
            [0, 0, 0.5*np.sin(2*phi),   np.sin(phi)**2]])
    P = P(phi)
    print('test1',sigma_0,identity4,P,np.matmul(sigma_0-identity4,P) )
    print(np.trace((sigma_0-identity4)@P))
    cov1=covariance_matrix(basis_ops,rho)
    cov2=(sigma_0 + 2*(np.matmul((sigma_0-identity4)@P,(sigma_0-identity4)))/np.trace((sigma_0-identity4)@P))/2
    print('Numerical matrix',cov1)
    print('Exact matrix', np.round(cov2,3))
    return cov1 - cov2

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
    plt.xlabel('Cutoff dimension')
    plt.ylabel(r'Energy of state $\rho$')
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
    

    c2=ax.pcolormesh(X_grid,Y_grid,en,cmap=cm.get_cmap('viridis_r', 7))

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


def ergotropy_qutip(rho, H):
    """
    Compute the global ergotropy of a state rho given Hamiltonian H.
    rho, H: qutip.Qobj (same dimensions)
    Returns: (ergotropy, total_energy, passive_energy)
    """
    # Ensure Hermitian
    #print(f'Initial state rho is hermitian: {rho == rho.dag()}')
    #print(f'Hamiltonian is hermitian: {H == H.dag()}')

    #ensure trace is 1
    unit_trace = rho.tr()
    #print(f'Trace of initial state is {unit_trace}')
    
    # Total energy
    E_tot = (rho * H).tr().real

    # Eigenvalues of rho (descending) and H (ascending)
    evals_rho = np.sort(rho.eigenenergies())[::-1]     # largest first
    evals_H = np.sort(H.eigenenergies())               # smallest first

    # Passive state's energy
    E_passive = np.dot(evals_rho, evals_H)
    #print('energy of rho:', E_tot, 'Energy of rho_passive',E_passive)
    return E_tot - E_passive

def number_operator_qutip(dim):
    """Number operator in given Fock dimension."""
    return qt.num(dim)

def two_mode_ho_hamiltonian_qutip(dA, dB, omegaA=1.0, omegaB=1.0, include_zp=False):
    """Two-mode free Hamiltonian."""
    nA = number_operator_qutip(dA)
    nB = number_operator_qutip(dB)
    if include_zp:
        HA = omegaA * (nA + 0.5 * qt.qeye(dA))
        HB = omegaB * (nB + 0.5 * qt.qeye(dB))
    else:
        HA = omegaA * nA
        HB = omegaB * nB
    H = qt.tensor(HA, qt.qeye(dB)) + qt.tensor(qt.qeye(dA), HB)
    return H

def local_ergotropy_qutip(rho, dA, dB, omegaA=1.0, omegaB=1.0):
    """Compute sum of local ergotropies."""
    rhoA = rho.ptrace(0)
    rhoB = rho.ptrace(1)
    HA = omegaA * number_operator_qutip(dA)
    HB = omegaB * number_operator_qutip(dB)
    ergA = ergotropy_qutip(rhoA, HA)
    ergB = ergotropy_qutip(rhoB, HB)
    #print("Local ergotropy sum:",ergA + ergB )
    #print("Local ergotropy A:", ergA, "B:", ergB)
    return ergA + ergB

def relative_ergotropic_gap_qutip(rho,H,dA,dB):
    ge=ergotropy_qutip(rho, H)
    le=local_ergotropy_qutip(rho,dA,dB)
    result= (ge - le)/ge
    print('REG calculated')
    return result

def gaussian_reg_phsub_on_superposition_ofmodes(z1,z2,phi,nu):
  def tests_ph_subnongauss(z1,z2,phi,nu):
    V_0 = np.array([[nu*z1,0,0,0],[0,nu/z1,0,0],[0,0,nu*z2,0],[0,0,0,nu/z2]])
    def P(phi):
        return np.array([
            [np.cos(phi)**2, 0,  0.5*np.sin(2*phi),0],
            [0, np.cos(phi)**2, 0,  0.5*np.sin(2*phi)],
            [0.5*np.sin(2*phi),0, np.sin(phi)**2,0],
            [0, 0.5*np.sin(2*phi), 0,  np.sin(phi)**2]])
    P = P(phi)
    Id = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    result_prel = (V_0 - Id)@P
    trace1 = result_prel.trace()
    result = V_0 + (2/trace1)*(result_prel@(V_0-Id))
    return result

  def apply_local_sq(matrix, s1,s2):
    S = np.array([[s1,0,0,0],[0,1/s1,0,0],[0,0,s2,0],[0,0,0,1/s2]])
    return S @ matrix @S.T
  res= tests_ph_subnongauss(z1,z2,phi,nu)
  #print(res)
  #print(res[1][1], res[3][3])

  std= apply_local_sq(res,np.sqrt(np.sqrt(res[1][1]/res[0][0])),np.sqrt(np.sqrt(res[3][3]/res[2][2])))
  #print(std)
  a= std[0][0]
  b= std[2][2]
  c1 = std[0][2]
  c2= std[1][3]

  #print(a,b,c1,c2)

  gamma = a**2 + b**2 + 2*c1*c2
  #print('gamma', gamma)
  det = np.linalg.det(res)
  #print(det)
  nu_plus= np.sqrt((gamma + np.sqrt(gamma**2 -4*det))/2)
  nu_minus = np.sqrt((gamma - np.sqrt(gamma**2 -4*det))/2)
  #print('nu_plus', nu_plus)
  #print('nu_min', nu_minus)

  EG = 0.5*(a+b-nu_plus-nu_minus)
  eg= 0.5*(nu_plus+nu_minus)-1
  return EG/eg


def plot_general_ergotropy(type_of_state):
    epsilon = 1e-4
    cutoff=10
    hamiltonian=two_mode_ho_hamiltonian_qutip(cutoff,cutoff)
    x_points=100
    y_points=100

    if type_of_state == 'tms photonsub on mode 1':
        z_vec=np.linspace(0.5,1,x_points)
        r_vec= np.array([-np.log(z)/2 for z in z_vec])
        t_vec = np.linspace(0.1,6,y_points)
        k_vec= np.array([1/np.tanh((1/(2*t))) for t in t_vec])
        x= np.pi/4
        X=z_vec
        Y=k_vec
        #W = [[np.float64(find_ergotropic_gap_phsub(z,k)) for z in z_vec] for k in k_vec]
        W = [[np.float64(relative_ergotropic_gap_qutip(photon_subtract(create_two_mode_gaussian_state(t_vec[j], t_vec[j], z, 1/z, x, cutoff),cutoff,[1,0]),hamiltonian,cutoff,cutoff)) for z in z_vec] for j in range(len(k_vec))]
        en = [[np.float64(log_negativity(photon_subtract(create_two_mode_gaussian_state(t_vec[j], t_vec[j], z,1/z, x, cutoff),cutoff,[1,0]))) for z in z_vec] for j in range(len(k_vec))]

    if type_of_state == 'tms photonsub on superposition of modes':
        z1=0.5
        z2=0.5
        phi_vec=np.linspace(0,np.pi/2,x_points)
        t_vec = np.linspace(0.1,6,y_points)
        k_vec= np.array([1/np.tanh((1/(2*t))) for t in t_vec])
        x= 0
        X=phi_vec
        Y=k_vec
        #W = [[np.float64(gaussian_reg_phsub_on_superposition_ofmodes(z1,z2,phi,nu)) for phi in phi_vec] for nu in k_vec]
        #W = [[np.float64(relative_ergotropic_gap_qutip(photon_subtract(create_two_mode_gaussian_state(t_vec[j], t_vec[j], z1, z2, x, cutoff),cutoff,[np.cos(phi),np.sin(phi)]),hamiltonian,cutoff,cutoff)) for phi in phi_vec] for j in range(len(k_vec))]
        en = [[np.float64(log_negativity(photon_subtract(create_two_mode_gaussian_state(t_vec[j], t_vec[j], z1,z2, x, cutoff),cutoff,[np.cos(phi),np.sin(phi)]))) for phi in phi_vec] for j in range(len(k_vec))]

    if type_of_state == 'noon':
        def create_mixed_noon(N,t,p):
            noon = (qt.tensor(qt.basis(cutoff, N), qt.basis(cutoff, 0)) + qt.tensor(qt.basis(cutoff, 0), qt.basis(cutoff, N))).unit()
            rho_noon = noon.proj()
            thermal1 = qt.thermal_dm(cutoff, n_th(t))
            thermal2 = qt.thermal_dm(cutoff, n_th(t))
            rho_thermal = qt.tensor(thermal1, thermal2)
            return p * rho_noon + (1 - p) * rho_thermal
        
        p_vec=np.linspace(0.1,1,x_points)   #mixedness
        t_vec = np.linspace(0.1,3,y_points)   
        k_vec= np.array([1/np.tanh((1/(2*t))) for t in t_vec])
        X=p_vec
        Y=k_vec
        W = [[np.float64(relative_ergotropic_gap_qutip(create_mixed_noon(5,t_vec[j],p),hamiltonian,cutoff,cutoff)) for p in p_vec] for j in range(len(k_vec))]
        en = [[np.float64(log_negativity(create_mixed_noon(3,t_vec[j],p))) for p in p_vec] for j in range(len(k_vec))]

    if type_of_state == 'cat':
        def create_2mode_mixed_cat(alpha,p):
            cs_pp = qt.tensor(qt.coherent(cutoff, alpha), qt.coherent(cutoff, alpha))
            cs_mm = qt.tensor(qt.coherent(cutoff, -alpha), qt.coherent(cutoff, -alpha))
            rho_cat2 = (cs_pp + cs_mm).unit().proj()
            rho_cat2_odd = (cs_pp - cs_mm).unit().proj()
            rho_th = qt.tensor(qt.thermal_dm(cutoff, n_th(t)), qt.thermal_dm(cutoff, n_th(t)))
            return p * rho_cat2 + (1 - p) * rho_cat2_odd
        
        p_vec=np.linspace(0.1,1,x_points)   #mixedness
        alpha_vec = np.linspace(0.1,6,y_points)   
        k_vec= np.array([1/np.tanh((1/(2*t))) for t in t_vec])
        X=alpha_vec
        Y=p_vec
        W = [[np.float64(relative_ergotropic_gap_qutip(create_2mode_mixed_cat(alpha_vec[j],p),hamiltonian,cutoff,cutoff))  for j in range(len(alpha_vec))] for p in p_vec]
        en = [[np.float64(log_negativity(create_2mode_mixed_cat(alpha_vec[j],p)))  for j in range(len(alpha_vec))] for p in p_vec]


    X_grid, Y_grid =np.meshgrid(X,Y)
    fig,ax=plt.subplots(1,1,figsize=(10,6))
    c1=ax.pcolormesh(X_grid,Y_grid,en,cmap=cm.get_cmap('viridis_r', 20))
    #c2=ax[1].pcolormesh(X_grid,Y_grid,en,cmap=cm.get_cmap('viridis_r', 50))
    #c2=ax[1].pcolormesh(X_grid,Y_grid,sep,cmap=cm.get_cmap('viridis', 10))
    cbar1=fig.colorbar(c1,ax=ax, label=r'$\Delta \epsilon_{\text{rel}}$')
    #cbar2=fig.colorbar(c2,ax=ax[1], label=f'logarithmic negtivity')
    #contour = ax.contour(X_grid, Y_grid,en,levels=[epsilon], colors='black', linestyles='dashed', linewidths=1.5)

    
    #ax.clabel(contour, inline=True, fontsize=30,fmt='log neg')
    #ax[1].clabel(contour, inline=True, fontsize=10,fmt='PPT')
    # ax.set_xlim(X.min(), X.max())
    ax.set_yscale('log')
    #ax[1].set_yscale('log')
    # ax.set_ylim(Y.min() , Y.max())
    # #ax.grid(True, which='both', linestyle='--')
    # ax.set_ylabel(r'$k$')
    # ax.set_yticks(ticks=[1,10], labels=['1', '10'])
    # ax.set_xlabel(r' $z$')

    plt.subplots_adjust(wspace=2)
    y=[1] + [1 + i for i in range(1,8)] + [10]
    ax.set_yticks(y)
    #ax[1].set_yticks(y)
    #beep()
    plt.savefig('Photon subtraction on superposition of modes_ LOG NEG')
    beep()
    plt.show()
    return

def tests_ph_subnongauss(z1,z2,phi,nu):
    V_0 = np.array([[nu*z1,0,0,0],[0,nu/z1,0,0],[0,0,nu*z2,0],[0,0,0,nu/z2]])
    def P(phi):
        return np.array([
            [np.cos(phi)**2, 0,  0.5*np.sin(2*phi),0],
            [0, np.cos(phi)**2, 0,  0.5*np.sin(2*phi)],
            [0.5*np.sin(2*phi),0, np.sin(phi)**2,0],
            [0, 0.5*np.sin(2*phi), 0,  np.sin(phi)**2]])
    P = P(phi)
    Id = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    result_prel = (V_0 - Id)@P
    trace1 = result_prel.tr()
    result = V_0 + (2/trace1)*(result_prel@(V_0-Id))
    return result


#covariance_matrix_check(2,0.6,np.pi/4,35,np.pi/6)
plot_general_ergotropy('tms photonsub on superposition of modes')
#PARAMETERS
t1= 1.5  #temperature of first mode (between 0 and infty)
t2= 1.5 #temperature of second mode (between 0 and infty)
z1= 0.3  #squeezing of first mode (between 0 and 1)
z2= 0.15 #squeezing of second mode  (between 0 and 1)
theta= np.pi/4  #beam splitter angle
cutoff= 8 #dimension of hilbert space cutoff
max_cutoff=30

#truncation_comparison(t1,t2,z1,z2,theta,max_cutoff)

#Example usage
dA, dB = cutoff, cutoff
#rhogauss = create_two_mode_gaussian_state(t1, t2, z1,z2, theta, cutoff)
#rho= photon_subtract(rhogauss,cutoff)
#H = two_mode_ho_hamiltonian_qutip(dA, dB)
# Compute global ergotropy
#erg_global = ergotropy_qutip(rho, H)
#print("Global ergotropy:", erg_global)
#print('Local ergotropy', local_ergotropy_qutip(rho,dA,dB))
# Compute local ergotropy sum




