import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigs

def checkpoint(debug=False, msg='checkpoint', vars=None):
    """
    Print a checkpoint message and optionally some variables.

    Parameters
    ----------
    debug : bool
        If True, print the message and variables.
    msg : str   
        The message to print.
    vars : dict
        A dictionary of variables to print.

    Returns
    -------
    None
    """
    if debug:
        print(msg)
        if vars is not None:
            for var in vars:
                print(var, ':', vars[var])
                


def QuantumIsingModel(N, h, J=1., dense=True, local_int=False, periodic_boundary=False):
    """
    Returns the Hamiltonian of the quantum Ising model with N spins on a 1D lattice, with periodic boundary conditions.

    Parameters
    ----------
    N : int
        Number of spins
    h : float
        Magnetic field
    J : float
        Coupling constant (default: 1)
    dense : bool
        If True, return a dense matrix instead of a sparse matrix (default: False)
    local_int : bool
        If True, return a matrix with only the local interaction (default: False)
    periodic_boundary : bool
        If True, add the periodic boundary condition (default: False)

    Returns
    -------
    H : scipy.sparse.csr_matrix
        Hamiltonian of the quantum Ising model
    """
    # check input
    if not isinstance(N, int):
        raise TypeError("N must be an integer")
    if not isinstance(h, float):
        raise TypeError("h must be a float")
    if not isinstance(J, float):
        raise TypeError("J must be a float")
    if N < 2:
        raise ValueError("N must be greater than 1")

    # construct Hamiltonian
    sigma_z = csr_matrix(np.array([[1, 0], 
                                   [0,-1]]))
    sigma_x = csr_matrix(np.array([[0, 1],
                                   [1, 0]]))

    # local interaction (magnenic field)
    H_local = csr_matrix((2**N, 2**N))
    for n in range(N):
    # i-th matrix: I_0 x I_1 x ... x sigma_z x I_i+1 x ... x I_N-1
        # on the left of sigma_z
        left_tens_prod = kron(identity(2**n, format='csr'), sigma_z)
        # on the right of sigma_z
        H_local += kron(left_tens_prod, identity(2**(N-n-1), format='csr'))
    if local_int:
        if dense:
            H_local = H_local.todense()
        return h*H_local

    # 2-body interaction
    H_int = csr_matrix((2**N, 2**N))
    for n in range(N-1):
        # i-th term
        ith_term = kron(kron(identity(2**n, format='csr'), sigma_x), identity(2**(N-n-1), format='csr'))
        # i+1-th term
        i1th_term = kron(kron(identity(2**(n+1), format='csr'), sigma_x), identity(2**(N-n-2), format='csr'))
        # then..
        H_int += ith_term.dot(i1th_term)

    if periodic_boundary:
        # add the periodic boundary condition (if N>2)
        if N > 2:
            n = 0
            first_term = kron(kron(identity(2**n, format='csr'), sigma_x), identity(2**(N-n-1), format='csr')) 
            n = N-1
            last_term = kron(kron(identity(2**n, format='csr'), sigma_x), identity(2**(N-n-1), format='csr')) 
            H_int += last_term.dot(first_term)

    H = h*H_local + J*H_int
    if dense:
        H = np.array(H.todense(), dtype=np.complex128)
    return H    



def RealSpaceRG(N0, num_iter, tol, H, **kwargs):
    """
    Real-space renormalization group transformation for a 1-dimensional transverse-field Ising model.

    Parameters
    ----------
    N0 : int
        Number of spins in the initial lattice
    num_iter : int
        Max number of iterations of the RG transformation (final lattice size is N0*2**num_iter)
    tol : float
        Tolerance for the convergence of the algorithm
    H : function
        Function that returns the Hamiltonian of the system (must return a numpy array dtype=np.complex128)
    **kwargs : dict
        Parameters of the Hamiltonian

    Returns
    -------
    e : np.ndarray
        Ground energy per spin for each iteration of the RG transformation
    N : np.ndarray
        Number of spins for each iteration of the RG transformation
    """
    # check input
    if not isinstance(N0, int):
        raise TypeError("N0 must be an integer")
    if not isinstance(num_iter, int):
        raise TypeError("num_iter must be an integer")
    if N0 < 2:
        raise ValueError("N0 must be greater than 1")
    if num_iter < 1:
        raise ValueError("num_iter must be greater than 0")
    
    # ALGORITHM

    sigma_x = np.array([[0, 1],
                        [1, 0]])
    m = 2**N0

    ## 1. Initialize the Hamiltonian
    H0 = H(N0, **kwargs)
    ### check type
    if not isinstance(H0, np.ndarray):
        raise TypeError("Hamiltonian must be a numpy array")
    if H0.dtype != np.complex128:
        raise TypeError("Hamiltonian must be of type np.complex128")
    
    e = np.array([])
    N = np.array([])

    n = 0
    convergence = False
    while ((n != num_iter) & (not convergence)):
        ## 2. Double the lattice size
        H_R = np.kron(np.eye(m), H0)
        H_L = np.kron(H0, np.eye(m))
        ### last spin of the left side, and first of the right one
        left_int = np.kron(np.kron(np.eye(int(m/2)), sigma_x), np.eye(m))
        right_int = np.kron(np.eye(m), np.kron(sigma_x, np.eye(int(m/2))))
        ### interaction
        H_int = left_int @ right_int 
        ### doubled Hamiltonian
        H_2 = H_L + H_R + H_int
        N = np.append(N, N0*2**(n+1))

        # 3. Diagonalize the Hamiltonian and project it 
        E, V = np.linalg.eigh(H_2)
        e = np.append(e, E[0]/(N[n]))
        ### Projector
        P = V[:, :m]
        ### Projection of H_2
        H0 = P.T.conj() @ (H_2 @ P)
        # Check convergence
        if n > 0:
            if np.abs(e[n] - e[n-1]) < tol:
                convergence = True
        n += 1
    # if (n == num_iter) & (not convergence):
    #     print('Warning: algorithm did not converge within the specified number of iterations')
    return e, N
    
    
    
def reduced_density_matrix(state_vector, subsystem):
	"""
	Compute the reduced density matrix of a quantum state.

	Parameters
	----------
	state_vector : np.ndarray
	    State vector of the quantum state
	subsystem : int
	    Subsystem for which to compute the reduced density matrix (left: 0, right: 1)

	Returns
	-------
	reduced_rho : np.ndarray
	    Reduced density matrix of the quantum state
	"""
	# check input
	if not isinstance(state_vector, np.ndarray):
	    raise TypeError("state_vector must be a numpy array")
	if not isinstance(subsystem, int):
	    raise TypeError("subsystem must be an integer")
	if subsystem != 0 and subsystem != 1:
	    raise ValueError("subsystem must be 0 or 1")

	# Compute the reduced density matrix (directly)
	if subsystem == 0:
	    idxs = 1
	else:
	    idxs = 0
	d = int(np.sqrt(len(state_vector)))
	if d**2 != len(state_vector):
	    raise ValueError("state_vector must be a vector of length d^2")
	state = state_vector.reshape((d, d))
	reduced_rho = np.tensordot(state, state.conj(), axes=(idxs, idxs))
		
	# check if the trace is 1
	if np.abs(np.trace(reduced_rho)-1) > 1e-8:
	    raise ValueError("The trace of the reduced density matrix is not 1.")
	return reduced_rho



def QIM_DMRG(h, max_num_steps=10**3, tol=10**-5, J=1.):
    """
    Perform the infinite-system DMRG algorithm for the quantum Ising model.

    Parameters
    ----------
    h : float
        Transverse field strength
    max_num_steps : int
        Maximum number of DMRG steps
    tol : float
        Convergence tolerance
    J : float
        Coupling strength

    Returns
    -------
    e : np.ndarray
        Energy per site
    N : np.ndarray
        Number of sites
    truncated_weight : np.ndarray
        Truncation error
    """
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Initial block (1 site)
    H_B = h*sigma_z                                                                       #2x2
    # Interaction term
    sigma_x_B = sigma_x
    # One site
    H_site = h*sigma_z                                                                    #2x2
    
    e = np.array([])
    N = np.array([])
    truncated_weight = np.array([])
    n = 0
    convergence = False
    while ((not convergence) and (n < max_num_steps)):
        # Interaction between block and site
        H_int_bs_L = J*np.kron(sigma_x_B, sigma_x)                                        #4x4
        # Left enlarged block
        H_L = np.kron(H_B, np.eye(2)) + np.kron(np.eye(2), H_site) + H_int_bs_L           #4x4
        # Interaction term of the left enlarged block within the superblock
        H_int_L = np.kron(np.eye(2), sigma_x)                                             #4x4                          

        # Reflected enlarged block
        # Interaction between block and site
        H_int_bs_R = J*np.kron(sigma_x, sigma_x_B)  
        H_R = np.kron(np.eye(2), H_B) + np.kron(H_site, np.eye(2)) + H_int_bs_R           #4x4
        # Interaction term of the reflected enlarged block within the superblock
        H_int_R = np.kron(sigma_x, np.eye(2))                                             #4x4

        # Interaction within the superblock
        H_int = J*np.kron(H_int_L, H_int_R)                                               #16x16

        # Superblock Hamiltonian
        H_SB = np.kron(H_L, np.eye(4)) + np.kron(np.eye(4), H_R) + H_int                  #16x16  

        # Find the ground state of the superblock Hamiltonian
        E, psi_0 = np.linalg.eigh(H_SB)
        psi_0 = psi_0[:, 0]
        N = np.append(N, 4+n*2)
        e = np.append(e, E[0]/(N[n]))

        # Reduced density matrix of the left block
        rho_L = reduced_density_matrix(psi_0, 0)                                          #4x4

        # Diagonalize the reduced density matrix
        E_L, U_L = np.linalg.eigh(rho_L)              
        # Sort the eigenvalues and eigenvectors
        idx = E_L.argsort()[::-1]
        E_L = E_L[idx]
        U_L = U_L[:, idx]                                                             
        # Select the two largest eigenvalues
        E_L = E_L[:2]
        U_L = U_L[:, :2]                                                                  #4x2
        # Compute the truncation error
        truncated_weight = np.append(truncated_weight, 1 - np.sum(E_L))

        # Project the Hamiltonian onto the truncated basis
        H_B = np.dot(np.dot(U_L.conj().T, H_L), U_L)                                      #2x2
        sigma_x_B = np.dot(np.dot(U_L.conj().T, H_int_L), U_L)                            #2x2

        if n > 0:
            if np.abs(e[n]-e[n-1]) < tol:
                convergence = True
        n += 1
    return e, N, truncated_weight
