import numpy as np
from scipy.sparse import csr_matrix, kron, identity



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



def QuantumIsingModel(N, h, J=1., dense=False, local_int=False):
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
    # add the periodic boundary condition (if N>2)
    if N > 2:
        n = 0
        first_term = kron(kron(identity(2**n, format='csr'), sigma_x), identity(2**(N-n-1), format='csr')) 
        n = N-1
        last_term = kron(kron(identity(2**n, format='csr'), sigma_x), identity(2**(N-n-1), format='csr')) 
        H_int += last_term.dot(first_term)

    H = h*H_local + J*H_int
    if dense:
        H = H.todense()   
    return H    

def memory_usage(N):
    """
    Returns the memory usage of the quantum Ising model Hamiltonian with N spins on a 1D lattice, with periodic boundary conditions.
    """
    print(f'Memory usage for H ~ {round((2**(N*2))*8*1e-9, 1)} GB')