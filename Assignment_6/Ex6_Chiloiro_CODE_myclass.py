import numpy as np

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


class NBodyState:
    def __init__(self, N, d, coeffs, separable=False):
        """
        Represents a quantum N-body state.

        Parameters
        ----------
        N : int
            Number of particles.
        d : int
            Dimension of the Hilbert space of each particle.
        coeffs : np.ndarray
            List of N(d-1) (complex) coefficients of the single-particle states if separable is True,
            or list of d^N-1 (complex) coefficients of the basis states if separable is False.
        separable : bool
            If True, given a list of N(d-1) (complex) coefficients of the single-particle states,
            return a separable N-body state.
            If False, given a list of d^N-1 (complex) coefficients of the basis states,
            return a general N-body state.
        """
        # Check input types and shapes
        if not isinstance(N, int):
            raise TypeError("N must be an integer.")
        if not isinstance(d, int):
            raise TypeError("d must be an integer.")
        if not isinstance(coeffs, np.ndarray):
            raise TypeError("coeffs must be a numpy array.")
        if not np.issubdtype(coeffs.dtype, np.complex):
            raise TypeError("coeffs must be a numpy array of complex numbers.")

        if separable:
            if coeffs.shape[0] != N * (d - 1):
                raise ValueError("coeffs must have length N(d-1).")
        else:
            if coeffs.shape[0] != d**N - 1:
                raise ValueError("coeffs must have length d^N-1.")

        # Check if the coefficients are normalized
        if separable:
            for i in range(N):
                if sum(np.abs(coeffs[i * (d - 1) : (i + 1) * (d - 1)])**2) > 1:
                    raise ValueError("coeffs must have norm <= 1.")
        else:
            if sum(np.abs(coeffs)**2) > 1:
                raise ValueError("coeffs must have norm <= 1.")

        # Recover coefficients of the single-particle states or basis states
        if separable:
            full_coeffs = coeffs.reshape((N, d - 1))
            full_coeffs = np.hstack((full_coeffs, np.sqrt(1 - np.linalg.norm(full_coeffs, axis=1)**2)[:, np.newaxis]))
            self.state_vector = np.array([1], dtype=np.complex)
            for i in range(N):
                single_particle_state = full_coeffs[i, :]
                self.state_vector = np.kron(self.state_vector, single_particle_state)
        else:
            full_coeffs = np.append(coeffs, np.sqrt(1 - np.linalg.norm(coeffs)**2))
            self.state_vector = full_coeffs

        self.N = N
        self.d = d


    def density_matrix(self):
        """
        Returns the density matrix of the N-body state.

        Returns
        -------
        np.ndarray
            Density matrix of the N-body state.
        """
        return np.outer(self.state_vector, self.state_vector.conj())
    

    def reduced_density_matrix(self, subsystem, method):
        """
        Returns the reduced density matrix of a subsystem of the 2-body state.

        Parameters
        ----------
        subsystem : int
            Index of the subsystem (0 or 1) whose reduced density matrix is to be computed.
        
        method : int
            If 0, use the formula for the reduced density matrix 
            If 1, tensor the density matrices of the two subsystems and trace out the other subsystem.
            If 2, tesor state vector and directly compute the reduced density matrix.            
        Returns
        -------
        np.ndarray
            Reduced density matrix of the subsystem.
        """
        # Check input type
        if not isinstance(subsystem, int):
            raise TypeError("subsystem must be an integer.")
        if self.N!=2:
            raise ValueError("reduced_density_matrix is only implemented for N=2.")
        if subsystem!=0 and subsystem!=1:
            raise ValueError("subsystem must be either 0 or 1.")
        if not isinstance(method, int):
            raise TypeError("method must be an integer.")
        if method!=0 and method!=1 and method!=2:
            raise ValueError("method must be either 0, 1 or 2.")
        
        # Compute the reduced density matrix (standard formula)
        if method == 0:
            d = self.d
            I = np.eye(d)
            basis = np.eye(d)
            state = self.state_vector
            reduced_rho = np.zeros((d, d), dtype=np.complex)

            for k in basis:
                if subsystem==0:
                    I_tens_base = np.kron(I, k).T
                else:
                    I_tens_base = np.kron(k, I).T
                left = state @ I_tens_base
                right = np.conj(state) @ I_tens_base
                reduced_rho += np.outer(left, right)
            
        # Compute the reduced density matrix (tensoring and tracing)
        if method == 1:
            d = self.d
            rho = self.density_matrix()
            rho_tens = rho.reshape((d, d, d, d))
            if subsystem == 0: #subsystem 1 is traced out
                idxs = (1, 3)
            else: #subsystem 0 is traced out
                idxs = (0, 2)
            reduced_rho = np.trace(rho_tens, axis1=idxs[0], axis2=idxs[1])

        # Compute the reduced density matrix (directly)
        if method == 2:
            if subsystem == 0:
               idxs = 1
            else:
               idxs = 0
            d = self.d
            state = self.state_vector.reshape((d, d))
            reduced_rho = np.tensordot(state, state.conj(), axes=(idxs, idxs))
                
        # check if the trace is 1
        if np.abs(np.trace(reduced_rho)-1) > 1e-8:
            raise ValueError("The trace of the reduced density matrix is not 1.")
        return reduced_rho