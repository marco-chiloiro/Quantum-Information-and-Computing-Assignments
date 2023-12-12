import numpy as np
from scipy.integrate import simpson
from scipy.special import hermite
import matplotlib.pyplot as plt



def eigenstate(x, n, m=1., w=1.):
    """
    Compute the n-th eigenstate value in the position x of a time-independent harmonic oscillator.
    
    Parameters
    ----------
    x : float
	spatial position, argument of the eigenfunction.
    n : int
        Quantum number.
    m : float, default=1.
    	particle mass
    w : float, default=1.
    	angular frequency of the oscillator

    Returns
    -------
    psi_n(x) : float
        n-th eigenstate value in the position x.
    """
    norm_const = 1./np.sqrt(2.**n*np.math.factorial(n)) * (m*w/(np.pi))**(1/4)
    gaussian = np.exp(-m*w*x**2/(2))
    H_n = hermite(n)
    
    return norm_const * gaussian * H_n(np.sqrt(m*w)*x)



def amplitude(x, n, m=1., w=1.):
    """
    Compute the n-th eigenstate amplitude value in the position x of a time-independent harmonic oscillator.
    
    Parameters
    ----------
    x : float
	spatial position, argument of the eigenfunction.
    n : int
        quantum number
    m : float, default=1.
    	particle mass
    w : float, default=1.
    	angular frequency of the oscillator

    Returns
    -------
    |psi_n(x)|^2 : float
        n-th eigenstate amplitude value in the position x.
    """
    norm_const = 1./np.sqrt(2.**n*np.math.factorial(n)) * (m*w/(np.pi))**(1/4)
    gaussian = np.exp(-m*w*x**2/(2))
    H_n = hermite(n)
    f = norm_const * gaussian * H_n(np.sqrt(m*w)*x)
    
    return np.abs(f)**2



def energy_level(n, w=1.):
    """
    Compute the n-th energy level of a time-independent harmonic oscillator.
    
    Parameters
    ----------
    n : int
        quantum number
    w : float, default=1.
    	angular frequency of the oscillator

    Returns
    -------
    E_n : float
        Hermite polynomial.
    """        
    return (2*n+1)*w/2




def kinetic_energy(N, delta_x, order='second'):
    """
    Kinetic operator discretization.
    
    Parameters
    ----------
    N : int
        number of discretization steps
    delta_x : float
        discretization step size
    order : string, default='second' 
    	'second' for second-order approximation;
    	'fourth' for fourth-order approximation

    Returns
    -------
    K : np.array of shape (N,N)
        NxN matrix representing the discretized kinetic operator
    """  
    # matrix inizialization
    K = np.zeros((N,N))
    
    if order == 'second':
        for i in range(N):
            K[i,i] = 2
            if i>0:
                K[i,i-1] = -1
            if i<N-1:
                K[i, i+1] = -1
    
    if order == 'fourth':
        for i in range(N):
            K[i,i] = 5/2
            if i>0:
                K[i,i-1] = -4/3
            if i<N-1:
                K[i, i+1] = -4/3
            if i>1:
                K[i, i-2] = 1/12
            if i<N-2:
                K[i, i+2] = 1/12
    
    return (1/(2*delta_x**2))*K



def potential(x, w=1):
    """
    Harmonic potential (1-D) discretization at the position x.
    
    Parameters
    ----------
    x : float
        particle position
    w : float, default=1.
    	angular frequency of the oscillator
    	
    Returns
    -------
    diag : np.array of shape (N,)
        diagonal (size N) of the discretized harmonic potential.
        """
    diag = (w**2)*(x**2)/2
    return diag


    
def normalize_eigenfunctions(V, x_d):
    """
    Given the eigenstates matrix, returns them normalized.
    
    Parameters
    ----------
    V : np.array of shape (N,N)
        Eigenstates matrix, where each column represents an eigenstate.
    x_d : np.arra of shape (N,)
    	Vector collecting the discretized positions

    Returns
    -------
    V_norm : np.array of shape (N,N)
        Eigenstates matrix, where each column represents an eigenstate normalized with respect its amplitude.
    """
    N = V.shape[0]
    norms = np.zeros(N)
    for ii in range(N):
        norms[ii] = np.sqrt(simpson(np.abs(V[:,ii])**2, x=x_d))
    return V / norms
    
    
    
def AE_threshold(est_energies, th):
    """
    Determine the maximum energy level quantum number (n) such that the absolute error between the analytically
    derived energy and its numerically estimated counterpart remains below a specified threshold.
    
    Parameters
    ----------
    est_energies : np.array
        Estimated energies
    th : real
        AE threshold

    Returns
    -------
    nn : int
        maximum energy level quantum number such that the absolute error between the analytically
        derived energy and its numerically estimated counterpart remains below a specified threshold.
        """
    # find the n for which the ADE is larger than 0.5.
    AE, nn = 0, 0
    while(AE<th):
        AE = np.abs(energy_level(nn)-est_energies[nn])
        nn += 1
    return nn


    
def poly_fit_function(x, m, c):
    """
    Parametric curve of the type c*x^m

    Parameters
    ----------
    x : real 
        curve variable
    c, m : real
        curve parameters

    Returns
    -------
    y : real
    """
    return c*x**m



def LinearfFitPlot(data, title, round_const, round_exp):
    """
    Linear fit of data, returns the optimal parameters (least-square) and plot results.

    Parameters
    ----------
    data : dataframe 
        two columns: N and time
    title: string
        title of the plot
    round_const: integer
        number of significative numebers show in the legend plot for the multiplicative parameter
    round_exp: integer
        number of significative numebers show in the legend plot for the exponent parameter
    Returns
    -------
    m, c : real
        optimal parameters
    """
    t = data['Time(s)']
    N = data['N']
    # fit
    x, y = np.log10(N), np.log10(t)
    m, log_c = np.polyfit(x, y, 1)
    c = 10**log_c
    
    # plot 
    _, ax = plt.subplots(figsize=(10,7))
    
    ax.scatter(N, t, label='Data')
    
    x_fit = np.linspace(min(data['N']), max(data['N']), 100)
    y_fit = poly_fit_function(x_fit, m, c)
    ax.plot(x_fit, y_fit, label=f'Fit: y={round(c,round_const)}x^{round(m,round_exp)}', color='orange')
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('N', fontsize=18)
    ax.set_ylabel('t (s)', fontsize=18)
    ax.legend(fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.loglog()
    ax.grid()
    
    return m, c
