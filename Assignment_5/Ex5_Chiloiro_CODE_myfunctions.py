import numpy as np
from scipy.integrate import simpson
from scipy.special import hermite
from math import isclose
from typing import Callable, Optional, List
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def eigenstate(x:np.ndarray, n:int, m:float=1., w:float=1.) -> np.ndarray:
    """
    Compute the n-th eigenstate value in the position x of a time-independent harmonic oscillator.
    
    Parameters
    ----------
    x : np.ndarray(dtype=float)
        spatial position, argument of the eigenfunction.
    n : int
        Quantum number.
    m : float, default=1.
    particle mass
    w : float, default=1.
        angular frequency of the oscillator

    Returns
    -------
    psi_n(x) : np.ndarray(dtype=float)
        n-th eigenstate value in the position x.
    """
    # Check input types
    if not np.issubdtype(x.dtype, np.float64):
        raise TypeError('Array type for `x` must be float.')
    if not isinstance(n, int):
        raise TypeError('`n` must be an integer.')
    if n < 0:
        raise ValueError('`n` must be greater or equal to 0.')
    if not all(val > 0 for val in [m, w]):
        raise ValueError('`m` and `w` must be positive values.')
    
    # function
    norm_const = 1./np.sqrt(2.**n*np.math.factorial(n)) * (m*w/(np.pi))**(1/4)
    gaussian = np.exp(-m*w*x**2/(2))
    H_n = hermite(n)
    
    return norm_const * gaussian * H_n(np.sqrt(m*w)*x)



def td_harmonic_potential(x_vec:np.ndarray, t:float, T:float=1., m:float=1., w:float=1.) -> np.ndarray:
    """
    Compute an harmonic time-dependent potential in the positions x_vec at time t.
    
    Parameters
    ----------
    x : np.ndarray(dtype=float)
        Positions.
    t : float
        Time.
    m : float, default=1.
        Particle mass.
    w : float, default=1.
        Angular frequency of the oscillator.

    Returns
    -------
    V : np.ndarray(dtype=float)
        Array containing the computed potential values given the positions.
    """
   # Check inputs
    if not np.issubdtype(x_vec.dtype, np.float64):
        raise TypeError('Array type for `x_vec` must be float.')
    if not isinstance(t, float):
        raise TypeError('Type of `t` must be float.')
    if t < 0:
        raise ValueError('Time `t` must be greater or equal to 0.')
    if not all(val > 0 for val in [T, m, w]):
        raise ValueError('T, m, and w must be positive values.')
    
    # function
    if t<=T:
        V = m*w**2*(x_vec-t/T)**2/(2.*m)
    else:
        V = m*w**2*(x_vec-1)**2/(2.*m)
    return V



def check_normalization(state:np.ndarray, x_vec:np.ndarray, debug:bool=True) -> None:
    """
    Check the normalization (by using Simpson method) of a given quantum state.
    
    Parameters
    ----------
    state : np.ndarray(dtype=complex)
        One dimensional discretized state.
    x_vec : np.ndarray(dtype=float)
        Discretized positions.
    debug : bool, default=True
        If True, print the output messages.
        
    Returns
    -------
    None
    """
    # Check inputs
    if not np.issubdtype(state.dtype, np.complex128):
        raise TypeError('Array type for `state` must be double-precision complex.')
    if not np.issubdtype(x_vec.dtype, np.float64):
        raise TypeError('Array type for `x_vec` must be float.')
    if not isinstance(debug, bool):
        raise TypeError('Debug must be a boolean value.')
    
    # subroutine
    ampl = np.abs(state)**2
    norm = simpson(ampl, x=x_vec, even='first')
    if debug:
        if isclose(norm, 1., rel_tol=1e-04):
            print('Normalized')
        else:
            print('Not normalized. The integral value is: ', norm)
    return None



def compute_energy(state:np.ndarray, x_vec:np.ndarray, L:float, potential_vec:np.ndarray) -> List[float]:
    """
    Compute the expected value of energy of a given quantum state.
    
    Parameters
    ----------
    state : np.ndarray(dtype=complex)
        One dimensional discretized state.
    x_vec : np.ndarray(dtype=float)
        Discretized positions.
    L : float
        Space interval length
    potential_vec : np.ndarray(dtype=float)
        Array (of the same size of state and x_vec) containing the potentiale values at the positions x_vec.

    Returns
    -------
    [energy_k.real, energy_p, energy_final] : list
        List containing kinetic, potential and total energy of the state.
    """
    # Check inputs
    if not np.issubdtype(state.dtype, np.complex128):
        raise TypeError('Array type for `state` must be double-precision complex.')
    if not np.issubdtype(x_vec.dtype, np.float64):
        raise TypeError('Array type for `x_vec` must be float.')
    if not isinstance(L, (int, float)):
        raise TypeError('L type must be integer or float')
    if L <= 0:
        raise ValueError('Space interval length must be strictly positive.')
    if (state.shape[0] != x_vec.shape[0]) :
        raise ValueError('State and positions shape must be equal.')    
    if (potential_vec.shape[0] != x_vec.shape[0]) :
        raise ValueError('Potential and positions shape must be equal.')

    # Function
    # Momentum discretization
    Ns = state.shape[0]
    dp = 2*np.pi/L
    p_discr = np.concatenate((np.arange(0, Ns/2), np.arange(-Ns/2, 0)))*dp
    # Creating real, momentum, and conjugate wavefunctions.
    state_k = np.fft.fft(state)
    state_c = np.conj(state)
    # Finding the momentum and real-space energy terms
    energy_k = simpson(0.5*state_c*np.fft.ifft((p_discr**2)*state_k), x=x_vec, even='first').real             
    energy_p = simpson(state_c*potential_vec*state, x=x_vec, even='first').real
    # Integrating over all space
    energy_final = energy_k + energy_p
    return [energy_k.real, energy_p, energy_final]



def average_position(state:np.ndarray, x_vec:np.ndarray) -> float:
    """
    Compute the expected value of position of a given quantum state.
    
    Parameters
    ----------
    state : np.ndarray(dtype=complex)
        One dimensional discretized state.
    x_vec : np.ndarray(dtype=float)
        Discretized positions.

    Returns
    -------
    avg_x, std_x : float
        Average position and corrispective standard deviation.
    """
    # check input types
    if not np.issubdtype(state.dtype, np.complex128):
        raise TypeError('Array type for `state` must be double-precision complex.')
    if not np.issubdtype(x_vec.dtype, np.float64):
        raise TypeError('Array type for `x_vec` must be float.')

    # average
    f_x = np.conj(state)*x_vec*state
    avg_x = simpson(f_x, x=x_vec, even='first').real
    # std
    f_x_square = np.conj(state)*x_vec**2*state
    avg_x_square = simpson(f_x_square, x=x_vec, even='first').real
    std_x = np.sqrt(avg_x_square - avg_x**2)
    
    return avg_x, std_x



def split_step_method(initial_state: np.ndarray,
                      space_int: List[float],
                      numslice_space: int,
                      time_int: List[float],
                      numslice_time: int,
                      potential: Callable[[float, float], float], 
                      particle_mass: float = 1.0,
                      num_frms: Optional[int] = None,
                      **kwargs) -> np.ndarray:
    """
    Perform the split-step method for solving a quantum system.

    Parameters
    ----------
    initial_state : np.ndarray
        NumPy array representing the initial state of the system.
    space_int : List[float]
        List representing the spatial interval.
    numslice_space : int
        Integer representing the number of slices in the spatial dimension.
    time_int : List[float]
        List representing the time interval.
    numslice_time : int
        Integer representing the number of slices in the time dimension.
    potential : Callable[[float, float], float]
        A function representing the potential. Takes two float arguments (position and time).
    particle_mass : float, optional
        A float representing the particle mass with a default value of 1.0.
    num_frms : int, optional
        An optional parameter, possibly an integer.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        The result of the split-step method.
    """
    # Check initial_state
    if not isinstance(initial_state, np.ndarray):
        raise TypeError('initial_state must be a NumPy array.')
    # Check space_int
    if not isinstance(space_int, list) or not all(isinstance(val, float) for val in space_int):
        raise TypeError('space_int must be a list of floats.')
    # Check numslice_space
    if not isinstance(numslice_space, int):
        raise TypeError('numslice_space must be an integer.')
    # Check time_int
    if not isinstance(time_int, list) or not all(isinstance(val, float) for val in time_int):
        raise TypeError('time_int must be a list of floats.')
    # Check numslice_time
    if not isinstance(numslice_time, int):
        raise TypeError('numslice_time must be an integer.')
    # Check potential
    if not callable(potential):
        raise TypeError('potential must be a callable function.')
    # Check particle_mass
    if not isinstance(particle_mass, float):
        raise TypeError('particle_mass must be a float.')
    # Check num_frms
    if num_frms is not None and not isinstance(num_frms, int):
        raise TypeError('num_frms must be an integer or None.')
        
    # Subroutine
    # Construct time and space discretization
    # Space
    xmin, xmax = space_int[0], space_int[1]
    Ns = numslice_space
    x_discr = np.linspace(xmin, xmax, Ns)
    # Time
    tmin, tmax = time_int[0], time_int[1]
    Nt = numslice_time
    t_discr = np.linspace(0, tmax, Nt)
    dt = t_discr[1]-t_discr[0]
    
    # Compute the kinetic part in the momentum space (with its own discretization)
    # Discretization
    L = xmax-xmin
    dp = 2*np.pi/L
    p_discr = np.concatenate((np.arange(0, Ns/2), np.arange(-Ns/2, 0)))*dp #np.linspace(0, 2*np.pi/L, Ns)
    # Kinetic
    K = p_discr**2/(2.*particle_mass)+1j*0
    
    # Method
    # (by default, python uses double-precision float)
    if num_frms is not None:
        states = np.zeros((Ns,  num_frms+1), dtype=np.complex128) 
        potential_for_plot = np.zeros((Ns,  num_frms+1))
    else:
        states = np.zeros((Ns,  Nt), dtype=np.complex128) 
    state_t = initial_state
    jj = 0
    for i, t in enumerate(t_discr):
        
        # Potential
        V_t = potential(x_discr, t, m=particle_mass, **kwargs)+1j*0
        
        # Apply potential
        state_t *= np.exp(-1j*V_t*dt/2.)
        check_normalization(state_t, x_discr, debug=False)
        
        # Fourier transform
        state_t = np.fft.fft(state_t)
        check_normalization(state_t, p_discr, debug=False)
        
        # Apply kinetic
        state_t *= np.exp(-1j*K*dt)
        check_normalization(state_t, p_discr, debug=False)
        
        # Inverse Fourier transform
        state_t = np.fft.ifft(state_t)
        check_normalization(state_t, x_discr, debug=False)
        
        # Apply potential
        state_t *= np.exp(-1j*V_t*dt/2.)
        check_normalization(state_t, x_discr, debug=False)
        
        # Save results
        if num_frms is not None:
            if i % (Nt //  num_frms) == 0:
                states[:, jj] = state_t
                potential_for_plot[:, jj] = np.real(V_t)
                jj += 1
        else:
            states[:, i] = state_t
        
    if num_frms is not None:
        return states, potential_for_plot
    else:
        return states

    

####################################################
######  Subroutines for plots and animations  ######
####################################################

def init_line():
    line.set_data([], [])
    return line,

def animate(i, states, custom_amplitudes, custom_potentials, x):
    Ns = custom_amplitudes.shape[0]
    y = custom_amplitudes[0:Ns, i]
    y1 = custom_potentials[0:Ns, i]
    line.set_data(x, y)
    line1.set_data(x, y1)
    L = max(x)-min(x)
    en = compute_energy(states[0:Ns, i], x, L, custom_potentials[0:Ns, i])
    text.set_text(f'E_tot = {en[2]:.2f}')
    text1.set_text(f'E_K = {en[0]:.2f}')
    text2.set_text(f'E_P = {en[1]:.2f}')
    return line,

def create_and_save_animation(states, custom_amplitudes, custom_potentials, x_values, num_frames, filename='slow.gif'):
    global line, line1, text, text1, text2  # Declare as global to avoid local variable error
    
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set(xlim=(-3, 3), ylim=(0, 1.5))
    line, = ax.plot([], [], label="Amplitude")
    line1, = ax.plot([], [], label="Potential")
    ax.plot([], [])  
    text = ax.text(2, 1.0, '', fontsize=12, ha='right')
    text1 = ax.text(2, 1.1, '', fontsize=12, ha='right')
    text2 = ax.text(2, 1.2, '', fontsize=12, ha='right')
    ax.legend(fontsize=15)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('f(x)', fontsize=15)
    anim = animation.FuncAnimation(
        fig, animate, fargs=(states, custom_amplitudes, custom_potentials, x_values), init_func=init_line,
        frames=num_frames, interval=20, blit=True
    )
    writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(filename, writer=writer)
    plt.show()
    
def plot_average_position(states, time_steps, x_values):
    """
    Plot the average position and error over time.

    Parameters
    ----------
    states : np.ndarray
        Two-dimensional array representing the quantum states over time.
    time_steps : np.ndarray
        One-dimensional array representing the time steps.
    x_values : np.ndarray
        One-dimensional array representing the spatial positions.

    Returns
    -------
    None
    """
    Nt = len(time_steps)
    x_avg, x_std = np.zeros(Nt), np.zeros(Nt)

    for jj in range(Nt):
        x_avg[jj], x_std[jj] = average_position(states[:, jj], x_values)

    _, ax = plt.subplots(figsize=(10,7))
    ax.plot(time_steps, x_avg, color='blue', label='Average')
    ax.fill_between(time_steps, x_avg - x_std, x_avg + x_std, color='lightblue', alpha=0.5, label='Std')
    ax.set_xlabel('t', fontsize=15)
    ax.set_ylabel('x', fontsize = 15)
    ax.legend(loc='upper left',fontsize = 15)
    plt.plot()    
