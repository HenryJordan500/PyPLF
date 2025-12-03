"""
Utility functions for creating flow fields, generating particle tracking arrays,
and saving/extracting simulation data.

Functions
---------
create_flow_functions(flow_list)
    Converts symbolic flow expressions into callable functions for evaluating
    the flow, its spatial Jacobian, and its time derivative.

generate_tracking_arrays(initial_particles, simulation_steps, dim_number)
    Allocates and initializes the position/velocity and acceleration arrays
    used to store particle data over all time steps.

save_data(particle_acc, particle_pos_vel, dim_number, save_path)
    Saves acceleration, velocity, and position arrays to an HDF5 file.

extract_data(save_path, extract)
    Loads one of the datasets ('acceleration', 'velocity', 'position') from the
    saved HDF5 file.
"""

import numpy as np
import h5py
import sympy as sp

def create_flow_functions(flow_list):
    """
    Create flow, Jacobian, and time-derivative functions from symbolic input.

    Parameters
    ----------
    flow_list : list[str]
        A list of strings representing each component of the flow field,
        written as SymPy-compatible expressions (e.g. ["x + t", "2*y"]).

    Returns
    -------
    tuple
        (dim_number, F, dFdx, dFdt) where:

        dim_number : int
            Number of spatial dimensions.
        F : callable
            Function returning the flow field at positions and time.
        dFdx : callable
            Function returning the spatial Jacobian of the flow.
        dFdt : callable
            Function returning the time derivative of the flow.
    """

    expr_list = sp.sympify(flow_list)

    dim_number = len(expr_list)

    # Set variables depeding on dimension number
    if dim_number == 1:
        var_symbols = sp.symbols('x t')
    elif dim_number == 2:
        var_symbols = sp.symbols('x y t')
    elif dim_number == 3:
        var_symbols = sp.symbols('x y z t')
    
    # Compute Jacbian derivatives and time derivatives of flow
    jac_matrix = sp.Matrix(expr_list).jacobian(var_symbols[:-1])
    dFdt_expr_list = [sp.diff(expr, var_symbols[-1]) for expr in expr_list]
    
    F_temp = sp.lambdify(var_symbols, expr_list, "numpy")
    J_temp = sp.lambdify(var_symbols, jac_matrix, "numpy")
    dFdt_temp = sp.lambdify(var_symbols, dFdt_expr_list, "numpy")

    # Define flow function
    def F(pos, t):

        if pos.ndim == 1:

            return np.array(F_temp(*pos, t), dtype='float32')

        # Enables vecorized computations with many particles
        cols = [pos[:, i] for i in range(dim_number)]
        out = F_temp(*cols, t)
        #print(out)
        #print(cols)
        out_array = np.column_stack(out).astype("float32")
        return out_array
    
    # Define time derivatives of flow
    def dFdt(pos, t):
        
        if pos.ndim == 1:

            return np.array(dFdt_temp(*pos, t), dtype="float32")

        cols = [pos[:, i] for i in range(dim_number)]
        out  = dFdt_temp(*cols, t)
        #print(out)
        #print(cols)
        out_array = np.column_stack(out).astype("float32")
        return out_array
    
    
    # Define Jacobian of flow
    def dFdx(pos, t):

        if pos.ndim == 1:
            
            return np.array(J_temp(*pos, t), dtype="float32")

        cols = [pos[:, i] for i in range(dim_number)]
        Jvals = J_temp(*cols, t)
        N = pos.shape[0]
        Jout = np.zeros((N, dim_number, dim_number), dtype="float32")
        for i in range(dim_number):
            for j in range(dim_number):
                Jout[:, i, j] = Jvals[i][j]
        
        return Jout

    return dim_number, F, dFdx, dFdt

def generate_tracking_arrays(initial_particles, simulation_steps, dim_number):
    """
    Initialize particle tracking arrays for positions/velocities and acceleration.

    Parameters
    ----------
    initial_particles : ndarray, shape (N, dim)
        Array containing initial positions of all particles.
    simulation_steps : int
        Total number of simulation time steps.
    dim_number : int
        Spatial dimensionality.

    Returns
    -------
    particle_acc : ndarray, shape (N, steps, dim)
        Acceleration array initialized to zeros.
    particle_pos_vel : ndarray, shape (N, steps, 2*dim)
        Position and velocity array where the first time slice contains initial
        positions and velocities initialized to zero.
    """

    # Generate arrays of correct size
    particle_pos_vel = np.zeros((len(initial_particles), simulation_steps, dim_number*2), dtype='float32')
    particle_acc = np.zeros((len(initial_particles), simulation_steps, dim_number), dtype='float32')
    
    # Function to set inital particle positions for different dimensions
    particle_pos_vel[:, 0][:, 0:dim_number] = initial_particles
    
    return particle_acc, particle_pos_vel

def save_data(particle_acc, particle_pos_vel, dim_number, save_path):
    """
    Save particle acceleration, velocity, and position data to an HDF5 file.

    Parameters
    ----------
    particle_acc : ndarray
        Acceleration array of shape (N, steps, dim).
    particle_pos_vel : ndarray
        Position/velocity array of shape (N, steps, 2*dim).
    dim_number : int
        Number of spatial dimensions.
    save_path : str
        Base filename (without extension). File saved as '<save_path>.hdf5'.
    """

    fil = h5py.File(f'{save_path}.hdf5', 'w')
    acc = fil.create_dataset("acceleration", data=particle_acc, dtype='float16')
    vel = fil.create_dataset("velocity", data=particle_pos_vel[:, :, dim_number:2*dim_number], dtype='float16')
    pos = fil.create_dataset("position", data=particle_pos_vel[:, :, 0:dim_number], dtype='float16')
    del acc, vel, pos
    fil.close()

    return

def extract_data(save_path, extract=None):
    """
    Load a specific dataset from a saved simulation file.

    Parameters
    ----------
    save_path : str
        Base filename of the HDF5 file (without extension).
    extract : {'acceleration', 'velocity', 'position'}
        The dataset to retrieve.

    Returns
    -------
    ndarray
        The requested dataset.

    Raises
    ------
    KeyError
        If `extract` is None or not one of the valid dataset names.
    """

    if extract is None:
        
        raise KeyError('Extract key must be set to either acceleration, velocity or position')
    
    fil = h5py.File(f'{save_path}.hdf5', 'r')

    if extract == 'acceleration':

        return fil['acceleration'][:]
    
    if extract == 'velocity':

        return fil['velocity'][:]
    
    if extract == 'position':

        return fil['position'][:]


