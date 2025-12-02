import numpy as np
import h5py
import sympy as sp

def create_flow_functions(flow_list):

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

    # Generate arrays of correct size
    particle_pos_vel = np.zeros((len(initial_particles), simulation_steps, dim_number*2), dtype='float32')
    particle_acc = np.zeros((len(initial_particles), simulation_steps, dim_number), dtype='float32')
    
    # Function to set inital particle positions for different dimensions
    particle_pos_vel[:, 0][:, 0:dim_number] = initial_particles
    
    return particle_acc, particle_pos_vel

def save_data(particle_acc, particle_pos_vel, dim_number, save_path):

    fil = h5py.File(f'{save_path}.hdf5', 'w')
    acc = fil.create_dataset("acceleration", data=particle_acc, dtype='float16')
    vel = fil.create_dataset("velocity", data=particle_pos_vel[:, :, dim_number:2*dim_number], dtype='float16')
    pos = fil.create_dataset("position", data=particle_pos_vel[:, :, 0:dim_number], dtype='float16')
    del acc, vel, pos
    fil.close()

    return

def extract_data(save_path, extract=None):

    if extract is None:
        
        raise KeyError('Extract key must be set to either acceleration, velocity or position')
    
    fil = h5py.File(f'{save_path}.hdf5', 'r')

    if extract == 'acceleration':

        return fil['acceleration'][:]
    
    if extract == 'velocity':

        return fil['velocity'][:]
    
    if extract == 'position':

        return fil['position'][:]


