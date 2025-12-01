import numpy as np
import h5py

def generate_tracking_arrays(initial_particles, simulation_steps, dim_number):

    # Generate arrays of correct size
    particle_pos_vel = np.zeros((initial_particles.size, simulation_steps, dim_number*2), dtype='float32')
    particle_acc = np.zeros((initial_particles.size, simulation_steps, dim_number), dtype='float32')

    # Function to set inital particle positions for different dimensions
    particle_pos_vel[:,0][:,0] = initial_particles

    return particle_acc, particle_pos_vel

def save_data(particle_acc, particle_pos_vel, save_path):

    fil = h5py.File(f'{save_path}.hdf5', 'w')
    acc = fil.create_dataset("acceleration", data=particle_acc, dtype='float16')
    vel = fil.create_dataset("velocity", data=particle_pos_vel[:, :, 1], dtype='float16')
    pos = fil.create_dataset("position", data=particle_pos_vel[:, :, 0], dtype='float16')
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
    
### Boundary Condition Functions ###

def open_bc(i1_particle_pos_vel):

    return i1_particle_pos_vel
