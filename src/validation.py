import numpy as np
import matplotlib as mpl
import h5py

def linear_flow(x):
    return 1

def validate_particle_flow(position_array,velocity_array,acceleration_array):
    # Calculate average position, velocity, and acceleration
    avgposition = []
    avgvelocity = []
    avgacceleration = []
    for i in range(position_array[1,:]):
        avgposition[i] = np.mean(position_array[:,i])
    for i in range(velocity_array[1,:]):
        avgvelocity[i] = np.mean(velocity_array[:,i])
    for i in range(acceleration_array[1,:]):
        avgacceleration[i] = np.mean(acceleration_array[:,i])
    
    
    return 1

def get_data(file_name):
    hf = h5py.File(file_name, 'r')
    hf.keys()
