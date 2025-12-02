import numpy as np

def open_bc(i1_particle_pos_vel):

    return i1_particle_pos_vel

def periodic_bc(SimulationRegion, i1_particle_pos_vel):

    lower_boundary = SimulationRegion.lower_boundaries[0]
    upper_boundary = SimulationRegion.upper_boundaries[0]

    L = upper_boundary - lower_boundary

    i1_particle_pos_vel[:, 0] = np.where(i1_particle_pos_vel[:,0] > lower_boundary, i1_particle_pos_vel[:,0], i1_particle_pos_vel[:,0] + L)
    i1_particle_pos_vel[:, 0] = np.where(i1_particle_pos_vel[:,0] < upper_boundary, i1_particle_pos_vel[:,0], i1_particle_pos_vel[:,0] - L)

    return i1_particle_pos_vel