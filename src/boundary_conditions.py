import numpy as np

def open_bc(SimulationRegion, i1_particle_pos_vel, k):

    return i1_particle_pos_vel

def periodic_bc(SimulationRegion, i1_particle_pos_vel, k):

    lower_boundary = SimulationRegion.lower_boundaries[k]
    upper_boundary = SimulationRegion.upper_boundaries[k]

    L = upper_boundary - lower_boundary

    i1_particle_pos_vel[:, k] = np.where(i1_particle_pos_vel[:,k] > lower_boundary, i1_particle_pos_vel[:,k], i1_particle_pos_vel[:,k] + L)
    i1_particle_pos_vel[:, k] = np.where(i1_particle_pos_vel[:,k] < upper_boundary, i1_particle_pos_vel[:,k], i1_particle_pos_vel[:,k] - L)

    return i1_particle_pos_vel