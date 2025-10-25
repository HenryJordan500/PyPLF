from initialize import *

import numpy as np
import h5py

def material_derivative(SimulationFlow, particle_stats):

    flow = SimulationFlow.flow(particle_stats[:,0], time)
    flow_spatial_deriv = SimulationFlow.flow_spaital_deriv(particle_stats[:,0], time)
    flow_time_deriv = SimulationFlow.flow_time_deriv(particle_stats[:,0], time)

    inner_product = np.sum(flow*flow_spatial_deriv)

    return flow_time_deriv + inner_product

def diff_eq(SimulationParameters, SimulationFlow, particle_stats):

    position = particle_stats[:,0]
    velocity = particle_stats[:,1]
    accceleration = particle_stats[:,2]

    st = SimulationParameters.st
    beta = SimulationParameters.beta

    return beta*material_derivative + (1/st) * (SimulationFlow.flow(position, time) - velocity)

def RK4_step(SimulationParameters, SimulationFlow, particles_stats):

    A = np.array([0,1/2,1/2,1])
    CH = np.array([1/6,1/3,1/3,1/6])
    B = np.array([0,1/2,1/2,1])

    k1 = diff_eq(SimulationParameters, SimulationFlow, posvosi, ti + A[0]*delt,st,fr,beta,ka,kd,Grav,FullNumParts,bottom)
    k2 = diff_eq(SimulationParameters, SimulationFlow,posvosi + B[1]*k1*delt, ti + A[1]*delt,st,fr,beta,ka,kd,Grav,FullNumParts,bottom)
    k3 = diff_eq(SimulationParameters, SimulationFlow,posvosi + B[2]*k2*delt, ti + A[2]*delt,st,fr,beta,ka,kd,Grav,FullNumParts,bottom)
    k4 = diff_eq(SimulationParameters, SimulationFlow, posvosi + B[3]*k3*delt, ti + A[3]*delt,st,fr,beta,ka,kd,Grav,FullNumParts,bottom)


def run_simulation(SimulationRegion, SimulaitonParameters, SimulationFlow, inital_particles, save_path):

    simulation_steps = SimulaitonParameters.steps
    simulation_time = SimulaitonParameters.time

    left_boundary = SimulationRegion.left_boundary
    right_boundary= SimulationRegion.right_boundary

    st = SimulationParameters.st
    beta = SimulationParameters.beta

    particle_stats = np.zeros((inital_particles.size, simulation_steps, 3), dtype='float32')
    particle_stats[:,0][:,0] = inital_particles

    for i in range(simulation_steps):

        particle_stats[:,i][:,2] = diff_eq()

        if i < simulation_steps - 1:
            
            particle_stats[:,i+1][0:2] = RK_step()

            # Implement BC
    
    fil = h5py.File(f'{save_path}.hdf5', 'w')
    acc = fil.create_dataset("acceleration", data=particle_stats[3], dtype='float16')
    vos = fil.create_dataset("velocity", data=vos[2], dtype='float16')
    pos = fil.create_dataset("position", data=pos[1], dtype='float16')
    del acc,vos,pos
    fil.close()






# RK4 step function

# ODE solver function

# Function to compute Du/Dt
# Subfunctions

# Function to compute Vorticity
# Subfunctions

# Function to compute u-xdot

# Function to save data as .h5py