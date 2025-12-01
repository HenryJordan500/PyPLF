from src.initialize import *
from src.utils import *

import numpy as np

def material_derivative(SimulationFlow, i_particle_pos_vel, time):

    position = i_particle_pos_vel[:, 0]

    flow = SimulationFlow.flow(position, time)
    flow_spatial_deriv = SimulationFlow.spatial_derivative(position, time)
    flow_time_deriv = SimulationFlow.time_derivative(position, time)

    # Take the inner product of the flow and it's spatial derivative
    inner_product = flow * flow_spatial_deriv

    return flow_time_deriv + inner_product

def diff_eq(SimulationParameters, SimulationFlow, i_particle_pos_vel, time):

    position = i_particle_pos_vel[:,0]
    velocity = i_particle_pos_vel[:,1]

    result = np.zeros((SimulationParameters.num_particles, 2), dtype='float32')

    st = SimulationParameters.st
    beta = SimulationParameters.beta

    material_derivative_result = material_derivative(SimulationFlow=SimulationFlow,
                                                     i_particle_pos_vel=i_particle_pos_vel,
                                                     time=time)
    
    # Compute result of coupled 1st order ODEs
    result[:,0] = velocity
    result[:,1] = beta*material_derivative_result + (1/st) * (SimulationFlow.flow(position, time) - velocity)

    return result

def RK4_step(SimulationParameters, SimulationFlow, i_particle_pos_vel, time):

    # Coefficents for 4th order Runge-Kutta Method
    A = np.array([0,1/2,1/2,1])
    CH = np.array([1/6,1/3,1/3,1/6])
    B = np.array([0,1/2,1/2,1])

    time_step = SimulationParameters.time_step

    # Compute terms in 4th order Runge-Kutta Method
    k1 = diff_eq(SimulationParameters, SimulationFlow, i_particle_pos_vel, time + A[0] * time_step)
    k2 = diff_eq(SimulationParameters, SimulationFlow, i_particle_pos_vel + B[1] * k1 * time_step, time + A[1] * time_step)
    k3 = diff_eq(SimulationParameters, SimulationFlow, i_particle_pos_vel + B[2] * k2 * time_step, time + A[2] * time_step)
    k4 = diff_eq(SimulationParameters, SimulationFlow, i_particle_pos_vel + B[3] * k3 * time_step, time + A[3] * time_step)

    # Calculate updated position and velocity
    result = i_particle_pos_vel + (CH[0] * k1 + CH[1] * k2 + CH[2] * k3 + CH[3] * k4) * time_step

    return result


def run_simulation(SimulationRegion, SimulationParameters, SimulationFlow, initial_particles, save_path):

    dim_number = 1

    # Extract parameters from classes
    num_steps = SimulationParameters.num_steps

    # Create particle tracking arrays
    particle_acc, particle_pos_vel = generate_tracking_arrays(initial_particles=initial_particles,
                                                              simulation_steps=num_steps,
                                                              dim_number=dim_number)
    
    # Set function to do boundary condition checks
    if SimulationRegion.boundary_condition == 'open':

        boundary_condition_function = open_bc

    # Solve the ODE with proper boundary conditions
    for i in range(num_steps):

        # Extract time for current step
        time = SimulationParameters.current_time(i)
        i_particle_pos_vel = particle_pos_vel[:, i]

        # Compute particle acceleration for current time step
        particle_acc[:,i][:,0] = diff_eq(SimulationParameters=SimulationParameters,
                                         SimulationFlow=SimulationFlow,
                                         i_particle_pos_vel=i_particle_pos_vel,
                                         time=time)[:,1]
        
        if i < num_steps - 1:
            
            # Compute particle position and velocity at next time step
            particle_pos_vel[:, i+1] = RK4_step(SimulationParameters=SimulationParameters,
                                                SimulationFlow=SimulationFlow,
                                                i_particle_pos_vel=i_particle_pos_vel,
                                                time=time)
            
            # Apply boundary condition
            particle_pos_vel[:, i+1] = boundary_condition_function(i1_particle_pos_vel=particle_pos_vel[:, i+1])
                
        else:

            break
    
    save_data(particle_acc=particle_acc,
              particle_pos_vel=particle_pos_vel,
              save_path=save_path)
    
    return
    