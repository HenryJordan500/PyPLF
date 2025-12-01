import numpy as np

from src.initialize import *
from src.utils import *


### Initalization Testing ###

def intialization_flow(x, t):

    return x + t
        
def intialization_spatial_derivative(x, t):

    return 1 + t

def intialization_time_derivative(x, t):

    return x + 1


### Simulation Testing ###

def simtest_flow(x, t):

    return x + t
        
def simtest_spatial_derivative(x, t):

    return np.array([1]*len(x)) + t

def simtest_time_derivative(x, t):

    return x + 1

SimFlowClass = SimulationFlow(flow=simtest_flow,
                              spatial_derivative=simtest_spatial_derivative,
                              time_derivative=simtest_time_derivative)

simtest_num_particles = 3
simtest_time_step = 0.5
simtest_total_simulation_time = 1
simtest_beta = 1
simtest_st = 1

SimParamsClass = SimulationParameters(num_particles=simtest_num_particles,
                                      time_step=simtest_time_step,
                                      total_simulation_time=simtest_total_simulation_time,
                                      beta=simtest_beta,
                                      st=simtest_st)

simtest_lower_boundary  = -1
simtest_upper_boundary = 1

SimRegionClass = SimulationRegion(lower_boundary=simtest_lower_boundary,
                                  upper_boundary=simtest_upper_boundary,
                                  boundary_condition='open')

simtest_initial_particles = initalize_particles(SimulationRegion=SimRegionClass,
                                                num_particles=SimParamsClass.num_particles,
                                                distribution='uniform')

simtest_particles_acc, simtest_particles_pos_vel = generate_tracking_arrays(initial_particles=simtest_initial_particles,
                                                                            simulation_steps=SimParamsClass.num_steps,
                                                                            dim_number=1)

diff_eq_expected_ans = np.array([[0, 6.5],
                                 [0, 9],
                                 [0, 11.5]])

rk4_expected_ans = np.array([[0.38769531, 3.80859375],
                             [1.19791667, 5.06217448],
                             [2.00813802, 6.31575521]])

run_simulation_expected_acc_ans = np.array([[[-0.5], [1.907]],
                                            [[1.], [3.908]],
                                            [[2.5], [5.91]]])

run_simulation_expected_vel_ans = np.array([[0.0, -0.009766],
                                            [0.0, 0.7173],
                                            [0.0, 1.444]])

run_simulation_expected_pos_ans = np.array([[-0.5, -0.526 ],
                                            [ 0.0, 0.1562],
                                            [ 0.5, 0.8384]])

### Utils Testing ###

expected_acc_ans = np.array([[[0], [0]],
                             [[0], [0]],
                             [[0], [0]]])
        
expected_pos_vel_ans = np.array([[[-0.5, 0],[0, 0]],
                                 [[0, 0], [0, 0]],
                                 [[0.5, 0], [0, 0]]])
