import numpy as np

from src.initialize import *
from src.utils import *


# Initalization Testing

def intialization_flow(x, t):

    return x + t


def intialization_spatial_derivative(x, t):

    return 1


def intialization_time_derivative(x, t):

    return 1


# Simulation Testing

simtest_flow_list_1d = ['x + t']
_, simtest_flow_1d, simtest_jacobian_1d, simtest_dFlowdt_1d = (
    create_flow_functions(flow_list=simtest_flow_list_1d)
)

simtest_flow_list_2d = ['x + t', 'x + t']
_, simtest_flow_2d, simtest_jacobian_2d, simtest_dFlowdt_2d = (
    create_flow_functions(flow_list=simtest_flow_list_2d)
)

simtest_flow_list_3d = ['x + t', 'x + t', 'x + t']
_, simtest_flow_3d, simtest_jacobian_3d, simtest_dFlowdt_3d = (
    create_flow_functions(flow_list=simtest_flow_list_3d)
)


SimFlowClass1d = SimulationFlow(
    dim_number=1,
    flow=simtest_flow_1d,
    jacobian=simtest_jacobian_1d,
    time_derivative=simtest_dFlowdt_1d)

simtest_num_particles = 3
simtest_time_step = 0.5
simtest_total_simulation_time = 1
simtest_beta = 1
simtest_st = 1

SimParamsClass1d = SimulationParameters(
    dim_number=1,
    num_particles=simtest_num_particles,
    time_step=simtest_time_step,
    total_simulation_time=simtest_total_simulation_time,
    beta=simtest_beta,
    st=simtest_st)

simtest_lower_boundaries = [-1, -1, -1]
simtest_upper_boundaries = [1, 1, 1]

SimRegionClass1d = SimulationRegion(
    dim_number=1,
    lower_boundaries=[simtest_lower_boundaries[0]],
    upper_boundaries=[simtest_upper_boundaries[0]],
    boundary_conditions=['open'])

simtest_initial_particles_1d = initalize_particles(
    SimulationRegion=SimRegionClass1d,
    SimulationParameters=SimParamsClass1d,
    distribution='uniform')

simtest_particles_acc_1d, simtest_particles_pos_vel_1d = (
    generate_tracking_arrays(
        initial_particles=simtest_initial_particles_1d,
        simulation_steps=SimParamsClass1d.num_steps,
        dim_number=1)
)

diff_eq_expected_ans_1d = np.array([
    [0, 4],
    [0, 5],
    [0, 6]])

rk4_expected_ans_1d = np.array([
    [-0.015625, 1.921875],
    [0.596354, 2.346354],
    [1.208333, 2.770833]])

run_simulation_expected_acc_ans_1d = np.array([
    [[0], [1.849]],
    [[1.], [2.648]],
    [[2], [3.447]]])

run_simulation_expected_vel_ans_1d = np.array([
    [[0.0], [0.224]],
    [[0.0], [0.6484]],
    [[0.0], [1.073]]])

run_simulation_expected_pos_ans_1d = np.array([
    [[-0.5], [-0.4636]],
    [[0.0], [0.1484]],
    [[0.5], [0.7603]]])

SimFlowClass2d = SimulationFlow(
    dim_number=2,
    flow=simtest_flow_2d,
    jacobian=simtest_jacobian_2d,
    time_derivative=simtest_dFlowdt_2d)

SimFlowClass3d = SimulationFlow(
    dim_number=3,
    flow=simtest_flow_3d,
    jacobian=simtest_jacobian_3d,
    time_derivative=simtest_dFlowdt_3d)

SimParamsClass2d = SimulationParameters(
    dim_number=2,
    num_particles=simtest_num_particles,
    time_step=simtest_time_step,
    total_simulation_time=simtest_total_simulation_time,
    beta=simtest_beta,
    st=simtest_st)

SimParamsClass3d = SimulationParameters(
    dim_number=3,
    num_particles=simtest_num_particles,
    time_step=simtest_time_step,
    total_simulation_time=simtest_total_simulation_time,
    beta=simtest_beta,
    st=simtest_st)

SimRegionClass2d = SimulationRegion(
    dim_number=2,
    lower_boundaries=simtest_lower_boundaries[0:2],
    upper_boundaries=simtest_upper_boundaries[0:2],
    boundary_conditions=['open', 'open'])

SimRegionClass3d = SimulationRegion(
    dim_number=3,
    lower_boundaries=simtest_lower_boundaries,
    upper_boundaries=simtest_upper_boundaries,
    boundary_conditions=['open', 'open', 'open'])

simtest_initial_particles_2d = initalize_particles(
    SimulationRegion=SimRegionClass2d,
    SimulationParameters=SimParamsClass2d,
    distribution='uniform')

simtest_particles_acc_2d, simtest_particles_pos_vel_2d = (
    generate_tracking_arrays(
        initial_particles=simtest_initial_particles_2d,
        simulation_steps=SimParamsClass2d.num_steps,
        dim_number=2)
)

simtest_initial_particles_3d = initalize_particles(
    SimulationRegion=SimRegionClass3d,
    SimulationParameters=SimParamsClass3d,
    distribution='uniform')

simtest_particles_acc_3d, simtest_particles_pos_vel_3d = (
    generate_tracking_arrays(
        initial_particles=simtest_initial_particles_3d,
        simulation_steps=SimParamsClass3d.num_steps,
        dim_number=3)
)

# Utils Testing

expected_acc_ans = np.array([[[0], [0]],
                             [[0], [0]],
                             [[0], [0]]])

expected_pos_vel_ans = np.array([[[-0.5, 0], [0, 0]],
                                 [[0, 0], [0, 0]],
                                 [[0.5, 0], [0, 0]]])

expected_2d_initial_particles = np.array([[-0.33, -0.33],
                                          [0.33, -0.33],
                                          [-0.33, 0.33],
                                          [0.33, 0.33]])

expected_3d_initial_particles = np.array([[-0.33, -0.33, -0.33],
                                          [-0.33, -0.33, 0.33],
                                          [-0.33, 0.33, -0.33],
                                          [-0.33, 0.33, 0.33],
                                          [0.33, -0.33, -0.33],
                                          [0.33, -0.33, 0.33],
                                          [0.33, 0.33, -0.33],
                                          [0.33, 0.33, 0.33]])
