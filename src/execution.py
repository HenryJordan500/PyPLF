from src.initialize import *
from src.simulate import *
from src.animate import *


import argparse
import yaml


parser = argparse.ArgumentParser(description='Execute code to conduct simulation',
                                 prog='Conduct simulation')

parser.add_argument('--config',
                    type=str,
                    help='Input the config file for the simulation',
                    required=True,
                    default='config.yml')

args = parser.parse_args()

f = open(file=args.config, mode='r')
config = yaml.load(f, Loader=yaml.FullLoader)
f.close()

save_path = config['save_path']

num_particles = config['num_particles']
time_step = config['time_step']
total_simulation_time = config['total_simulation_time']

st = config['st']
beta = config['beta']

lower_boundary = config['lower_boundaries']
upper_boundary = config['upper_boundaries']
boundary_condition = config['boundary_conditions']

flow_list = config['flow']
dim_number, flow, jacobian, time_derivative = create_flow_functions(flow_list=flow_list)


RunSimulationRegion = SimulationRegion(dim_number=dim_number,
                                       lower_boundaries=lower_boundary,
                                       upper_boundaries=upper_boundary,
                                       boundary_conditions=boundary_condition)
RunSimulationFlow = SimulationFlow(dim_number=dim_number,
                                   flow=flow,
                                   jacobian=jacobian,
                                   time_derivative=time_derivative)
RunSimulationParameters = SimulationParameters(dim_number=dim_number,
                                               num_particles=num_particles,
                                               time_step=time_step,
                                               total_simulation_time=total_simulation_time,
                                               beta=beta,
                                               st=st)
run_initial_particles = initalize_particles(SimulationRegion=RunSimulationRegion,
                                           SimulationParameters=RunSimulationParameters,
                                           distribution='uniform')

run_simulation(SimulationRegion=RunSimulationRegion,
               SimulationParameters=RunSimulationParameters,
               SimulationFlow=RunSimulationFlow,
               initial_particles=run_initial_particles,
               save_path=save_path)

make_animation(save_path=save_path,
               SimulationRegion=RunSimulationRegion,
               SimulationParameters=RunSimulationParameters)
