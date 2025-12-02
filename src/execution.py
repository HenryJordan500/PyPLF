from src.initialize import *
from src.simulate import *


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


RealSimulationRegion = SimulationRegion(dim_number=dim_number,
                                        lower_boundary=lower_boundary,
                                        upper_boundary=upper_boundary,
                                        boundary_condition=boundary_condition)
RealSimulationFlow = SimulationFlow(dim_number=dim_number,
                                    flow=flow,
                                    jacobian=jacobian,
                                    time_derivative=time_derivative)
RealSimulationParameters = SimulationParameters(dim_number=dim_number,
                                                num_particles=num_particles,
                                                time_step=time_step,
                                                total_simulation_time=total_simulation_time,
                                                beta=beta,
                                                st=st)

run_simulation(SimulationRegion=RealSimulationRegion,
               SimulationParameters=RealSimulationParameters,
               SimulationFlow=RealSimulationFlow,
               initial_particles=real_initial_particles,
               save_path=save_path)
