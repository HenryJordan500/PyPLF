"""
Command-line execution script for running a complete PyPLF simulation.

This module reads simulation settings from a YAML configuration file, constructs
the simulation objects (region, parameters, flow, and initial particles), runs
the particle simulation, and generates an animation of particle trajectories.

Typical usage
-------------
Run from the command line:

    python -m src.execution --config config.yml

Expected configuration fields
-----------------------------
save_path : str
    Base filename (without extension) for saving HDF5 data and the MP4 animation.
flow : list[str]
    SymPy-compatible expressions defining each component of the flow field.
lower_boundaries : list[float]
upper_boundaries : list[float]
    Domain boundaries for each spatial dimension.
boundary_conditions : list[str]
    Boundary condition for each dimension ('periodic' or 'open').
num_particles : int
time_step : float
total_simulation_time : float
    Controls for simulation length and resolution.
st : float
beta : float
    Dimensionless particle parameters used in the ODE.
"""

from src.initialize import *
from src.simulate import *
from src.animate import *


import argparse
import yaml

# Step 1: Parse command-line arguments
# ------------------------------------
# This section reads the `--config` argument pointing to a YAML configuration
# file that contains all simulation parameters.

parser = argparse.ArgumentParser(description='Execute code to conduct simulation',
                                 prog='Conduct simulation')

parser.add_argument('--config',
                    type=str,
                    help='Input the config file for the simulation',
                    required=True,
                    default='config.yml')

args = parser.parse_args()

# Step 2: Load configuration
# --------------------------
# The YAML file is expected to define simulation domain boundaries, flow
# expressions, particle parameters, and output paths. Keys are validated only
# implicitly; missing keys cause normal Python KeyError exceptions.

f = open(file=args.config, mode='r')
config = yaml.load(f, Loader=yaml.FullLoader)
f.close()

# Step 3: Construct simulation components
# ---------------------------------------
# From the configuration dictionary we create:
# - SimulationRegion  (domain + boundaries)
# - SimulationFlow    (flow, Jacobian, and time derivative)
# - SimulationParameters (time stepping and particle constants)
# - initial particle positions (via `initalize_particles`)

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

# Step 4: Execute the simulation
# ------------------------------
# Runs the full RK4 particle integration loop and saves acceleration,
# velocity, and position data to an HDF5 file at <save_path>.hdf5.

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

# Step 5: Generate animation
# --------------------------
# Creates an MP4 animation of particle trajectories using the saved data.

make_animation(save_path=save_path,
               SimulationRegion=RunSimulationRegion,
               SimulationParameters=RunSimulationParameters)
