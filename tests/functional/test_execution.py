from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_simulation_pipeline_creates_hdf5(tmp_path: Path) -> None:
    """
    End-to-end functional test of the simulation pipeline (no animation).

    This test:
    - constructs a simple 2-D linear flow
    - sets up region, parameters, and initial particles
    - runs the simulation
    - asserts that the HDF5 file is created
    - checks that datasets have the expected shapes
    """
    from src.initialize import (
        SimulationRegion,
        SimulationFlow,
        SimulationParameters,
        initalize_particles,
    )
    from src.simulate import run_simulation
    from src.utils import create_flow_functions

    save_path = tmp_path / "output"

    # Simple 2-D flow: u = x, v = y
    flow_list = ["x", "y"]
    dim_number, flow, jacobian, time_derivative = create_flow_functions(
        flow_list=flow_list,
    )

    # Domain and boundary conditions
    lower_boundaries = [0.0, 0.0]
    upper_boundaries = [1.0, 1.0]
    boundary_conditions = ["periodic", "periodic"]

    region = SimulationRegion(
        dim_number=dim_number,
        lower_boundaries=lower_boundaries,
        upper_boundaries=upper_boundaries,
        boundary_conditions=boundary_conditions,
    )

    # Simulation parameters: very small test
    num_particles = 4
    time_step = 0.1
    total_simulation_time = 0.2
    st = 1.0
    beta = 1.0

    params = SimulationParameters(
        dim_number=dim_number,
        num_particles=num_particles,
        time_step=time_step,
        total_simulation_time=total_simulation_time,
        beta=beta,
        st=st,
    )

    flow_obj = SimulationFlow(
        dim_number=dim_number,
        flow=flow,
        jacobian=jacobian,
        time_derivative=time_derivative,
    )

    # Initialize particles uniformly
    initial_particles = initalize_particles(
        SimulationRegion=region,
        SimulationParameters=params,
        distribution="uniform",
    )

    # Sanity check: we requested 4, but initializer may adjust to a grid
    assert initial_particles.shape[1] == dim_number

    run_simulation(
        SimulationRegion=region,
        SimulationParameters=params,
        SimulationFlow=flow_obj,
        initial_particles=initial_particles,
        save_path=str(save_path),
    )

    hdf5_path = Path(f"{save_path}.hdf5")
    assert hdf5_path.exists(), "HDF5 output file was not created."

    with h5py.File(hdf5_path, "r") as f:
        acceleration = f["acceleration"][()]
        velocity = f["velocity"][()]
        position = f["position"][()]

    # Check that dimensions match the simulation setup.
    # First axis is number of particles; second is number of time steps.
    assert acceleration.ndim == 3
    assert velocity.ndim == 3
    assert position.ndim == 3

    # Check that positions and velocities/accelerations
    # share the same time axis.
    assert acceleration.shape[1] == velocity.shape[1] == position.shape[1]
    # Spatial dimension matches setup.
    assert acceleration.shape[2] == dim_number
    assert velocity.shape[2] == dim_number
    assert position.shape[2] == dim_number
