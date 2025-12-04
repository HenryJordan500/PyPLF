"""
Initialization tools for simulation domains, flow definitions, simulation
parameters, and particle initial conditions.
"""

import numpy as np


class SimulationRegion():
    """
    Represents the spatial domain in which particles move.

    Parameters
    ----------
    dim_number : int
        Number of spatial dimensions.
    lower_boundaries : list[float]
        Lower domain boundary for each dimension.
    upper_boundaries : list[float]
        Upper domain boundary for each dimension.
    boundary_conditions : list[str]
        Boundary condition per dimension ('periodic' or 'open').

    Raises
    ------
    ValueError
        If any list does not match `dim_number` in length.
    """

    def __init__(self,
                 dim_number,
                 lower_boundaries,
                 upper_boundaries,
                 boundary_conditions):

        self.dim_number = dim_number
        self.lower_boundaries = lower_boundaries
        self.upper_boundaries = upper_boundaries
        self.boundary_conditions = boundary_conditions

        if self.dim_number != len(self.lower_boundaries):

            raise ValueError(
                'Ensure number of dimensions is the same '
                'as number of provided lower_boundary conditions')

        if self.dim_number != len(self.upper_boundaries):

            raise ValueError(
                'Ensure number of dimensions is the same '
                'as number of provided upper_boundary conditions')

        if self.dim_number != len(self.boundary_conditions):

            raise ValueError(
                'Ensure number of dimensions is the same '
                'as number of provided boundary conditions')


def initalize_particles(SimulationRegion, SimulationParameters, distribution):
    """
    Generate initial particle positions using a specified distribution.

    Parameters
    ----------
    SimulationRegion : SimulationRegion
    SimulationParameters : SimulationParameters
    distribution : str
        Currently only 'uniform' is supported.

    Returns
    -------
    ndarray
        Array of shape (num_particles, dim_number)
        containing initial positions.
    """

    lower_boundaries = SimulationRegion.lower_boundaries
    upper_boundaries = SimulationRegion.upper_boundaries

    num_particles = SimulationParameters.num_particles
    dim_number = SimulationParameters.dim_number

    if dim_number == 1:

        if distribution == 'uniform':

            particles_axis1 = np.linspace(
                start=lower_boundaries[0],
                stop=upper_boundaries[0],
                num=num_particles + 2)[1:-1]

            axis1_grid = np.meshgrid(particles_axis1)
            particle_initial_positions = np.column_stack(axis1_grid)

            return particle_initial_positions

    if dim_number == 2:

        # Adjust particle number so that it has integer per dimension
        num_particles_each_side = int(
            np.ceil(np.sqrt(SimulationParameters.num_particles)))
        SimulationParameters.num_particles = int((num_particles_each_side)**2)

        if distribution == 'uniform':

            particles_axis1 = np.linspace(
                start=lower_boundaries[0],
                stop=upper_boundaries[0],
                num=num_particles_each_side + 2)[1:-1]

            particles_axis2 = np.linspace(
                start=lower_boundaries[1],
                stop=upper_boundaries[1],
                num=num_particles_each_side + 2)[1:-1]

            axis1_grid, axis2_grid = np.meshgrid(
                particles_axis1,
                particles_axis2)
            particle_initial_positions = np.column_stack(
                (axis1_grid.ravel(), axis2_grid.ravel()))

            return particle_initial_positions

    if dim_number == 3:

        # Adjust particle number so that it has integer per dimension
        num_particles_each_side = int(
            np.ceil(np.cbrt(SimulationParameters.num_particles)))
        SimulationParameters.num_particles = int(
            (num_particles_each_side)**3)

        if distribution == 'uniform':

            particles_axis1 = np.linspace(
                start=lower_boundaries[0],
                stop=upper_boundaries[0],
                num=num_particles_each_side + 2)[1:-1]

            particles_axis2 = np.linspace(
                start=lower_boundaries[1],
                stop=upper_boundaries[1],
                num=num_particles_each_side + 2)[1:-1]

            particles_axis3 = np.linspace(
                start=lower_boundaries[2],
                stop=upper_boundaries[2],
                num=num_particles_each_side + 2)[1:-1]

            axis1_grid, axis2_grid, axis3_grid = np.meshgrid(
                particles_axis1,
                particles_axis2,
                particles_axis3,
                indexing='ij')
            particle_initial_positions = np.column_stack(
                (axis1_grid.ravel(), axis2_grid.ravel(), axis3_grid.ravel()))

            return particle_initial_positions


class SimulationParameters():
    """
    Stores global simulation parameters and constructs time arrays.

    Parameters
    ----------
    dim_number : int
    num_particles : int
    time_step : float
    total_simulation_time : float
    beta : float
    st : float
        Dimensionless parameters from the particle equation of motion.

    Attributes
    ----------
    num_steps : int
        Total number of simulation steps.
    time_array : ndarray
        Array of time values for each step.
    """

    def __init__(self,
                 dim_number,
                 num_particles,
                 time_step,
                 total_simulation_time,
                 beta,
                 st):

        self.dim_number = dim_number
        self.num_particles = num_particles
        self.total_simulation_time = total_simulation_time
        self.time_step = time_step
        self.num_steps = int(total_simulation_time/time_step)
        self.time_array = np.linspace(
            0,
            self.total_simulation_time,
            self.num_steps)

        self.beta = beta
        self.st = st

    def current_time(self, i):

        return self.time_array[i]


class SimulationFlow():
    """
    Container for flow-related functions.

    Parameters
    ----------
    dim_number : int
        Spatial dimension.
    flow : callable
        Flow field function F(x, t).
    jacobian : callable
        Spatial derivative dF/dx.
    time_derivative : callable
        Time derivative dF/dt.
    """

    def __init__(self,
                 dim_number,
                 flow,
                 jacobian,
                 time_derivative):

        self.dim_number = dim_number
        self.flow = flow
        self.jacobian = jacobian
        self.time_derivative = time_derivative
