import numpy as np
class SimulationRegion():
    
    def __init__(self,
                 dim_number,
                 lower_boundaries,
                 upper_boundaries,
                 boundary_conditions):
        
        self.dim_number = dim_number
        self.lower_boundaries = lower_boundaries
        self.upper_boundaries = upper_boundaries
        self.boundary_conditions = boundary_conditions    # e.g., "periodic", "reflective", "open", but use "periodic" for now

        if self.dim_number != len(self.lower_boundaries):

            raise ValueError('Ensure number of dimensions is the same as number of provided lower_boundary conditions')
        
        if self.dim_number != len(self.upper_boundaries):

            raise ValueError('Ensure number of dimensions is the same as number of provided upper_boundary conditions')
        
        if self.dim_number != len(self.boundary_conditions):

            raise ValueError('Ensure number of dimensions is the same as number of provided boundary conditions')

def initalize_particles(SimulationRegion, SimulationParameters, distribution):

    lower_boundaries = SimulationRegion.lower_boundaries
    upper_boundaries = SimulationRegion.upper_boundaries

    num_particles = SimulationParameters.num_particles
    dim_number = SimulationParameters.dim_number

    
    if dim_number == 1:

        if distribution == 'uniform':

            particles_axis1 = np.linspace(start=lower_boundaries[0],
                                                     stop=upper_boundaries[0],
                                                     num=num_particles + 2)[1:-1]

            axis1_grid = np.meshgrid(particles_axis1)
            particle_initial_positions = np.column_stack(axis1_grid)
            
            return  particle_initial_positions
            
    if dim_number == 2:

        # Adjust particle number so that it has integer per dimension
        num_particles_each_side = int(np.ceil(np.sqrt(SimulationParameters.num_particles)))
        SimulationParameters.num_particles = int((num_particles_each_side)**2)

        if distribution == 'uniform':
            
            particles_axis1 = np.linspace(start=lower_boundaries[0],
                                          stop=upper_boundaries[0],
                                          num=num_particles_each_side + 2)[1:-1]
            
            particles_axis2 = np.linspace(start=lower_boundaries[1],
                                          stop=upper_boundaries[1],
                                          num=num_particles_each_side + 2)[1:-1]

            axis1_grid, axis2_grid = np.meshgrid(particles_axis1, particles_axis2)
            particle_initial_positions = np.column_stack((axis1_grid.ravel(), axis2_grid.ravel()))

            return particle_initial_positions
    
    if dim_number == 3:

        # Adjust particle number so that it has integer per dimension
        num_particles_each_side = int(np.ceil(np.cbrt(SimulationParameters.num_particles)))
        SimulationParameters.num_particles = int((num_particles_each_side)**3)

        if distribution == 'uniform':

            particles_axis1 = np.linspace(start=lower_boundaries[0],
                                          stop=upper_boundaries[0],
                                          num=num_particles_each_side + 2)[1:-1]
            
            particles_axis2 = np.linspace(start=lower_boundaries[1],
                                          stop=upper_boundaries[1],
                                          num=num_particles_each_side + 2)[1:-1]
            
            particles_axis3 = np.linspace(start=lower_boundaries[2],
                                          stop=upper_boundaries[2],
                                          num=num_particles_each_side + 2)[1:-1]

            axis1_grid, axis2_grid, axis3_grid = np.meshgrid(particles_axis1, particles_axis2, particles_axis3, indexing='ij')
            particle_initial_positions = np.column_stack((axis1_grid.ravel(), axis2_grid.ravel(), axis3_grid.ravel()))

            return particle_initial_positions


        

class SimulationParameters():

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
        self.time_array = np.linspace(0, self.total_simulation_time, self.num_steps)

        self.beta = beta
        self.st = st

    def current_time(self, i):

        return self.time_array[i]

class SimulationFlow():

    def __init__(self,
                 dim_number,
                 flow,
                 jacobian,
                 time_derivative):
        
        self.dim_number = dim_number
        self.flow = flow
        self.jacobian = jacobian
        self.time_derivative = time_derivative

# Workflow: 1. Set up left and right x boundary. 2. Set boundary conditions. 3. Set up particles using these conditions

# Create boundary object so that boundaries can be stored, accessed, and updated