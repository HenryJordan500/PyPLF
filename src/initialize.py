import numpy as np
class SimulationRegion():
    
    def __init__(self,
                 lower_boundary,
                 upper_boundary,
                 boundary_condition):
        
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.boundary_condition = boundary_condition    # e.g., "periodic", "reflective", "open", but use "periodic" for now

    # def update_boundary(self):
    #     if (self.boundary_condition == "periodic"):
    #         # Update boundaries for periodic conditions
    #         pass
    #     elif (self.boundary_condition == "reflective"):
    #         # Update boundaries for reflective conditions
    #         pass
    #     elif (self.boundary_condition == "open"):
    #         # Update boundaries for reflective conditions
    #         pass

def initalize_particles(SimulationRegion, num_particles, distribution):

    lower_boundary = SimulationRegion.lower_boundary
    upper_boundary = SimulationRegion.upper_boundary
    length = upper_boundary - lower_boundary

    if distribution == 'uniform':

        dist_between_parts = length/num_particles
        half_dist = dist_between_parts/2
        particle_initial_positions = np.linspace(start=lower_boundary + half_dist,
                                                 stop=upper_boundary - half_dist,
                                                 num=num_particles)

        return  particle_initial_positions

class SimulationParameters():

    def __init__(self,
                 num_particles,
                 time_step,
                 simulation_time,
                 beta,
                 st):
        
        self.num_particles = num_particles
        self.total_time = simulation_time
        self.time_step = time_step
        self.num_steps = simulation_time/time_step
        self.time_array = np.linspace(0, self.simulation_time, self.num_steps)

        self.beta = beta
        self.st = st

class SimulationFlow():

    def __init__(self,
                 region: SimulationRegion,
                 parameters: SimulationParameters,
                 flow,
                 spatial_derivative,
                 time_derivative):
        
        self.region = region
        self.parameters = parameters
        self.flow = flow
        self.spatial_derivative = spatial_derivative
        self.time_derivative = time_derivative

# Workflow: 1. Set up left and right x boundary. 2. Set boundary conditions. 3. Set up particles using these conditions

# Create boundary object so that boundaries can be stored, accessed, and updated