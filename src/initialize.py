
class Simulation_Region():
    
    def __init__(self,
                 lower_boundary,
                 upper_boundary,
                 boundary_condition):
        
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.boundary_condition = boundary_condition    # e.g., "periodic", "reflective", "open", but use "periodic" for now

    def update_boundary(self):
        if (self.boundary_condition == "periodic"):
            # Update boundaries for periodic conditions
            pass
        elif (self.boundary_condition == "reflective"):
            # Update boundaries for reflective conditions
            pass
        elif (self.boundary_condition == "open"):
            # Update boundaries for reflective conditions
            pass
        
class Particle():

    def __init__(self,
                 position,
                 velocity):
        
        self.position = position
        self.velocity = velocity

class Simulation_Parameters():

    def __init__(self,
                 num_particles,
                 num_steps,
                 total_time):
        
        self.num_particles = num_particles
        self.num_steps = num_steps
        self.total_time = total_time
        self.time_step = total_time / num_steps
    
    def make_time_array(self):
        time_array = []
        start_time = 0
        for i in num_steps:
            time_array.append(start_time + time_step * i)
        return time_array

class Simulation_Flow():

    def __init__(self,
                 region: Simulation_Region,
                 parameters: Simulation_Parameters):
        
        self.region = region
        self.parameters = parameters
        self.particles = self.initialize_particles()

    def initialize_particles(self):
        particles = []
        for i in range(self.parameters.num_particles):
            # Initialize particle positions and velocities here
            position = self.region.lower_boundary + (self.region.upper_boundary - self.region.lower_boundary) * i / self.parameters.num_particles
            velocity = 0  # Initial velocity can be set as needed
            particles.append(Particle(position, velocity))
        return particles

# Workflow: 1. Set up left and right x boundary. 2. Set boundary conditions. 3. Set up particles using these conditions

# Create boundary object so that boundaries can be stored, accessed, and updated