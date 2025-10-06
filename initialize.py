
class Simulation_Region():
    
    def __init__(self,
                 lower_boundary,
                 upper_boundary,
                 boundary_condition):
        
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.boundary_condition = boundary_condition

    # def_update bounadary




# Workflow: 1. Set up left and right x boundary. 2. Set boundary conditions. 3. Se up particles using these conditions

# Create boundary object so that boundaries can be stored, accessed, and updated