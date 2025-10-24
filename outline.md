# Setting up flow
Input: 
- analytical nD flow U(x_i, t)
- analytical derivatives of the flow
Output: 
- Object with stored Functions that calculate the flow and its derivatives at a given point in space and time

# Setting up Simulation space
Input: 
- 2nD boundarys
- 2nD boundary conditions
Output:
- Object that can be called to get the boundaries and conditions

# Setup Particles
Input: 
- Number of particles 
    - Evently Distributed
    - Randomly Distributed
Output:
- Numpy array of all particle positions at inital time

# Run Simulation
Input: 
- Inital Particle Positions
- Simulation time
- Simulation time steps
- Simualtion Space Object
- Flow object
Output:
- h5py file with position, velocity, acceleration of all particles at all time

# Visualizations
Input:
- h5py file
- Simulation parameters (time step, time, simulation space)
Ouputs:
- Animation of particles positions over time
- Plot of Particle positoins at one time 
- Animaiton of a single particles trajectory over time
- Plot of a single particles trajectory at one time




