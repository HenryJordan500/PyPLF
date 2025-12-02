# PyPLF
Library for simulating particle laden flows in python


### Array Slicing Guide

We take full advantage of numpy's built in vectorization in this library. It leads to order of magntitude speed ups in the differential equation solving part of this code. It requires some complicated array slicing to implement this properly so we have provided an explanation of all the array slicing that we use.


We initalize 2 arrays for tracking particle statistics. Each array stores data for all particles for all time steps.

particle_pos_vel is a initalized with dimensions 
(particle number, simulation step number, number of dimensions * 2). This stores each particles postion coordiantes and velocity components. These are stored togeather for more efficent computations. 

particle_acc is initalized with dimensions (particle number, simulation step number, number of dimensions). This stores each particles acceleration components.

The array slicing is nearly the same for both arrays. 

particle_acc[k] - Returns data for all time steps and components for the kth particle 

particle_acc[:, i] - Returns data for ith time step of all particles

particle_acc[:, :, j] - Returns data for jth component for all particles and time steps

particle_acc[:, i][k] - Returns data for all components of the kth particle and the ith time step


### ODE Details

Genral Particle Equation of Motion:

$\mathbf{\ddot{x}_p} = \beta \frac{D\mathbf{u}}{Dt} + \frac{1}{St}(\mathbf{u} - \mathbf{\dot{x}_p}) + \frac{1}{Fr}\mathbf{\hat{g}} + \frac{\beta}{3} (\mathbf{u} - \mathbf{\dot{x}_p}) \times \mathbf{\omega}$

$St$ - Stokes Number $(0,\infty)$. Large number is high inertia. How quickly particle responds to the flow. Formula depends on system 

$\beta$ - Dimensionless density ratio $\frac{3\rho_f}{\rho_f+2\rho_p}$. $0 < \beta < 1$, $ 1 < \beta < 3$ (Rising Particle). Same formula for all systems.

$Fr$ - Froude Number. Can be recast as $Fr = St/v_s$ where $v_s$ is the dimensionless settling speed.

$\omega = \nabla \times \mathbf{u}$. Vorticity. Zero if potential flow $u = -\nabla \phi$. Most analytical flows have no voriticity.

Note if no gravity in simulation Fr is very large. It is best practice to remove the term from the equation. It is also best practice to remove the vorticity term for potential flow.

