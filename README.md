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







Maybe write functions to do the slicing so it is more clear what is happening?

