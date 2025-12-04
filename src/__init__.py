"""
PyPLF (Python Particle-Laden Flow).

This package provides tools for simulating inertial particle trajectories
in prescribed fluid flows.

Submodules
----------
initialize
    Simulation domain, flow, and parameter setup.
simulate
    Numerical integrator (RK4) and ODE definitions.
boundary_conditions
    Periodic and open boundary condition implementations.
utils
    Flow construction, array creation, saving, and loading utilities.
animate
    Functions to generate 2-D animations from saved data.
execution
    Command-line interface for running full simulations from a config file.

Users typically import from submodules directly instead of importing from
the top-level package.
"""
