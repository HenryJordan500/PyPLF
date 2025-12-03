"""
PyPLF (Python Particle‑Laden‑Flow) package.

This package provides tools for simulating the motion of inertial particles in prescribed flows.
Key submodules include:
- initialize – classes for simulation regions, flow definitions and parameters, and particle initialization.
- simulate   – numerical integration of particle trajectories using a Runge–Kutta 4th‑order solver.
- boundary_conditions – functions implementing periodic/open boundary conditions.
- utils      – utilities for creating flow functions, generating tracking arrays, and saving/extracting data.
- animate    – functions for creating animations from saved simulation data.
- execution  – command‑line interface for running a simulation from a YAML configuration file.

Users typically import from submodules rather than from the package root.
"""
