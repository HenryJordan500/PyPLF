"""
Boundary condition implementations for particle positions.

Provides:
- open boundary conditions (no modification)
- periodic boundary conditions (wrap-around)
"""

import numpy as np


def open_bc(SimulationRegion, i1_particle_pos_vel, k):

    return i1_particle_pos_vel


def periodic_bc(SimulationRegion, i1_particle_pos_vel, k):
    """
    Apply periodic boundary conditions in dimension `k`.

    Any particle coordinate outside the interval
    [lower_boundary, upper_boundary]
    is wrapped around by the domain length.

    Parameters
    ----------
    SimulationRegion : SimulationRegion
    i1_particle_pos_vel : ndarray
    k : int
        Dimension index.

    Returns
    -------
    ndarray
        Array with periodic boundary corrections applied.
    """

    lower_boundary = SimulationRegion.lower_boundaries[k]
    upper_boundary = SimulationRegion.upper_boundaries[k]

    L = upper_boundary - lower_boundary

    i1_particle_pos_vel[:, k] = np.where(
        i1_particle_pos_vel[:, k] > lower_boundary,
        i1_particle_pos_vel[:, k],
        i1_particle_pos_vel[:, k] + L)
    i1_particle_pos_vel[:, k] = np.where(
        i1_particle_pos_vel[:, k] < upper_boundary,
        i1_particle_pos_vel[:, k],
        i1_particle_pos_vel[:, k] - L)

    return i1_particle_pos_vel
