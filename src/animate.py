"""
Create 2-D animations of particle trajectories using saved simulation data.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils import *


def make_animation(save_path, SimulationRegion, SimulationParameters, fps=30):
    """
    Generate and save an MP4 animation of particle motion.

    Parameters
    ----------
    save_path : str
        Base file path (no extension) for loading data and saving animation.
    SimulationRegion : SimulationRegion
        Provides spatial boundaries for plotting.
    SimulationParameters : SimulationParameters
        Contains total simulation time, timestep, and dimensionless parameters.
    fps : int, optional
        Frames per second in the output video (default 30).

    Notes
    -----
    This function assumes 2-D simulations for visualization.
    """

    st = SimulationParameters.st
    beta = SimulationParameters.beta
    positions = extract_data(save_path=save_path,
                             extract='position')

    sim_time = SimulationParameters.total_simulation_time
    time_step = SimulationParameters.time_step

    nfo = int(sim_time*1/time_step)
    num_steps = SimulationParameters.num_steps

    # Number of Frames to Animate For 30 fps
    if nfo <= fps*sim_time:
        every = 1
        nf = nfo
    else:
        every = int(1/(fps*num_steps))
        nf = int(nfo/every)

    t = np.linspace(0, sim_time, num_steps)  # Initialize time array

    xmin = SimulationRegion.lower_boundaries[0]
    xmax = SimulationRegion.upper_boundaries[0]
    zmin = SimulationRegion.lower_boundaries[1]
    zmax = SimulationRegion.upper_boundaries[1]

    aspect = (zmax-zmin)/(xmax-xmin)
    width = 8
    height = 8
    fig, ax = plt.subplots(1, 1, figsize=(width, height*aspect))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_xlabel(f'X')
    ax.set_ylabel(f'Y')

    dot = ax.plot(
        positions[:, 0][:, 0],
        positions[:, 0][:, 1],
        'r.',
        markersize=4*width/4
    )[0]
    title_str = (
        f'Particles in linear flow at t = {t[0]:.3f}, '
        f'$St$ = {st}, $\\beta$ = {beta}'
    )
    tit = ax.set_title(title_str, fontsize=12 * width / 8)

    def frame(i):

        dot.set_data(positions[:, i*every][:, 0], positions[:, i*every][:, 1])
        title_str = (
            f'Particles in linear flow at t = {t[i*every]:.3f}, '
            f'$St$ = {st} , $\\beta$ = {beta}'
        )
        tit.set_text(title_str)

        return dot, tit

    anim = FuncAnimation(
        fig, frame, frames=nf,
        interval=1000*time_step*every*1, blit=True)
    anim.save(f'{save_path}.mp4')
