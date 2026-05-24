#!/usr/bin/env python
#####################################################
# Computational Physics b, final project:
# a small cosmological simulation of a ΛCDM universe
#
# Written by Nils Thiessen, Philip Stoot, May 2026
#####################################################
"""
This is the main module of this code package. It contains
the simulation class used to run the cosmological simulation. 
This class can be used interactively in scripts, and is also 
used under-the-hood by the CLI part of the project.
"""

import os
import sys
import json
import warnings
from time import time, sleep
from pathlib import Path
from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

class Universe:
    def __init__(
            self,
            n_particles: int = 32,
            n_cells: int = 64, 
            mass: float = 1,
            delta_t: float = 0.1, # decide later, we also need to change to use scale factor as timestep
            redshift: float = 1,
            scale_factor: float = 0.5,
            time_period: tuple = (0, 10),
    ):
        
        self.n_particles = n_particles # we can use n_cells as equivalent to 'size' in the first assignment, since the size of a cell is 1
        self.n_cells = n_cells
        self.mass = mass
        self.delta_t = delta_t
        self.time_period = time_period
        self.redshift = redshift
        self.scale_factor = scale_factor

        # initialize
        self.potential = np.zeros((self.n_cells, self.n_cells, self.n_cells))
        self.density = np.zeros((self.n_cells, self.n_cells, self.n_cells))
        self.positions = self._init_positions()
        self.momenta = np.zeros((self.n_particles, self.n_particles, self.n_particles))
        self.time_current = 0
        self._status = 0 # let's work with numerical codes: 0 = initialized, 1 = running and 2 = complete or smth like that
        

# we need: function to initialize particles on a grid, then apply perturbation theory to displace them slightly. I believe we don't give them velocities yet.
# Then: a function to estimate the density field
# poisson solver
# update momenta
# update positions
# step
# run
# reset
# think of checks!

    def __repr__(self):
        return (
            f"Universe(n_particles={self.n_particles}, "
            f"n_cells={self.n_cells}, redshift={self.redshift}, scale_factor={self.scale_factor},"
            f"time_current={self.time_current}, _status={self._status})"
            )
    
    def __str__(self):
        status_map = {0: "Initialized", 1: "Running", 2: "Complete"}
        status_str = status_map.get(self._status, "! Unknown !")
        return (
            f"{'='*50}\n"
            f"Cosmological Particle-Mesh N-Body Simulation Object\n"
            f"{'='*50}\n\n"
            f"Status: {status_str}\n"
            f"Time: t = {self.time_current:.4f} / {self.time_period[1]:.4f}\n"
            f"Redshift: z = {self.redshift:.4f}\n"
            f"Scale factor: a = {self.scale_factor:.4f}\n\n"
            f"Parameters:\n"
            f"  Particles: {self.n_particles}³\n"
            f"  Grid cells: {self.n_cells}³\n"
            f"  Timestep: {self.delta_t}\n"
        )

    def _init_positions(self):
        # For a realistic simulation, we need to either write a sophisticated code to generate IC's, or use other code for this. 
        # For now, start with regularly placed particles throughout the volume and give them small random deviations
        positions = []
        fill_axis = np.linspace(0, self.n_cells, self.n_particles, endpoint=False, dtype=float)

        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for k in range(self.n_particles):
                    x = fill_axis[i]
                    y = fill_axis[j]
                    z = fill_axis[k]
                    positions.append([x, y, z])
        positions = np.array(positions)
        
        # perturb each particle slightly, random for now
        rng = np.random.default_rng()
        for p in positions:
            p += rng.random(3)
        return positions

    def density_npg(self):
        """Method to interpolate the mesh density field from the current particle positions, using the Nearest Grid Point (NPG) method.
        This simply means giving the parent grid cell the mass of the particle.
        """
        for i, p in enumerate(self.positions):
           cell_index = p.astype(int)
           self.density[tuple(cell_index)] += self.mass

    def density_cic(self):
        """Method to interpolate the mesh density field from the current particle positions, using the Cloud In Cell (CIC) method.
        """
        pass

    def poisson_solver(self):
        """Method to solve Poisson's equation with the estimated density and calculate the potential
        """
        pass

    def update_momenta(self):
        """Method to update the momenta of the particles in the simulation, called in step
        """
        pass

    def update_positions(self):
        """Method to update the positions of the particles in the simulation, called in step
        """
        pass

    def step(self):
        """Step function to integrate the simulation one step forward in time. More details to follow
        """
        pass

    def run(self):
        """Main method to run simulation instance
        """
        pass

    def reset(self):
        """Method to reset the simulation instance. 
        """
        pass

    def plot_state(self):
        print(self)
        print("---> Plotting...")
        fig = plt.figure(figsize=(6,6)) 
        self.ax = fig.add_subplot(projection='3d')
        self.ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], s=3, c='black') # type: ignore
        plt.show()

    def save_checkpoint(self):
        pass

    def save_finalstate(self):
        pass