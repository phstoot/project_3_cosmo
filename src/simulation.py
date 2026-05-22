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
            size: float,
            n_particles: int,
            n_grid: int, # rename to n_cells?
            delta_t: float,
            redshift: float,
            scale_factor: float,
            time_period: tuple = (0, 10),
    ):
        self.size = size
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.delta_t = delta_t
        self.time_period = time_period
        self.redshift = redshift
        self.scale_factor = scale_factor

        # initialize
        self.positions = 0
        self.momenta = 0
        self.potential = 0
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
            f"Universe(size={self.size}, n_particles={self.n_particles}, "
            f"n_grid={self.n_grid}, redshift={self.redshift}, scale_factor={self.scale_factor},"
            f"time_current={self.time_current}, _status={self._status})"
            )
    
    def __str__(self):
        status_map = {0: "Initialized", 1: "Running", 2: "Complete"}
        status_str = status_map.get(self._status, "! Unknown !")
        return (
            f"{'='*50}\n"
            f"Cosmological Particle-Mesh Simulation Object\n"
            f"{'='*50}\n\n"
            f"Status: {status_str}\n"
            f"Time: t = {self.time_current:.4f} / {self.time_period[1]:.4f}\n"
            f"Redshift: z = {self.redshift:.4f}\n"
            f"Scale factor: a = {self.scale_factor:.4f}\n\n"
            f"Parameters:\n"
            f"  Box size: {self.size}\n"
            f"  Particles: {self.n_particles}\n"
            f"  Grid cells: {self.n_grid}³\n"
            f"  Timestep: {self.delta_t}\n"
        )

    def init_grid(self):
        pass

    def density_field(self):
        """Method to interpolate the density field from the current state of the simulation
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
        pass

    def save_checkpoint(self):
        pass

    def save_finalstate(self):
        pass