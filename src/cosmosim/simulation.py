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
import itertools
from time import time, sleep
from pathlib import Path
from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy import fft

class Universe:
    def __init__(
            self,
            n_particles: int        = 32,
            n_cells: int            = 64, 
            redshift: float         = 1,
            scale_factor: float     = 0.1,
            delta_a: float          = 0.001, # decide later, we also need to change to use scale factor as timestep
            time_period: tuple      = (0, 10),
            interpolate_method: str = 'ngp'
    ):
        
        # assign attributes
        self.n_particles            = n_particles 
        self.n_cells                = n_cells
        self.mass                   = self.n_cells**3 / self.n_particles**3 # normalize density field
        self.time_period            = time_period
        self.redshift               = redshift
        self.scale_factor           = scale_factor
        self.delta_a                = delta_a
        self.interpolate_method     = interpolate_method

        # cosmo model
        self.omega_m                = 1
        self.omega_k                = 0
        self.omega_lambda           = 0
        self.omega_r                = 0
        self.H_0                    = 1

        # initialize
        self._G_denom               = self._greenfunc_denom() # cache for speedup
        self.potential              = np.zeros((self.n_cells, self.n_cells, self.n_cells))
        self.density                = np.zeros((self.n_cells, self.n_cells, self.n_cells))
        self.acceleration_grid      = np.zeros((self.n_cells, self.n_cells, self.n_cells, 3))
        self.acceleration_particles = np.zeros((self.n_particles**3, 3))
        self.positions              = self._init_positions()
        self.momenta                = self._init_momenta()
        self._status                = 0 # let's work with numerical codes: 0 = initialized, 1 = running and 2 = complete or smth like that


# think of checks!

    def __repr__(self):
        #TODO when code finished, fix this
        return (
            f"Universe(n_particles={self.n_particles}, "
            f"n_cells={self.n_cells}, redshift={self.redshift}, scale_factor={self.scale_factor},"
            f"time_current=TODO, _status={self._status})"
            )
    
    def __str__(self):
        #TODO when code finished, fix this
        status_map = {0: "Initialized", 1: "Running", 2: "Complete"}
        status_str = status_map.get(self._status, "! Unknown !")
        return (
            f"{'='*50}\n"
            f"Cosmological Particle-Mesh N-Body Simulation Object\n"
            f"{'='*50}\n\n"
            f"Status: {status_str}\n"
            f"Time: t = ??? / {self.time_period[1]:.4f}\n"
            f"Redshift: z = {self.redshift:.4f}\n"
            f"Scale factor: a = {self.scale_factor:.4f}\n\n"
            f"Parameters:\n"
            f"  Particles: {self.n_particles}³\n"
            f"  Grid cells: {self.n_cells}³\n"
            f"  Timestep (da): {self.delta_a}\n"
        )

    def _init_positions(self) -> np.ndarray:
        """Initialize particle positions. For now, we use random perturbations. We should use appropriate initial conditions
        """
        positions = []
        fill_axis = np.linspace(0, self.n_cells, self.n_particles, endpoint=False)

        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for k in range(self.n_particles):
                    x = fill_axis[i]
                    y = fill_axis[j]
                    z = fill_axis[k]
                    positions.append([x, y, z])
        positions = np.array(positions)
        
        rng = np.random.default_rng()
        spacing = self.n_cells / self.n_particles
        for p in positions:
            p += rng.uniform(-0.5*spacing, 0.5*spacing, 3)
            p %= self.n_cells 
        return positions
    
    def _init_momenta(self) -> np.ndarray:
        """Initialize momenta half a timestep backwards to prepare for leapfrog integration scheme.
        """
        momenta = np.zeros((self.n_particles**3, 3))
        # Do an initial half-kick back so that momenta is on correct scale-factor for leapfrog integration
        self.interpolate_density()
        self.poisson_solver()
        self.potential_to_acceleration()
        self.interpolate_acceleration()
        momenta -= 0.5 * self.timestep_factor(self.scale_factor) * self.acceleration_particles * self.delta_a
        return momenta
    
    def interpolate_density(self) -> None:
        """Interpolate the mesh density field from the current particle positions, using the Nearest Grid Point (NPG) method.
        This simply means giving the parent grid cell the mass of the particle.
        """
        self.density = np.zeros((self.n_cells, self.n_cells, self.n_cells)) 
        if self.interpolate_method == 'ngp':
            self.method = 'ngp'
            for pos in self.positions:
                i, j, k = pos.astype(int)
                self.density[i, j, k] += self.mass
        elif self.interpolate_method =='cic':
            raise RuntimeError('Not implemented yet')
        else:
            raise RuntimeError("Unknown method provided. Options are: 'ngp'")

    def poisson_solver(self) -> None:
        """Solve Poisson's equation with the estimated density and calculate the potential. We first calculate the overdensity delta(r), then
        transform to Fourier space using scipy.fft, we compute the potential with the appropriate Green function and transform back to real 
        space.
        """
        del_r = self.density - 1
        rho_k = fft.fftn(del_r)
        G_k = -(3/8) * (self.omega_m / self.scale_factor) * self._G_denom**-1
        G_k[0, 0, 0] = 0 # handle singularity
        phi_k = G_k * rho_k #type: ignore
        phi_r = fft.ifftn(phi_k).real #type: ignore 
        self.potential = phi_r

    
    def potential_to_acceleration(self) -> None: #TODO: this is the main bottleneck of the simulation. Speed up here with Numba
        """The gravity is the negative gradient of the potential. Use central finite difference formula to estimate 
        gradient of potential at cell center from neighbouring cells. 
        """
        phi = self.potential
        for i, j, k in itertools.product(range(self.n_cells), repeat=3): 
            gx = -(phi[(i+1)%self.n_cells, j, k] - phi[(i-1)%self.n_cells, j, k]) * 0.5
            gy = -(phi[i, (j+1)%self.n_cells, k] - phi[i, (j-1)%self.n_cells, k]) * 0.5
            gz = -(phi[i, j, (k+1)%self.n_cells] - phi[i, j, (k-1)%self.n_cells]) * 0.5
            self.acceleration_grid[i, j, k, 0] = gx
            self.acceleration_grid[i, j, k, 1] = gy
            self.acceleration_grid[i, j, k, 2] = gz

    def interpolate_acceleration(self) -> None:
        """Interpolate acceleration on grid back to particles using same scheme as 
        with density assignment.
        """ 
        if self.interpolate_method == 'ngp':
            for pid, pos in enumerate(self.positions):
                i, j, k = pos.astype(int)
                g = self.acceleration_grid[i, j, k, :]
                self.acceleration_particles[pid, :] = g
        elif self.interpolate_method =='cic':
            raise RuntimeError('Not implemented yet')
        else:
            raise RuntimeError("Unknown interpolation method provided. Options are: 'ngp'") 

    def update_momenta(self) -> None:
        """Method to update the momenta of the particles in the simulation, called in step. Goes from n-1/2 to n+1/2
        """
        self.momenta += self.timestep_factor(self.scale_factor) * self.acceleration_particles * self.delta_a

    def update_positions(self) -> None:
        """Method to update the positions of the particles in the simulation, called in step. Goes from n to n+1,
        """
        self.positions += (
        (self.timestep_factor(self.scale_factor + 0.5*self.delta_a) * self.momenta) / (self.scale_factor + 0.5*self.delta_a)**2
        ) * self.delta_a
        self.positions %= self.n_cells

    def step(self) -> None:
        """Step function to integrate the simulation one step forward in time. More details to follow
        """
        self.interpolate_density()
        self.poisson_solver()
        self.potential_to_acceleration()
        self.interpolate_acceleration()
        self.update_momenta()
        self.update_positions()
        self.scale_factor += self.delta_a

    def run(self, steps : int = 900) -> None:
        """Main method to run simulation instance
        """
        self._status = 1
        for _ in tqdm(range(steps)):
            self.step()
        self._status = 2
        print('Done, bye.')

    def reset(self):
        """Method to reset the simulation instance. 
        """
        pass

    def _greenfunc_denom(self) -> np.ndarray:
        """Calculate the denominator of the green function for the box, which stays the same throughout the simulation. In the poisson_solver this is then used to 
        evaluate the full Green function which is redshift-dependent.
        """
        L = self.n_cells
        denom = np.zeros((self.n_cells, self.n_cells, self.n_cells)) 
        for l, m, n in itertools.product(range(self.n_cells), repeat=3):
            denom[l, m, n] = np.sin((np.pi * l)/ L)**2 + np.sin((np.pi * m)/ L)**2 + np.sin((np.pi * n)/ L)**2
        # handle singularity
        denom[0, 0, 0] = 1
        return denom
    
    def timestep_factor(self, scale_factor : float) -> float:
        """Calculate the timestep - scalefactor conversion f(a) to use for integration

        Parameters
        ----------
        scale_factor : float
            The scale factor a to evaluate f(a) for. This differs for momenta and positions in a timestep due to leapfrog scheme, calling for custom input here.
        
        Returns
        -------
        f : float
            Conversion factor
        """
        f = (
            (self.omega_m + self.omega_k * scale_factor + self.omega_lambda * scale_factor**3) / scale_factor
        ) ** -0.5
        return f

    def plot(self):
        print(self)
        print("---> Plotting...")
        fig = plt.figure(figsize=(9,9)) 
        self.ax = fig.add_subplot(projection='3d')
        self.ax.set_aspect('equal')
        self.ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], s=1, c='black') # type: ignore
        plt.show()

    def save_checkpoint(self):
        pass

    def save_finalstate(self):
        pass