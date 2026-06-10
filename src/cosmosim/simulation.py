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
from cosmosim.utils import potential_to_acceleration_numba, interpolate_density_cic_numba, interpolate_acceleration_cic_numba
import cProfile

class Universe:
    def __init__(
            self,
            n_particles: int        = 32,
            n_cells: int            = 64, 
            redshift: float         = 1,
            scale_factor: float     = 0.1,
            delta_a: float          = 0.001, # decide later, we also need to change to use scale factor as timestep
            time_period: tuple      = (0, 10),
            interpolate_method: str = 'cic'
    ):
        
        # assign attributes
        self.n_particles            = n_particles 
        self.n_cells                = n_cells
        self.mass                   = self.n_cells**3 / self.n_particles**3 # normalize density field
        self.time_period            = time_period
        self.redshift               = redshift
        self.scale_factor_start     = scale_factor
        self.scale_factor           = scale_factor
        self.delta_a                = delta_a
        self.interpolate_method     = interpolate_method

        # cosmo model
        self.omega_m                = 1 # all matter 
        self.omega_c                = 0.95 # cdm fraction
        self.omega_b                = 0.05 # baryon fraction of total matter
        self.omega_k                = 0 # curvature
        self.omega_lambda           = 0 # dark energy
        self.omega_r                = 0
        self.H_0                    = 1

        # initialize
        self._G_denom               = self._greenfunc_denom() # cache for speedup, stays the same for all simulation
        self.potential              = np.zeros((self.n_cells, self.n_cells, self.n_cells))
        self.density                = np.zeros((self.n_cells, self.n_cells, self.n_cells))
        self.acceleration_grid      = np.zeros((self.n_cells, self.n_cells, self.n_cells, 3))
        self.acceleration_particles = np.zeros((self.n_particles**3, 3))
        self.positions              = np.zeros((self.n_particles**3, 3))   #= self._init_positions()
        self.momenta                = np.zeros((self.n_particles**3, 3))   #= self._init_momenta()
        self._status                = 0 # let's work with numerical codes: 0 = initialized, 1 = running and 2 = complete or smth like that

        # storage
        self.positions_hist         = []
        self.momenta_hist           = [] # not needed right?


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
        """Initialize particle positions. For now, we use random perturbations. We should use appropriate initial conditions.
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
        self.displacement = rng.normal(loc=0.0, scale= 1.3* spacing, size=(self.n_particles**3, 3))
        positions += self.displacement
        positions %= self.n_cells
        return positions

    def _init_momenta(self) -> np.ndarray:
        """Compute initial momenta with a backward half-kick for leapfrog synchronisation.

        Side effects
        ------------
        Populates self.density, self.potential, self.acceleration_grid, and
        self.acceleration_particles as a by-product of computing g₀. These are left
        in a valid state and reused on the first call to step().

        Returns
        -------
        momenta : np.ndarray, shape (n_particles**3, 3)
        """
        # for initial setup
        # momenta = np.zeros((self.n_particles**3, 3))
        # # Do an initial half-kick back so that momenta is on correct scale-factor for leapfrog integration
        # self.interpolate_density()
        # self.poisson_solver()
        # self.potential_to_acceleration()
        # self.interpolate_acceleration()
        # momenta -= 0.5 * self.timestep_factor(self.scale_factor) * self.acceleration_particles * self.delta_a

        # following random gaussian seeds:
        momenta = self.displacement * self.scale_factor**1.5
        return momenta
    
    def init_zeldovich(self): #TODO not finished
        """Set up a 1D sine wave in x, to test the simulation against the known analytic solution"""
        # 1) compute wave parameters
        a_cross = 10 * self.scale_factor
        k = 2 * np.pi / self.n_cells
        A = -1.0 / (a_cross * k)   # D+(a) = a for EdS (see paper for formula)

        # 2) set positions
        q = np.linspace(0, self.n_cells, self.n_particles, endpoint=False)
        x_displaced = q + self.scale_factor * A * np.sin(k * q)  # D+(a_ini) = a_ini

    def interpolate_density(self) -> None:
        """Interpolate the mesh density field from the current particle positions, using the Nearest Grid Point (NPG) method.
        This simply means giving the parent grid cell the mass of the particle.
        comment on speed: for ngp we use numpy vectorization. np.add.at is the equivalent of +=.
        """
        self.density.fill(0.0)
        if self.interpolate_method == 'ngp':
            parent_cell = np.round(self.positions).astype(int) % self.n_cells 
            np.add.at(self.density, (parent_cell[:,0], parent_cell[:,1], parent_cell[:,2]), self.mass)
        elif self.interpolate_method =='cic':
            interpolate_density_cic_numba(
                self.positions, 
                self.density, 
                self.n_particles, 
                self.n_cells, 
                self.mass
                )
        else:
            raise RuntimeError("Unknown method provided. Options are: 'ngp' or 'cic'")

    def poisson_solver(self) -> None:
        """Solve Poisson's equation with the estimated density and calculate the potential. We first calculate the overdensity delta(r), then
        transform to Fourier space using scipy.fft, we compute the potential with the appropriate Green function and transform back to real 
        space.
        """
        del_r = self.density - 1
        del_k = fft.fftn(del_r)
        G_k = -(3/8) * (self.omega_m / self.scale_factor) * self._G_denom**-1
        G_k[0, 0, 0] = 0 # handle singularity
        phi_k = G_k * del_k #type: ignore
        phi_r = fft.ifftn(phi_k).real #type: ignore

        self.potential = phi_r

    def normalize_test(self):
        """Follow the routine of poisson_solver but transform back to check normalization."""
        del_r = self.density - 1
        del_k = fft.fftn(del_r)
        del_r_back = fft.ifftn(del_k).real #type: ignore
        return del_r, del_r_back

    def potential_to_acceleration(self, numba=True) -> None:
        """The gravity is the negative gradient of the potential. Use central finite difference formula to estimate 
        gradient of potential at cell center from neighbouring cells. 
        """
        if numba:
            potential_to_acceleration_numba(
                self.potential,
                self.acceleration_grid,
                self.n_cells
            )
        else:
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
        Comment on speed: for ngp we don't use numba but numpy vectorization which is quick enough here.
        """ 
        if self.interpolate_method == 'ngp':
            parent_cell = np.round(self.positions).astype(int) % self.n_cells
            self.acceleration_particles = self.acceleration_grid[parent_cell[:,0], parent_cell[:,1], parent_cell[:,2], :]
        elif self.interpolate_method =='cic':
            interpolate_acceleration_cic_numba(
                self.positions,
                self.acceleration_particles,
                self.acceleration_grid,
                self.n_particles,
                self.n_cells
            )
        else:
            raise RuntimeError("Unknown interpolation method provided. Options are: 'ngp' or 'cic'") 

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

    def step(self, numba=True) -> None:
        """Step function to integrate the simulation one step forward in time. More details to follow
        """
        self.interpolate_density()
        self.poisson_solver()
        self.potential_to_acceleration(numba=numba)
        self.interpolate_acceleration()
        self.update_momenta()
        self.update_positions()
        self.scale_factor += self.delta_a

    def run(self, steps : int = 900, numba=True, store=False) -> None:
        """Main method to run simulation instance
        """
        self._status = 1
        print('Starting cosmological particle-mesh simulation...\n')
        
        if numba:
            print('numba go fast!')
            self.compile_numba()
        else:
            print('Numba disabled. Falling back to pure python loops.')
        
        if store == True:
            print('Storing positions and momenta..')
            for _ in tqdm(range(steps)):
                self.positions_hist.append(self.positions)
                self.momenta_hist.append(self.momenta)
                self.step(numba=numba)
        else:
            for _ in tqdm(range(steps)):
                self.step(numba=numba) 
       
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
    
    def compile_numba(self):
        """Do a dummy run to warm up
        """
        potential_to_acceleration_numba(
            np.zeros((self.n_cells, self.n_cells, self.n_cells)),
            np.zeros((self.n_cells, self.n_cells, self.n_cells, 3)),
            self.n_cells
        )
        interpolate_density_cic_numba(
            np.zeros((self.n_particles**3, 3)),
            np.zeros((self.n_cells, self.n_cells, self.n_cells)),
            self.n_particles,
            self.n_cells,
            self.mass
        )
        interpolate_acceleration_cic_numba(
            np.zeros((self.n_particles**3, 3)), 
            np.zeros((self.n_particles**3, 3)),
            np.zeros((self.n_cells, self.n_cells, self.n_cells, 3)),
            self.n_particles,
            self.n_cells
        )

    def plane_wave_1D_test(self, a_ini, a_cross, interpolate_method='cic'):
        """Test the simulation by initializing a 1D plane wave in Zeldovich Approximation perturbation. 
        The analytic solution is plotted alongside numerical results for comparison. Calling this method will reset 
        and overwrite arrays, so simulation results will be lost.
        """
        print('Starting a 1D sine wave collapse test... Warning: this overwrites parameters and resets the simulation results!')
        self.scale_factor_start = a_ini
        self.scale_factor = a_ini
        sheet_size = self.n_particles**2
        steps = int((a_cross - a_ini) / self.delta_a)
        self.interpolate_method = interpolate_method
        
        # compute wave parameters
        k_wave = 2 * np.pi / self.n_cells
        A = 1.0 / (a_cross * k_wave)   # D+(a) = a for EdS (see paper for formula)

        # analytic solution
        q = np.linspace(0, self.n_cells, self.n_particles, endpoint=False)

        # set positions and momenta in x
        x_displaced = q + a_ini * A * np.sin(k_wave * q)  # D+(a_ini) = a_ini
        a_half = a_ini - 0.5 * self.delta_a
        p_displaced = a_half**(1.5) * A * np.sin(k_wave * q)   # only x-component non-zero

        # fill arrays of positions and momenta
        positions = []
        momenta = []
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for k in range(self.n_particles):
                    xx = x_displaced[i]
                    xy = q[j]
                    xz = q[k]
                    positions.append([xx, xy, xz])

                    px = p_displaced[i]
                    py = 0
                    pz = 0
                    momenta.append([px, py, pz])
        positions = np.array(positions)
        momenta = np.array(momenta)
        for p in positions:
            p %= self.n_cells
        
        self.positions = positions
        self.momenta = momenta

        x_exact_ini = q + a_ini * A * np.sin(k_wave * q)
        p_exact_ini = a_ini**1.5 * A * np.sin(k_wave * q)
        x_sim_ini = self.positions[::sheet_size, 0]
        p_sim_ini = self.momenta[::sheet_size, 0]

        plt.plot(x_exact_ini, p_exact_ini, label='analytic', zorder=-1)
        plt.scatter(x_sim_ini, p_sim_ini, label='numerical', marker='d', c='black')
        plt.legend()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$p$')
        plt.title(f'ZA test: 1D sine wave collapse\na = {a_ini:.1f}')
        plt.show()
        plt.close()

        self.run(steps=steps)

        x_exact_cross = q + a_cross * A * np.sin(k_wave * q)
        p_exact_cross = a_cross**1.5 * A * np.sin(k_wave * q)
        x_sim_cross = self.positions[::sheet_size, 0]
        p_sim_cross = self.momenta[::sheet_size, 0]
 
        plt.plot(x_exact_cross, p_exact_cross, label='analytic', zorder=-1)
        plt.scatter(x_sim_cross, p_sim_cross, label='numerical', marker='d', c='black')
        plt.legend()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$p$')
        plt.title(f'ZA test: 1D sine wave collapse\na = {a_cross:.1f}')
        plt.show()
        plt.close()


    def plot(self):
        print(self)
        print("---> Plotting...")
        fig = plt.figure(figsize=(9,9)) 
        self.ax = fig.add_subplot(projection='3d')
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], s=0.1, c='black') # type: ignore
        plt.show()
        plt.close('all')

    def save_checkpoint(self):
        pass

    def save_finalstate(self):
        pass