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
from PIL import Image
import pandas as pd
from scipy import fft
from cosmosim.utils import *
from scipy.spatial import cKDTree # type: ignore
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import cProfile

class Universe:
    """
    Cosmological particle-mesh (PM) N-body simulation in comoving coordinates.

    Evolves a self-gravitating dark matter particle distribution from an initial
    scale factor to a target epoch using leapfrog integration in scale factor a.
    Initial conditions are generated via the Zeldovich approximation, seeded
    either from the Eisenstein & Hu (1997) transfer function with σ₈ or Aₛ
    normalisation, or from simple Gaussian random perturbations.

    Algorithm per step
    ------------------
    1: Density assignment    : NGP or CIC mass deposition onto n_cells³ mesh.
    2: Poisson solver        : FFT-based solution of ∇²φ = (3/2)(Ω_m/a)δ,
                               using a discrete Green function.
    3: Force computation     : negative central finite-difference gradient of φ.
    4: Force interpolation   : same scheme as density assignment (NGP or CIC).
    5: Leapfrog kick         : p_{n+1/2} = p_{n-1/2} + f(aₙ) g_n Δa.
    6: Leapfrog drift        : x_{n+1}   = x_n + f(a_{n+1/2}) p_{n+1/2} / a²_{n+1/2} Δa.

    Positions are in comoving cell units [0, n_cells). The canonical momentum p
    is defined such that dx/da = f(a) p / a², where
    f(a) = [a / (Ω_m + Ω_k a + Ω_Λ a³)]^{1/2} (H₀ = 1 convention).

    Parameters
    ----------
    n_particles : int
        Number of particles per side; total particle count is n_particles³.
    n_cells : int
        Number of mesh cells per side; total grid size is n_cells³.
    boxlength : float
        Comoving side length of the simulation box in Mpc/h.
    scale_factor : float
        Initial scale factor a₀ (default 0.1, i.e. z = 9).
    delta_a : float
        Scale factor step size Δa per leapfrog step.
    interpolate_method : {'ngp', 'cic'}
        Mass/force interpolation scheme. CIC is recommended; it must be the
        same for both density assignment and force interpolation to avoid
        self-forces.
    h : float
        Dimensionless Hubble parameter (H₀ = 100h km s⁻¹ Mpc⁻¹).
    omega_0m : float
        Total matter density parameter Ω_{m,0} = Ω_c + Ω_b.
    omega_0b : float
        Baryon density parameter Ω_{b,0}.
    omega_0k : float
        Curvature density parameter Ω_{k,0}.
    omega_0lamb : float
        Cosmological constant density parameter Ω_{Λ,0}.
    As : float
        Primordial scalar amplitude at pivot scale k_p = 0.05 Mpc⁻¹
        (Planck convention). Used when amplitude='physical' in generate_ics().
    ns : float
        Primordial spectral index nₛ.
    sigma8 : float
        Target RMS matter fluctuation amplitude in spheres of radius 8 h⁻¹ Mpc
        at z = 0. Used when amplitude='normalized' in generate_ics().
    T_cmb : float
        CMB temperature today in Kelvin (Fixsen 2009: 2.7255 K).

    Key methods
    -----------
    generate_ics(amplitude)
        Compute Zeldovich-approximation initial conditions from the EH97 power
        spectrum. Call before run(). Supports 'physical' (Aₛ-based),
        'normalized' (σ₈-based), and 'custom' amplitude modes.
    run(steps, store, interval)
        Advance the simulation for a given number of steps. Optionally store
        snapshots of positions and density for later animation or analysis.
    plane_wave_1D_test(a_ini, a_cross)
        Validate the PM solver against the analytic 1D plane-wave collapse
        solution (Zeldovich 1970); plots the phase diagram at initial and
        crossing epochs.
    plot_colour(three_D, thickness)
        Visualise the particle distribution as a projected density heatmap
        or 3D colour scatter plot.
    plot_animation(three_D, fps, name)
        Render a GIF animation of the stored snapshots.

    Notes
    -----
    - The Green function denominator is precomputed in __init__ and cached as
      _G_denom since the sin² terms are time-independent.
    - Numba JIT compilation is triggered on the first run() call and used in step
    - Radiation (Ω_r) is neglected throughout.

    Examples
    --------
    EdS matter-only run from z = 9:

    >>> sim = Universe(n_particles=32, n_cells=64, omega_0m=1, omega_0lamb=0)
    >>> sim.generate_ics(amplitude='normalized')
    >>> sim.run(steps=900, store=True, interval=10)
    >>> sim.plot_colour()

    Plane-wave validation test:

    >>> sim = Universe(n_particles=32, n_cells=64, omega_0m=1)
    >>> sim.plane_wave_1D_test(a_ini=0.1, a_cross=1.0)
    """

    def __init__(
            self,
            n_particles: int        = 32,
            n_cells: int            = 64, 
            boxlength: float        = 64,
            scale_factor: float     = 0.1,
            delta_a: float          = 0.001,
            interpolate_method: str = 'cic',
            h: float                = 0.67,
            omega_0m: float         = 0.31,
            omega_0b: float         = 0.045,
            omega_0k: float         = 0,
            omega_0lamb: float      = 0.69,
            As : float              = 2.105e-9,
            ns : float              = 0.967,
            sigma8 : float          = 0.81,
            T_cmb : float           = 2.7255
    ):
        """
        Neat Particle-Mesh Simulation of ΛCDM universe
        ---------- 
        The Universe class encapsulates the state and dynamics of a cosmological N-body simulation using the Particle-Mesh method. 
        It provides methods for initializing particle positions and momenta, interpolating density fields, solving Poisson's equation for gravitational potential, and updating particle states over time.
        The main density interpolation method follows the Cloud in Cell algorithm, which provides a decently accurate replacement of calculating each individual force between particles.
        Instances of the class can be run with or without animation, and one can store the history of positions and momenta of particles,
        as well as the density and scale factor.
        
        The core of the simulation implements the numba JIT compiler to significantly speed up runtime.  In a 
        standard user laptop/computer, this code should be compute 10^3 timesteps in less than a minute for a 64^3 sized system.
        
        Parameters
        --------
        n_particles : int
            number of particles per dimension, by default 32
        n_cells : int
            number of cells per dimension, by default 64
        redshift : float, optional
            redshift, by default 1
        scale_factor : float
            initial scale factor of the simulation, by default 0.1
        delta_a: float
            increases in scale factor per timestep, by default 0.001
        time_period: tuple
        interpolate_method : str 
            density interpolation algorithm, by default 'cic'
        h : float                
            little h for Hubble constant, by default 0.67
        omega_0m : float 
            Matter fraction, by default 0.31
        omega_0b: float         
            Baryon fraction, by default 0.045
        omega_0k: float   
            Curvature, by default 0
        omega_0lamb: float   
            Dark Energy fraction, by default 0.69
            
        """
        
        # assign attributes
        self.n_particles            = n_particles 
        self.n_cells                = n_cells
        self.boxlength              = boxlength
        self.mass                   = self.n_cells**3 / self.n_particles**3 # normalize density field
        self.scale_factor_start     = scale_factor
        self.scale_factor           = scale_factor
        self.delta_a                = delta_a
        self.interpolate_method     = interpolate_method

        # cosmo model
        self.omega_0m                = omega_0m # all matter 
        self.omega_0b                = omega_0b # baryon fraction of total matter
        self.omega_0c                = self.omega_0m - self.omega_0b
        self.omega_0k                = omega_0k # curvature
        self.omega_0lamb             = omega_0lamb # dark energy
        self.omega_0r                = 0
        self.h                       = h
        self.As                      = As
        self.ns                      = ns
        self.sigma8                  = sigma8
        self.T_cmb                   = T_cmb

        # initialize
        self._G_denom               = self._greenfunc_denom() # cache for speedup, stays the same for all simulation
        self.potential              = np.zeros((self.n_cells, self.n_cells, self.n_cells))
        self.density                = np.ones((self.n_cells, self.n_cells, self.n_cells)) * self.mass ## Minor detail, but might need to change that
        self.acceleration_grid      = np.zeros((self.n_cells, self.n_cells, self.n_cells, 3))
        self.acceleration_particles = np.zeros((self.n_particles**3, 3))
        self.positions              = np.zeros((self.n_particles**3, 3))   #= self._init_positions()
        self.momenta                = np.zeros((self.n_particles**3, 3))   #= self._init_momenta()
        self._status                = 0 # let's work with numerical codes: 0 = initialized, 1 = running and 2 = complete or smth like that
        self.z                      = self.get_redshift()

        # storage
        self.positions_hist         = []
        self.momenta_hist           = []
        self.density_hist           = [] # Maybe not necessary
        self.scale_factor_hist      = []


    def __repr__(self):
        #TODO when code finished, fix this
        return (
            f"Universe(n_particles={self.n_particles}, "
            f"n_cells={self.n_cells}, redshift={self.get_redshift()}, scale_factor={self.scale_factor},"
            f" _status={self._status})"
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
            f"Redshift: z = {self.get_redshift():.4f}\n"
            f"Scale factor: a = {self.scale_factor:.4f}\n\n"
            f"Parameters:\n"
            f"  Particles: {self.n_particles}³\n"
            f"  Grid cells: {self.n_cells}³\n"
            f"  Timestep (da): {self.delta_a}\n"
        )

    def get_redshift(self):
        """Get redshift from scale factor"""
        return 1/self.scale_factor - 1

    def _init_positions(self) -> np.ndarray:
        """Initialize particle positions. Use random perturbations, no power spectrum.
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
        # following random gaussian seeds:
        a_half = self.scale_factor_start - 0.5 * self.delta_a
        momenta = self.displacement * a_half**1.5
        return momenta 

    def interpolate_density(self) -> None:
        """Interpolate the mesh density field from the current particle positions, using the Nearest Grid Point (NPG) method.
        This simply means giving the parent grid cell the mass of the particle.
        For speeding up computation, ngp is used for numpy vectorization. np.add.at is the equivalent of +=.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
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

    def generate_ics(self, random: bool =False, amplitude: str ='physical', A_custom: float = 1):
        """Compute Zeldovich-approximation initial conditions from the EH97 power
        spectrum. Call before run(). Supports 'physical' (Aₛ-based),
        'normalized' (σ₈-based), and 'custom' amplitude modes. 

        Parameters
        ----------
        random : bool, optional
            whether to use random perturbations without cosmological transfer function, by default False
        amplitude : str, optional
            'physical', 'normalized', 'custom': use the physical amplitude based on As, 
            a simple normalized amplitude based on sigma8, or provide custom amplitude. By default 'physical'
        A_custom : float, optional
            The custom amplitude of the linear matter power spectrum to use, by default 1

        Raises
        ------
        RuntimeError
            _description_
        """
        if random == True:
            print('Setting random gaussian initial perturbations without cosmological power spectrum')
            self.positions = self._init_positions()
            self.momenta = self._init_momenta()
            pass
        
        if amplitude == 'physical':
            A = amplitude_physical(
                self.scale_factor_start,
                self.h,
                self.As,
                self.omega_0m,
                self.omega_0k,
                self.omega_0lamb,
                self.ns,
            )
        elif amplitude == 'normalized':
            A = amplitude_normalized(
                self.scale_factor_start,
                self.h,
                self.sigma8,
                self.omega_0m,
                self.omega_0k,
                self.omega_0lamb,
                self.ns, 
                self.T_cmb
            )
        elif amplitude == 'custom':
            if A_custom == 1:
                print('A_custom = 1. Did you forget to provide a value?')
            A = A_custom
        else:
            raise RuntimeError("Unknown option provided for amplitude. Choose 'physical', 'normalized' or 'custom'")
        
        # FS wave parameters and arrays
        dk = 2.0 * np.pi / self.boxlength # define fundamental wave mode with physical units

        k_axis = np.fft.fftfreq(self.n_cells) * self.n_cells * dk
        kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing='ij')
        k_grid = np.sqrt(kx**2 + ky**2 + kz**2)
        k_grid_safe = np.where(k_grid == 0.0, 1.0, k_grid)  # Prevent 0/0 division

        # calculate power spectrum with analytical formula
        powerspectrum_EdS = eh97_power_spectrum(
            k_grid,
            h=self.h,
            omega_m=self.omega_0m,
            omega_b=self.omega_0b,
            omega_nu=0,
            n=self.ns,
            A=A,
            N_nu=1,
            T_cmb=self.T_cmb
        )

        # generate random numbers and initialize three grids
        rng = np.random.default_rng(20260610)
        gauss_noise_real = (np.sqrt(powerspectrum_EdS) * rng.normal(0, 1, size=k_grid.shape)) / k_grid_safe**2
        gauss_noise_imag = (np.sqrt(powerspectrum_EdS) * rng.normal(0, 1, size=k_grid.shape)) / k_grid_safe**2
        c_k = 0.5 * (gauss_noise_real - 1j * gauss_noise_imag)

        alpha = self.n_cells**3 / self.boxlength**3
        S_x = np.real(np.fft.ifftn(kx * c_k)) * alpha
        S_y = np.real(np.fft.ifftn(ky * c_k)) * alpha
        S_z = np.real(np.fft.ifftn(kz * c_k)) * alpha


        # now apply Zeldovich Approximation to regular grid 
        # 1D unperturbed grid coordinates
        q_1d = np.linspace(0, self.n_cells, self.n_particles, endpoint=False)

        # Compute the momentum scaling factor
        a_half = self.scale_factor_start - 0.5 * self.delta_a
        p_factor = a_half**2 * growth_factor_deriv(a_half, omega_0m=self.omega_0m, omega_0k=self.omega_0k, omega_0lamb=self.omega_0lamb)

        positions = []
        momenta = []
        # Single nested loop to sample the 3D S_x, S_y, S_z grids
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for k in range(self.n_particles):
                    # Calculate displaced positions
                    x = q_1d[i] + growth_factor(self.scale_factor_start, self.omega_0m, self.omega_0k, self.omega_0lamb) * S_x[i, j, k]
                    y = q_1d[j] + growth_factor(self.scale_factor_start, self.omega_0m, self.omega_0k, self.omega_0lamb) * S_y[i, j, k]
                    z = q_1d[k] + growth_factor(self.scale_factor_start, self.omega_0m, self.omega_0k, self.omega_0lamb) * S_z[i, j, k]
                    positions.append([x, y, z])
                    
                    # Calculate initial momenta (matching the displacement sign!)
                    px = p_factor * S_x[i, j, k]
                    py = p_factor * S_y[i, j, k]
                    pz = p_factor * S_z[i, j, k]
                    momenta.append([px, py, pz])
        self.positions = np.array(positions)
        self.momenta = np.array(momenta)
        self.positions %= self.n_cells
        print('Succesfully initialized positions and momenta using cosmological power spectrum. Ready for run()')
  
    def poisson_solver(self) -> None:
        """Solve Poisson's equation with the estimated density and calculate the potential. We first calculate the overdensity delta(r), then
        transform to Fourier space using scipy.fft, we compute the potential with the appropriate Green function and transform back to real 
        space.
        """
        del_r = self.density - 1
        del_k = fft.fftn(del_r)
        G_k = -(3/8) * (self.omega_0m / self.scale_factor) * self._G_denom**-1
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

    def run(self, steps : int = 900, numba=True, store=False, interval: int = 5) -> None:
        """Main method to run simulation instance.
        
        Parameters
        ---------
        steps : int
            Number of steps to run the simulation for.
        numba : bool   
            If numba should be used for computation.
        store : bool
            If True, position, density and scale factor are stored per interval.
        interval : int
            Determines the interval at which quantities are stored.
            
        Returns
        --------
        None        
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
                if _ % interval == 0:
                    self.positions_hist.append(self.positions.copy())
                    # self.momenta_hist.append(self.momenta.copy())
                    self.density_hist.append(self.density.copy())
                    self.scale_factor_hist.append(self.scale_factor)
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
            (self.omega_0m + self.omega_0k * scale_factor + self.omega_0lamb * scale_factor**3) / scale_factor
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


    def plot(self, mode='show', fname=f'3d.png'):
        print(self)
        print("---> Plotting...")
        fig = plt.figure(figsize=(9,9)) 
        self.ax = fig.add_subplot(projection='3d')
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], s=0.1, c='black') # type: ignore
        if mode == 'show':
            plt.show()
        elif mode == 'save':
            plt.savefig(fname, dpi=300)
        else:
            raise UserWarning("unknown mode provided. Choose 'show' or 'save'")
        plt.close('all')
        
    def plot_colour(self, three_D: bool = False, gridoff: bool = False, thickness: int = 3, time_interval: int = 5):
        """Plotting method to visualize the particle distribution. 
        Two different plotting versions, one 3D scatter with density-based colouring and one 2D heatmap of the projected density. 
        
        Parameters
        ----------
        three_D : bool, optional
            If True, plot a 3D scatter plot with density-based colouring. If False, plot a 2D heatmap of the projected density.
        gridoff : bool, optional
            If True, turn off the grid in the 3D plot.
        thickness : int, optional
            The thickness of the slice to project for the 2D heatmap.
        time_interval : int, optional
            The time interval between frames in the animation.
        
        Returns
        -------
        None
        """
        print(self)
        print("---> Plotting...")
        
        cosmo_cmap = LinearSegmentedColormap.from_list(
            "cosmo",
            [
                (0.00, "#000000"),  # black
                (0.25, "#001a66"),  # dark blue
                (0.55, "#0066ff"),  # blue
                (0.85, "#66ffff"),  # cyan
                (1.00, "#76ff7d"),  # green
            ]
        )  
        
        cosmo_cmap.set_bad("magenta")
        cosmo_cmap.set_over("red")
        cosmo_cmap.set_under("yellow")      
        
        if three_D:
            fig = plt.figure(figsize=(9,9)) 
            self.ax = fig.add_subplot(projection='3d')
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_zlabel('z')
        
            tree = cKDTree(self.positions)

            # Distance to 16th nearest neighbour
            distances, _ = tree.query(self.positions, k=16)

            r = distances[:, -1]

            # Density estimate
            density = 16 / (4/3 * np.pi * r**3)
            
            vmin = np.percentile(density[density > 0], 5)
            vmax = np.percentile(density, 99.5)
            
            self.ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], s=0.2, c=density, cmap = cosmo_cmap, norm = LogNorm(vmin=vmin, vmax=vmax)) # type: ignore
            self.ax.set_title(f"3D particle distribution with density-based colouring, a={self.scale_factor:.3f}")
            
            if gridoff:
                self.ax.grid(False)
                fig.patch.set_visible(False)
                self.ax.patch.set_visible(False)
                self.ax.set_axis_off()
                self.ax._axis3don = False
            
            else: 
                fig.colorbar(mappable = self.ax.collections[0], label='Projected density')
            plt.show()
        
        else: 
            z_center = int(self.n_cells / 2)

            heatmap = np.sum(
                self.density[:, :, z_center-thickness:z_center+thickness+1],
                axis=2
            )   + 0.00001# (remove later on, already calculate self.density when initializing)
            
            vmin = np.percentile(heatmap[heatmap > 0], 5)
            vmax = np.percentile(heatmap, 99.5)
            
            heatmap = np.maximum(heatmap, vmin) ## Deals with 0 values
            heatmap = np.minimum(heatmap, vmax)
            
            plt.figure(figsize=(8,8))
            plt.imshow(
                heatmap.T,
                origin='lower',
                cmap=cosmo_cmap,
                norm=LogNorm(vmin=vmin, vmax=vmax)
            )
            plt.colorbar(label='Projected density')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f"Projected density heatmap, a={self.scale_factor:.3f}")
            plt.show()
        
        plt.close('all')
        
    def _update_animation(self,frame: int):
        """Update function for animation, called either by FuncAnimation in the 2D plot or manually in the 3D plot.
        The artists are updated in-place of the whole plot.
        
        Parameters        
        ----------
        frame : int
            The index of the frame to update to, corresponding to the indices of the stored position and density arrays.
        
        Returns
        -------
        artists : tuple
            The artists that were updated, to be redrawn by the animation function.
        """
        
        idx = self.indices[frame]
        density = self.density_hist[idx]

        if self.three_D:
            idx = self.indices[frame]
            
            pos = self.positions_hist[idx]
            grid = self.density_hist[idx]

            ix = np.floor(pos[:,0]).astype(int) % self.n_cells
            iy = np.floor(pos[:,1]).astype(int) % self.n_cells
            iz = np.floor(pos[:,2]).astype(int) % self.n_cells

            particle_density = grid[ix, iy, iz]
            particle_density = np.clip(particle_density, self.all_min, self.all_max)
            
            
            
            self.scatter._offsets3d = ( #type: ignore
                pos[:,0],
                pos[:,1],
                pos[:,2]
            )

            self.scatter.set_array(
                particle_density
            )
            
            self.title.set_text(
                f"3D particle distribution with density-based colouring, a = {self.scale_factor_hist[idx]:.3f}"
            )

            print(
                f"Rendering frame "
                f"{frame+1}/{len(self.indices)}"
            )
                        
            return (self.scatter, self.title)
            
        else:       
            heatmap = np.sum(
                density[:, :, self.z_center-self.thickness:self.z_center+self.thickness+1],
                axis=2)
            
            # print("Density stats for frame {}: min = {:.5f}, max = {:.5f}, sum = {:.5f}".format(
            #     frame,
            #     np.min(heatmap),
            #     np.max(heatmap),
            #     np.sum(heatmap)
            # )) # As diagnostic

            heatmap = np.maximum(heatmap, self.all_min)
            heatmap = np.minimum(heatmap, self.all_max)
            self.im.set_data(heatmap.T)

            return (self.im,)     
        
    def plot_animation(self, three_D: bool = False, gridoff: bool = False, thickness: int = 3, batch_interval: int = 1, fps: int = 20, name: str = "cosmology"):
        """Animated version of plotting the evolution of the particle distribution. 
        Two different plotting versions, one 3D scatter with density-based colouring and one 2D heatmap of the projected density. 
        Requires storing of position and density arrays when using run() with store=True.
        For the 3D version, the animation using FuncAnimation is wonky, so the generation of frames is done manually, potentially slowing down the process.
        Saves a gif of the animation.
        
        Parameters
        ----------
        three_D : bool, optional
            If True, plot a 3D scatter plot with density-based colouring. If False, plot a 2D heatmap of the projected density.
        gridoff : bool, optional
            If True, turn off the grid in the 3D plot.
        thickness : int, optional
            The thickness of the slice for the 2D heatmap projection.
        batch_interval : int, optional
            The interval between frames in the animation.
        fps : int, optional
            The frames per second for the animation.
        name : str, optional
            The name of the animation file.
            
        Returns
        -------
        None

        """
        
        self.cosmo_cmap = LinearSegmentedColormap.from_list(
            "cosmo",
            [
                (0.00, "#000000"),  # black
                (0.25, "#001a66"),  # dark blue
                (0.55, "#0066ff"),  # blue
                (0.85, "#66ffff"),  # cyan
                (1.00, "#76ff7d"),  # green
            ]
        )        
        
        self.cosmo_cmap.set_bad("magenta")
        self.cosmo_cmap.set_over("red")
        self.cosmo_cmap.set_under("yellow")
        
        self.indices = np.arange(
            0,
            len(self.density_hist),
            batch_interval
        )
        
        self.three_D = three_D
        self.gridoff = gridoff
        
        if three_D:    
            ## A bit inaccurate way of finding percentiles but less memory-intensive.
            mins = [np.min(d[d > 0]) for d in self.density_hist]
            maxs = [np.max(d) for d in self.density_hist]

            self.all_min = np.percentile(mins, 5)
            self.all_max = np.percentile(maxs, 99.9)
            
            self.norm = LogNorm(vmin=self.all_min, vmax=self.all_max)
            
            ## First frame
            idx0 = self.indices[0]

            pos0 = self.positions_hist[idx0]
            grid0 = self.density_hist[idx0]

            ix = np.floor(pos0[:,0]).astype(int) % self.n_cells
            iy = np.floor(pos0[:,1]).astype(int) % self.n_cells
            iz = np.floor(pos0[:,2]).astype(int) % self.n_cells

            density0 = grid0[ix, iy, iz]

            self.fig = plt.figure(figsize=(9, 9))
            self.ax = self.fig.add_subplot(projection="3d")

            self.scatter = self.ax.scatter(
                pos0[:,0],
                pos0[:,1],
                pos0[:,2],
                c=density0,
                s=0.2,
                cmap=self.cosmo_cmap,
                norm=self.norm
            )

            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")

            self.ax.set_xlim(0,self.n_cells)
            self.ax.set_ylim(0, self.n_cells)
            self.ax.set_zlim(0, self.n_cells)
            
            self.title = self.ax.set_title(
                f"3D particle distribution with density-based colouring, a = {self.scale_factor_hist[idx0]:.3f}"
            )
            
            if self.gridoff:
                self.ax.grid(False)
                self.fig.patch.set_visible(False)
                self.ax.patch.set_visible(False)
                self.ax.set_axis_off()
                self.ax._axis3don = False
            
            else: 
                self.fig.colorbar(mappable = self.ax.collections[0], label='Projected density')
            
            ## Actual animation update
            frames = []

            for i, idx in enumerate(self.indices):
                self._update_animation(i)
                self.ax.figure.canvas.draw()
                
                buf = self.ax.figure.canvas.buffer_rgba()
                image = Image.frombytes("RGBA", self.ax.figure.canvas.get_width_height(), buf)
                frames.append(image.convert("RGB"))
            
            frames[0].save(
                f"animations/3d/{name}.gif",
                save_all=True,
                append_images=frames[1:],
                loop=0,
                duration=int(1000 / fps)
            )

            print(f"Saved {name}.gif")
        
        
        else: 
            fig, self.ax = plt.subplots()
            
            self.z_center = int(self.n_cells / 2)
            self.thickness = thickness
            
            density0 = self.density_hist[self.indices[0]]
            
            heatmap = np.sum(
                density0[:, :, self.z_center-self.thickness:self.z_center+self.thickness+1],
                axis=2
            )   + 0.00001# (remove later on, already calculate self.density when initializing)
                    
                    
            all_maps = np.array([
                d[:, :, self.z_center-self.thickness:self.z_center+self.thickness+1].sum(axis=2)
                for d in self.density_hist
            ])

            self.all_min = np.percentile(
                all_maps[all_maps > 0],
                5
            )

            self.all_max = np.percentile(
                all_maps,
                99.9
            )        
            
            heatmap = np.maximum(heatmap, self.all_min) ## Deals with 0 values
            heatmap = np.minimum(heatmap, self.all_max)
            
            self.im = self.ax.imshow(
                heatmap.T,
                origin="lower",
                cmap=self.cosmo_cmap,
                norm=LogNorm(vmin=self.all_min, vmax=self.all_max), 
                interpolation = "nearest",
                resample=False
            )
            
            ani = FuncAnimation(
                fig,
                self._update_animation,
                frames=len(self.indices),
                interval=100,
                blit=False,
                repeat=True
            )
        
            plt.colorbar(self.im, label='Projected density')
            plt.xlabel('x')
            plt.ylabel('y')
            
            
            ani.save(
                f"animations/slice/{name}.gif",
                writer="pillow",
                fps=fps
            )
            
            plt.show()
        
        plt.close('all')

    def _calculate_power_spectrum(self, cic_correction: bool = True):
        """Calculate the power spectrum of the density field. 
        One can compare this to theoretical predictions or other simulations for validation.
        The basic idea is to calculate the density field in real space, transform to Fourier space and average the squared amplitudes of the Fourier modes in spherical shells.
        It is assumed that the power spectrum is isotropic. 
        Since the Cloud-In-Cell method affects the Fourier modes, one can optionally apply a correction factor to the power spectrum to account for this. 
        
        Parameters
        ----------
        cic_correction : bool, optional
            If True, apply a correction factor to the power spectrum to account for the effects of the Cloud-In-Cell interpolation method. Default is True.
        
        Returns
        -------
        power_spec_k : np.ndarray
            The estimated power spectrum values for each k-bin.
        k_centres : np.ndarray
            The central k values for each bin, corresponding to the power spectrum values.
        """
        delta_real = self.density / np.mean(self.density) - 1.0
        delta_k = np.fft.fftn(delta_real)
        
        N = self.density.shape[0]
        dx = self.n_cells / N  ### This formula is not exactly correct, it should be L_box / N where N is grid resolution. 
        
        kx = 2*np.pi*np.fft.fftfreq(N, d=dx) # This spacing correct?
        ky = 2*np.pi*np.fft.fftfreq(N, d=dx)
        kz = 2*np.pi*np.fft.fftfreq(N, d=dx)

        KX, KY, KZ = np.meshgrid(
            kx, ky, kz,
            indexing="ij"
        )

        kmag = np.sqrt(
            KX**2 + KY**2 + KZ**2
        )
        
        print(f"K-magnitude range: {np.min(kmag[kmag > 0])} to {np.max(kmag)}")
        
        k_nyquist = np.pi / dx
        kmin = np.min(kmag[kmag > 0])
        kmax = min(k_nyquist, np.max(kmag))
        
        k_bins = np.logspace(
            np.log10(kmin),
            np.log10(kmax),
            50
        )

        if cic_correction:
            Wx = sinc(KX * dx / 2)**2
            Wy = sinc(KY * dx / 2)**2
            Wz = sinc(KZ * dx / 2)**2

            Wcic = Wx * Wy * Wz
            
            delta_k /= Wcic
        
        power_spec_k_grid = np.abs(delta_k)**2
        
        power_spec_k = np.zeros(len(k_bins)-1)
        k_centres = np.zeros(len(k_bins)-1)
        
        for i in range(len(k_bins)-1):
            k_low = k_bins[i]
            k_high = k_bins[i+1]

            mask = (
                (kmag >= k_low) &
                (kmag < k_high)
            )

            if len(power_spec_k_grid[mask]) == 0:
                power_spec_k[i] = 0
            
            else:
                power_spec_k[i] = np.mean(
                    power_spec_k_grid[mask]
                )

            k_centres[i] = np.sqrt(
                k_low * k_high
            )
        
        power_shot_noise = (self.n_cells)**3 / self.positions.shape[0]
        
        corrected_power_spectrum = np.maximum(
            power_spec_k - power_shot_noise,
            0
        )
        return corrected_power_spectrum, k_centres
    
