#!/usr/bin/env python
#####################################################
# Computational Physics b, final project:
# a small cosmological simulation of a ΛCDM universe
#
# Written by Nils Thiessen, Philip Stoot, May 2026
#####################################################
"""
This module contains several utility functions used in the simulation.
"""

import os
import sys
import json
import warnings
import itertools
import math
from time import time, sleep
from pathlib import Path
from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy import fft
import numba as nb
from numba import njit, prange

@njit(parallel = True)
def potential_to_acceleration_numba(phi, acc, n_cells):
    """Numba go fast!"""
    for i in prange(n_cells):
        ip = (i + 1) % n_cells
        im = (i - 1) % n_cells
        
        for j in range(n_cells):
            jp = (j + 1) % n_cells
            jm = (j - 1) % n_cells
            
            for k in range(n_cells):
                kp = (k + 1) % n_cells
                km = (k - 1) % n_cells

                acc[i, j, k, 0] = -(phi[ip, j, k] - phi[im, j, k]) * 0.5
                acc[i, j, k, 1] = -(phi[i, jp, k] - phi[i, jm, k]) * 0.5
                acc[i, j, k, 2] = -(phi[i, j, kp] - phi[i, j, km]) * 0.5

@njit(fastmath=True)
def interpolate_density_cic_numba(positions, density, n_particles, n_cells, mass):
    """Numba implementation of Cloud In Cell density interpolation. Be careful to pass an empty density array into this function!
    Numba parallel is not possible due to race condition.
    """
    for pid in range(n_particles**3):
        # 1. Shift by 0.5 to align with cell centers
        x_shifted = positions[pid, 0] - 0.5
        y_shifted = positions[pid, 1] - 0.5
        z_shifted = positions[pid, 2] - 0.5

        # 2. math.floor safely handles negative numbers
        i = math.floor(x_shifted)
        j = math.floor(y_shifted)
        k = math.floor(z_shifted)

        # 3. Fractional distance is now guaranteed to be between 0.0 and 1.0
        dx = x_shifted - i
        dy = y_shifted - j
        dz = z_shifted - k

        tx = 1.0 - dx
        ty = 1.0 - dy
        tz = 1.0 - dz

        # 4. Wrap the indices safely to enforce periodic boundary conditions
        i = i % n_cells
        j = j % n_cells
        k = k % n_cells

        ip = (i + 1) % n_cells
        jp = (j + 1) % n_cells
        kp = (k + 1) % n_cells

        density[i, j, k] += mass * tx * ty * tz
        density[ip, j, k] += mass * dx * ty * tz
        density[i, jp, k] += mass * tx * dy * tz
        density[i, j, kp] += mass * tx * ty * dz
        density[ip, jp, k] += mass * dx * dy * tz
        density[ip, j, kp] += mass * dx * ty * dz
        density[i, jp, kp] += mass * tx * dy * dz
        density[ip, jp, kp] += mass * dx * dy * dz

@njit(parallel=True, fastmath=True)
def interpolate_acceleration_cic_numba(positions, acceleration_particles, acceleration_grid, n_particles, n_cells):
    """Numba implementation with parallelization of Cloud In Cell acceleration interpolation.
    """
    for pid in prange(n_particles**3):
        # 1. Shift by 0.5 to align with cell centers
        x_shifted = positions[pid, 0] - 0.5
        y_shifted = positions[pid, 1] - 0.5
        z_shifted = positions[pid, 2] - 0.5

        # 2. math.floor safely handles negative numbers
        i = math.floor(x_shifted)
        j = math.floor(y_shifted)
        k = math.floor(z_shifted)

        # 3. Fractional distance is now guaranteed to be between 0.0 and 1.0
        dx = x_shifted - i
        dy = y_shifted - j
        dz = z_shifted - k

        tx = 1.0 - dx
        ty = 1.0 - dy
        tz = 1.0 - dz

        # 4. Wrap the indices safely to enforce periodic boundary conditions
        i = i % n_cells
        j = j % n_cells
        k = k % n_cells

        ip = (i + 1) % n_cells
        jp = (j + 1) % n_cells
        kp = (k + 1) % n_cells

        acceleration_particles[pid] = (
            acceleration_grid[i, j, k, :] * tx * ty * tz +
            acceleration_grid[ip, j, k, :] * dx * ty * tz +
            acceleration_grid[i, jp, k, :] * tx * dy * tz +
            acceleration_grid[i, j, kp, :] * tx * ty * dz +
            acceleration_grid[ip, jp, k, :] * dx * dy * tz +
            acceleration_grid[ip, j, kp, :] * dx * ty * dz +
            acceleration_grid[i, jp, kp, :] * tx * dy * dz +
            acceleration_grid[ip, jp, kp, :] * dx * dy * dz 
        )

def sinc(x):
        out = np.ones_like(x)

        mask = (x != 0)

        out[mask] = np.sin(x[mask]) / x[mask]

        return out

def eh97_power_spectrum(
        k_hMpc,
        h,
        omega_m,
        omega_b,
        omega_nu,
        n,
        A,
        N_nu,
        T_cmb
):
    """Computes the Eisenstein & Hu (1997) master transfer function and linear power spectrum
    running smoothly through baryon suppression and neutrino free-streaming.
    
    Parameters
    -----------
      k_hMpc   : array_like, wavenumber in units of h/Mpc.
      omega_m  : float, total matter density parameter (Omega_c + Omega_b + Omega_nu)
      omega_b  : float, baryon density parameter
      omega_nu : float, massive neutrino density parameter
      h        : float, Hubble parameter (H0 / 100)
      n        : float, primordial spectral index (typically 1.0)
      A        : float, overall normalization factor
      N_nu     : int, number of degenerate massive neutrino species
      T_cmb    : float, CMB temperature today in Kelvin (default 2.728K)
    
    Returns
    -------
      Pk       : array_like, linear power spectrum
    """
    # convert k from comoving to physical, as required by the equations
    k = k_hMpc * h

    # Guard against log(0) or division by zero at the exact DC (k=0) mode
    k = np.where(k == 0, 1e-10, k)

    # compute cosmological fractions
    omega_c = omega_m - omega_b - omega_nu
    f_c = omega_c / omega_m
    f_b = omega_b / omega_m
    f_nu = omega_nu / omega_m
    f_cb = f_c + f_b
    f_nub = f_nu + f_b

    theta27 = T_cmb / 2.7
    omhh = omega_m * h**2
    obhh = omega_b * h**2

    # Eq 1: redshift of matter-radiation equality
    z_eq = 2.5e4 * omhh * theta27**-4

    # Eq 2,3: drag release redshift parameters
    b1 = 0.313 * omhh**-0.419 * (1.0 + 0.607 * omhh**0.674)
    b2 = 0.238 * omhh**0.223
    z_d = 1291 * omhh**0.251 / (1.0 + 0.659 * omhh**0.828) * (1 + b1 * obhh**b2)
    y_d = (1 + z_eq) / (1 + z_d)

    # Eq 4: sound horizon
    s = 44.5 * np.log(9.83 / omhh) / np.sqrt(1 + 10 * obhh**0.75)

    # Eq 5: dimensionless scale normalized by equality horizon
    q = k * theta27**2 / omhh 

    # Eq 11: growth suppression indices
    p_c = 0.25 * (5 - np.sqrt(1 + 24* f_c))
    p_cb = 0.25 * (5 - np.sqrt(1 + 24* f_cb))

    # Eq 15: small-scale suppression factor alpha_nu
    alpha_nu = (f_c / f_cb) * ((5 - 2*(p_c + p_cb)) / (5 - 4 * p_cb)) * \
               ((1 - 0.553 * f_nub + 0.126 * f_nub**3) / (1 - 0.193 * np.sqrt(f_nu * N_nu) + 0.169 * f_nu * N_nu**0.2)) * \
               (1 + y_d)**(p_cb - p_c) * \
               (1 + 0.5 * (p_c - p_cb) * (1 + 1 / ((3 - 4 * p_c) * (7 - 4 * p_cb))) * (1 + y_d)**-1 )
    
    # Eq 16-17: Gamma shape parameter
    gamma_eff = omhh * (
        np.sqrt(alpha_nu) + (1 - np.sqrt(alpha_nu)) / (1 + (0.43 * k * s)**4)
        )
    q_eff = (k * theta27**2) / gamma_eff 

    # Eq 18-21: suppression Transfer function
    beta_c = (1 - 0.949 * f_nub)**-1
    C = 14.4 + (325 / (1 + 60.5 * q_eff**1.08))
    L = np.log(np.e + 1.84 * beta_c * np.sqrt(alpha_nu) * q_eff)

    T_sup = L / (L + C * q_eff**2)

    # Eq 22-23: free streaming correction factor
    if f_nu > 0:
        q_nu = 3.92 * q * np.sqrt(N_nu / f_nu)
        Bk = 1.0 + (
            (1.24 * f_nu**0.64 * N_nu**(0.3 + 0.6 * f_nu)) / (q_nu**-1.6 + q_nu**0.8)
        )
    else:
        Bk = 1.0

    # Eq 24: master transfer function
    T_master = T_sup * Bk

    # We absorb scale-dependent growth functions into amplitude A, and get the linear power spectrum
    Pk = A * (k_hMpc**n) * T_master**2

    return Pk

def hubble_param(a, H0, omega_0m, omega_0k, omega_0lamb):
    """Compute Hubble parameter based on z = 0 parameters, as a function of a. Radiation is ignored

    Parameters
    ----------
    a : arraylike
        scale_factor
    H0 : float
        Hubble constant (Hubble paramater today)
    omega_0m : float
        matter fraction
    omega_0k : float
        curvature
    omega_0lamb : float
        dark energy

    Returns
    -------
    arraylike
        Hubble parameter as a function of a
    """
    return H0 * np.sqrt(omega_0m/a**3 + omega_0k/a**2 + omega_0lamb)

def omega_fractions(a, omega_0m, omega_0k, omega_0lamb):
    """Track cosmological density parameters over time, radiation ignored.

    Parameters
    ----------
    a : arraylike
        scale_factor
    omega_0m : float
        matter fraction
    omega_0k : float
        curvature
    omega_0lamb : float
        dark energy

    Returns
    -------
    omega_m, omega_lamb, omega_k 
        the three fractions as a function of a
    """
    E2 = omega_0m/a**3 + omega_0k/a**2 + omega_0lamb
    omega_m = (omega_0m / a**3) / E2
    omega_lamb = omega_0lamb / E2
    omega_k = (omega_0k / a**2) / E2 
    return omega_m, omega_lamb, omega_k

def growth_factor(a, omega_0m, omega_0k, omega_0lamb):
    """Compute growth factor D+(a) 

    Parameters
    ----------
    a : arraylike
        scale_factor
    omega_0m : float
        matter fraction
    omega_0k : float
        curvature
    omega_0lamb : float
        dark energy

    Returns
    -------
    D
        The growth factor as a function of a
    """
    omega_m, omega_lamb, omega_k = omega_fractions(a, omega_0m, omega_0k, omega_0lamb)
    D = (5/2) * omega_m / (omega_m**(4/7) - omega_lamb + (1 + 0.5 * omega_m) * (1 + omega_lamb/70)) * a
    return D

def growth_factor_deriv(a, H0, omega_0m, omega_0k, omega_0lamb):
    """Compute the time derivative of the growth factor, used in ZA momenta perturbations

    Parameters
    ----------
    a : arraylike
        scale_factor
    H0 : float
        Hubble constant (Hubble paramater today)
    omega_0m : float
        matter fraction
    omega_0k : float
        curvature
    omega_0lamb : float
        dark energy

    Returns
    -------
    D_deriv_t
        time derivative of growth factor as a function of a
    """
    Da = growth_factor(a, omega_0m, omega_0k, omega_0lamb)
    E2 = omega_0m/a**3 + omega_0k/a**2 + omega_0lamb
    omega_m = (omega_0m / a**3) / E2
    D_deriv_a = omega_m / (2 * a**4 * E2) * (5*a - 3*Da)
    a_deriv_t = a * hubble_param(a, H0, omega_0m, omega_0k, omega_0lamb)
    D_deriv_t = D_deriv_a * a_deriv_t
    return D_deriv_t