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