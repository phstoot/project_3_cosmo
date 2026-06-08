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