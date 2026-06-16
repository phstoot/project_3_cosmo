import numpy as np
import matplotlib.pyplot as plt
from cosmosim.simulation import Universe

"""This script runs the 1D sine wave collapse to test the simulation engine against analytical solutions.
Both the Nearest Grid Point and Cloud In Cell methods are shown to compare."""

#######################################
# MODEL PARAMETERS -> global variables

n_particles = 32
n_cells = 64
a_ini = 0.1
delta_a = 0.001
L_box = 64.0  # Mpc/h

if __name__ == '__main__':
    test_ngp = Universe(
        n_particles=n_particles, 
        n_cells=n_cells,
        boxlength=L_box,
        scale_factor=a_ini,
        omega_0m=1,
        omega_0lamb=0
        )
    test_ngp.plane_wave_1D_test(0.1, 1, interpolate_method='ngp')
    
    test_ngp.plot()

    test_cic = Universe(
        n_particles=n_particles, 
        n_cells=n_cells,
        boxlength=L_box,
        scale_factor=a_ini,
        omega_0m=1,
        omega_0lamb=0
        )
    test_cic.plane_wave_1D_test(0.1, 1, interpolate_method='cic') 
    
    test_cic.plot()