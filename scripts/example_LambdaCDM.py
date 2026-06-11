###############################################
# EXAMPLE RUN: Show basic application of simulation:
# This script runs a simultion with ΛCDM initial conditions.
#
# outputs: 
#  - 3D plots
#  -  sliced animations
#  - Power spectrum plots (with and without CIC correction)
###############################################

from time import sleep
import matplotlib.pyplot as plt
from cosmosim.utils import section, spacer
import numpy as np
from cosmosim.simulation import Universe

#######################################
# MODEL PARAMETERS -> global variables

n_particles = 32
n_cells = 64
a_ini = 0.1
delta_a = 0.001
L_box = 64.0  # Mpc/h

# physical constants and variables
As = 2.105e-9
ns = 0.967
omega_0m = 0.31 #0.31 # total matter fraction
omega_0b = 0.045 # baryon fraction
omega_0lamb= 0.69 # 0.69 # dark energy fraction
omega_0k = 0 # curvature
h = 0.67
H0 = 100 * h
T_cmb = 2.7255 # Fixsen 2009
sigma8_target = 0.81


if __name__ == '__main__':
    section("Overview")
    print("This script demonstrates a LambdaCDM simulation.")
    spacer()
    sleep(1)

    print("Initializing...")
    ## Initialize Lambda CDM simulation
    
    test = Universe(
        n_particles=n_particles, 
        n_cells=n_cells,
        boxlength=L_box,
        scale_factor=a_ini,
        h=h,
        omega_0m=omega_0m,
        omega_0b=omega_0b,
        omega_0k=omega_0k,
        omega_0lamb=omega_0lamb,
        As=As,
        ns=ns,
        sigma8=sigma8_target,
        T_cmb=T_cmb
        )
    
    
    test.generate_ics(amplitude='normalized')

    # Run for 900 steps until a=1, today
    print("Initialization done, commencing runs...")
    test.run(steps=900, store = True)
    
    # Visualize evolution, Final state (as plot) and Power Spectrum
    print("Plotting Power Spectrum with CloudInCell correction")
    Power_spectrum, k_bins = test._calculate_power_spectrum()
    plt.scatter(k_bins, Power_spectrum)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('Power Spectrum')
    plt.grid()
    plt.show()
    
    print("Plotting Power Spectrum without CloudInCell correction")
    Power_spectrum, k_bins = test._calculate_power_spectrum(cic_correction=False)
    plt.scatter(k_bins, Power_spectrum)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('Power Spectrum without CIC correction')
    plt.grid()
    plt.show()
    
    print("Finally, here some animation of the evolution (slices projection)")
    test.plot_animation(three_D=False, gridoff=True)
    print("... and the final state.")
    test.plot()




# Run for 900 steps


# Visualize evolution, Final state (as plot) and Power Spectrum



## Initialize \lambda CDM simulation


# Same steps as before



# Some comment on this (what changed? Filaments observed? Requires bigger model)





# Potentially bigger simulation
