import numpy as np
import matplotlib.pyplot as plt
from cosmosim.simulation import Universe

"""This example script explores the different amplitudes provided by the Universe class through the utils module."""

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
omega_0m = 1 #0.31 # total matter fraction
omega_0b = 0.045 # baryon fraction
omega_0lamb= 0 # 0.69 # dark energy fraction
omega_0k = 0 # curvature
h = 0.67
H0 = 100 * h
T_cmb = 2.7255 # Fixsen 2009
sigma8_target = 0.81


if __name__ == '__main__':  
    test1 = Universe(
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
    
    test1.generate_ics(amplitude='normalized')
    test1.run(steps=900, store = True)
    test1.plot()

    test2 = Universe(
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
    
    test1.generate_ics(amplitude='physical')
    test1.run(steps=900, store = True)
    test1.plot()
    
    # Visualize evolution, Final state (as plot) and Power Spectrum
    Power_spectrum, k_bins = test1._calculate_power_spectrum(cic_correction=False)
    plt.scatter(k_bins, Power_spectrum)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('Power Spectrum, sigma8-based amplitude')
    plt.grid()
    plt.show()
    
    Power_spectrum, k_bins = test2._calculate_power_spectrum(cic_correction=False)
    plt.scatter(k_bins, Power_spectrum)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('Power Spectrum, As-based amplitude')
    plt.grid()
    plt.show()