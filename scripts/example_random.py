import numpy as np
import matplotlib.pyplot as plt
from cosmosim.simulation import Universe

#######################################
# MODEL PARAMETERS -> global variables

n_particles = 32
n_cells = 64
a_ini = 0.1
delta_a = 0.001
L_box = 64.0  # Mpc/h

# physical constants and variables
omega_0m = 1 #0.31 # total matter fraction
omega_0lamb= 0 # 0.69 # dark energy fraction
omega_0k = 0 # curvature
h = 0.67
H0 = 100 * h


if __name__ == '__main__':
    
## Initialize random simulation (no transfer function)
    
    test = Universe(
        n_particles=n_particles, 
        n_cells=n_cells,
        boxlength=L_box,
        scale_factor=a_ini,
        h=h,
        omega_0m=omega_0m,
        omega_0k=omega_0k,
        omega_0lamb=omega_0lamb,
        )
    
    # no transfer function
    test.generate_ics(random=True) 

    # Run for 900 steps until a=1, today
    test.run(steps=900, store = True)
    
    # Visualize evolution, Final state (as plot) and Power Spectrum
    Power_spectrum, k_bins = test._calculate_power_spectrum()
    plt.scatter(k_bins, Power_spectrum)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('Power Spectrum')
    plt.grid()
    plt.show()
    
    Power_spectrum, k_bins = test._calculate_power_spectrum(cic_correction=False)
    plt.scatter(k_bins, Power_spectrum)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('Power Spectrum without CIC correction')
    plt.grid()
    plt.show()
    
    test.plot_animation(three_D=False, gridoff=True)
    test.plot()