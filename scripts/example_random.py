###############################################
# EXAMPLE RUN: Show basic application of simulatio:
# This script runs a simultion with random initial conditions.
#
# outputs: 
#  - 3D plots
#  -  sliced animations
#  - Power spectrum plots (with and without CIC correction)
###############################################

import numpy as np
import matplotlib.pyplot as plt
from cosmosim.simulation import Universe
from cosmosim.utils import section, spacer
from time import sleep

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
    section("Overview")
    print("This script demonstrates a random simulation.")
    spacer()
    sleep(1)

    print("Initializing...")
    # test = Universe(
    #     n_particles=n_particles, 
    #     n_cells=n_cells,
    #     boxlength=L_box,
    #     scale_factor=a_ini,
    #     h=h,
    #     omega_0m=omega_0m,
    #     omega_0k=omega_0k,
    #     omega_0lamb=omega_0lamb,
    #     )
    
    # # no transfer function
    # test.generate_ics(random=True) 

    # Run for 900 steps until a=1, today
    # print("Initialization done, commencing runs...")
    # test.run(steps=900, store = True)
    
    n_runs = 5

    all_pk = []

    for run in range(n_runs):

        print(f"Run {run+1}/{n_runs}")

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

        test.run(steps=900, store=True)

        pk, k_bins = test._calculate_power_spectrum()

        all_pk.append(pk)
        print(f"power spectrum for {run}: {pk}")
        
        print(test.scale_factor)

    all_pk = np.asarray(all_pk)
    
    pk_mean = np.mean(all_pk, axis=0)
    pk_std = np.std(all_pk, axis=0, ddof=1)
    print(f"mean power spectrum: {pk_mean}")
    print(f"std power spectrum: {pk_std}")

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.errorbar(
        k_bins,
        pk_mean,
        pk_std,
        fmt='o',                 # marker style
        markersize=7,
        color="#479d2a",         # main color
        ecolor="#63d03fff",       # lighter errorbar color
        elinewidth=1.2,
        capsize=3,
        capthick=1,
        # linestyle='--',           # connect points
        linewidth=1,
        alpha=0.9,
        label="w/ CIC correction"
        )
    
    ax1.set_ylim([min(pk_mean[pk_mean > 0]) / 1.5, 1.5 * max(pk_mean)])
    
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$k \, [\mathrm{h Mpc^{-1}}]$")
    plt.ylabel(r"$P(k) \, [\mathrm{(Mpc / h)^{3}}]$")

    # plt.legend()
    plt.grid(alpha=0.3)
    plt.title(rf"Power Spectrum, a={test.scale_factor:.3f}, ${test.n_particles}^3$ particles, random start")

    plt.show()
    
    # # Visualize evolution, Final state (as plot) and Power Spectrum
    # print("Plotting Power Spectrum with CloudInCell correction")
    # Power_spectrum, k_bins = test._calculate_power_spectrum()
    # plt.scatter(k_bins, Power_spectrum)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('k')
    # plt.ylabel('P(k)')
    # plt.title('Power Spectrum')
    # plt.grid()
    # plt.show()
    
    # print("Plotting Power Spectrum without CloudInCell correction")
    # Power_spectrum, k_bins = test._calculate_power_spectrum(cic_correction=False)
    # plt.scatter(k_bins, Power_spectrum)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('k')
    # plt.ylabel('P(k)')
    # plt.title('Power Spectrum without CIC correction')
    # plt.grid()
    # plt.show()
    
    print("Finally, here some animation of the evolution (slices projection)")
    test.plot_animation(three_D=False, gridoff=True)
    print("... and the final state.")
    test.plot() 