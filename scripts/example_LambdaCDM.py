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
L_box = 64  # Mpc/h

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
    
    # Power_spectrum_data = []
    # k_bins = []
    
    
    # for i in range(5):
    #     test = Universe(
    #         n_particles=n_particles, 
    #         n_cells=n_cells,
    #         boxlength=L_box,
    #         scale_factor=a_ini,
    #         h=h,
    #         omega_0m=omega_0m,
    #         omega_0b=omega_0b,
    #         omega_0k=omega_0k,
    #         omega_0lamb=omega_0lamb,
    #         As=As,
    #         ns=ns,
    #         sigma8=sigma8_target,
    #         T_cmb=T_cmb
    #         )
        
    #     test.generate_ics(amplitude='normalized')
    #     Power_spectrum, k_bins = test._calculate_power_spectrum()
            
    
    # test = Universe(
    #     n_particles=n_particles, 
    #     n_cells=n_cells,
    #     boxlength=L_box,
    #     scale_factor=a_ini,
    #     h=h,
    #     omega_0m=omega_0m,
    #     omega_0b=omega_0b,
    #     omega_0k=omega_0k,
    #     omega_0lamb=omega_0lamb,
    #     As=As,
    #     ns=ns,
    #     sigma8=sigma8_target,
    #     T_cmb=T_cmb
    #     )
    
    
    # test.generate_ics(amplitude='normalized')

    # # Run for 900 steps until a=1, today
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
            omega_0b=omega_0b,
            omega_0k=omega_0k,
            omega_0lamb=omega_0lamb,
            As=As,
            ns=ns,
            sigma8=sigma8_target,
            T_cmb=T_cmb
            )

        # no transfer function
        test.generate_ics(amplitude='normalized')
        # test.generate_ics(amplitude='custom', A_custom=3e6)

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
    
    plt.ylim()

    # plt.legend()
    plt.grid(alpha=0.3)
    plt.title(rf"Power Spectrum, a={test.scale_factor:.3f}, ${test.n_particles}^3$ particles, $\Lambda \, CDM$ initial conditions")

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
    
    
    print("Finally, here some animation of the evolution (slices projection)")
    test.plot_animation(three_D=False, gridoff=True)
    print("... and the final state.")
    test.plot()

