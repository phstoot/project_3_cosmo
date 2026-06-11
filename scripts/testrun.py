import numpy as np
import matplotlib.pyplot as plt
from cosmosim.simulation import Universe

n_particles = 32
n_cells = 64

if __name__ == '__main__':
    test = Universe(n_particles=n_particles, n_cells=n_cells)
    test.positions = test._init_positions()
    test.momenta = test._init_momenta()
    # test.plot_colour(three_D=True, gridoff=True)
    test.run(steps=100, store = True)
    # print("Colour map with gist_earth")
    
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
    # test.plot_animation(three_D=False, gridoff=True)