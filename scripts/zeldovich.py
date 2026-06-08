# this is a script to set up the 1D sine wave Zeldovich test

import numpy as np
import matplotlib.pyplot as plt
from cosmosim.simulation import Universe

n_particles = 32
n_cells = 32
a_ini = 0.1
delta_a = 0.001
sheet_size = n_particles**2
# 1) compute wave parameters
a_cross = 10 * a_ini
k = 2 * np.pi / n_cells
A = 1.0 / (a_cross * k)   # D+(a) = a for EdS (see paper for formula)

def plotphase():
    pos_x = zeldovich_test.positions[::sheet_size,0]
    mom_x = zeldovich_test.momenta[::sheet_size,0]
    pos_analytic, mom_analytic = analytic_zeldovich(zeldovich_test.scale_factor)
    plt.scatter(pos_x, mom_x)
    # plt.plot(pos_analytic, mom_analytic)
    plt.title(f'phase diagram a = {zeldovich_test.scale_factor:.3f}')
    plt.show()
    plt.close()

def analytic_zeldovich(a):
    q = np.linspace(0, n_cells, n_particles, endpoint=False)
    x = q + a * A * np.sin(k * q)
    p = a**1.5 * A * np.sin(k * q)
    return x, p

if __name__ == '__main__':
    
    # 2) set positions in x
    q = np.linspace(0, n_cells, n_particles, endpoint=False)
    x_displaced = q + a_ini * A * np.sin(k * q)  # D+(a_ini) = a_ini

    # 3) set momenta in x
    a_half = a_ini - 0.5 * delta_a
    p_x = a_half**(1.5) * A * np.sin(k * q)   # only x-component non-zero

    plt.scatter(x_displaced, p_x)
    plt.title('template')
    plt.show()
    plt.close()

    # fill arrays of positions and momenta
    positions = []
    fill_axis = np.linspace(0, n_cells, n_particles, endpoint=False)
    for i in range(n_particles):
        for j in range(n_particles):
            for k in range(n_particles):
                x = x_displaced[i]
                y = fill_axis[j]
                z = fill_axis[k]
                positions.append([x, y, z])
    positions = np.array(positions)
    for p in positions:
        p %= n_cells

    momenta = []
    for i in range(n_particles):
        for j in range(n_particles):
            for k in range(n_particles):
                x = p_x[i]
                y = 0
                z = 0
                momenta.append([x, y, z])
    momenta = np.array(momenta)

    # set up simulation
    zeldovich_test = Universe(n_particles=n_particles, n_cells=n_cells, delta_a=delta_a)
    zeldovich_test.positions = positions
    zeldovich_test.momenta = momenta
    zeldovich_test.plot()

    plotphase()

    for _ in range(9):
        zeldovich_test.run(steps=100, numba=True, store=False)
        # zeldovich_test.plot()
        plotphase()
    
    # zeldovich_test.plot()

    # pos_x = np.reshape(zeldovich_test.positions_hist[0][:,0], (16, 256))[:,0] 
    # mom_x = np.reshape(zeldovich_test.momenta_hist[0][:,0], (16, 256))[:,0] 
    # vel_x = mom_x / (a_ini)
    # plt.scatter(pos_x, mom_x)
    # plt.show()

