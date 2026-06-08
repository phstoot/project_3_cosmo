import numpy as np
import matplotlib.pyplot as plt
from cosmosim.simulation import Universe

n_particles = 32
n_cells = 64

if __name__ == '__main__':
    test = Universe(n_particles=n_particles, n_cells=n_cells)
    test.positions = test._init_positions()
    test.momenta = test._init_momenta()
    test.plot()
    test.run(steps=900)
    test.plot()