import numpy as np
import matplotlib.pyplot as plt
from cosmosim.simulation import Universe

test = Universe()
print(test)


n_cells = 10
n_particles = 5

positions = []
fill_axis = np.linspace(0, n_cells, n_particles, endpoint=False, dtype=float)

for i in range(n_particles):
    for j in range(n_particles):
        for k in range(n_particles):
            x = fill_axis[i]
            y = fill_axis[j]
            z = fill_axis[k]
            positions.append([x, y, z])

positions = np.array(positions)