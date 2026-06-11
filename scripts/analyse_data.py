import numpy as np
import matplotlib.pyplot as plt
import cosmosim
from cosmosim.simulation import Universe
from cosmosim.utils import *
from astropy import units as u

positions = np.load('data/finalpositions_1Gpc.npy')
density = np.load('data/finaldensity_1Gpc.npy')
emptysim = Universe()
emptysim.positions = positions
emptysim.density = density

Pk, k = emptysim._calculate_power_spectrum(cic_correction=False)

emptysim.plot_colour(three_D=True)

plt.scatter(k, Pk)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.title('Power Spectrum')
plt.grid()
plt.show()