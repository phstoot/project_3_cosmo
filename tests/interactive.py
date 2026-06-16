import numpy as np
import matplotlib.pyplot as plt
import cosmosim
from cosmosim.simulation import Universe
from cosmosim.utils import *
from astropy import units as u

# This file is free to edit interactive
#####################################

As = 2.105e-9
ns = 0.967

n_particles = 32
n_cells = 64
a_ini = 0.1
delta_a = 0.001

L_box = 64.0  # Mpc/h



# # final part: normalizing sigma8

def tophat_window(k, R=8.0):
    x = k * R
    # Handle the limit x -> 0 smoothly to avoid 0/0 errors
    return np.where(x < 1e-3, 1.0, 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3))

# 1. Create a fine 1D array of physical k values for integration
k_integration = np.logspace(-4, 2, 2000) # h/Mpc

# 2. Evaluate your unnormalized spectrum (A=1) along this 1D array
pk_unnorm = eh97_power_spectrum(k_integration, omega_m=0.31, omega_b=0.045, h=0.7, A=1.0, omega_nu=0, n=ns, N_nu=1, T_cmb = 2.7255)

# 3. Integrate to find the unnormalized sigma_8
window = tophat_window(k_integration, R=8.0)
integrand = (k_integration**2) * pk_unnorm * (window**2) / (2.0 * np.pi**2)
sigma8_unnorm_sq = np.trapezoid(integrand, k_integration) #type: ignore

# 4. Compute the true physical normalization constant A
# Let's assume a standard observational target like sigma_8 = 0.81
sigma8_target = 0.81
A_true = (sigma8_target**2) / sigma8_unnorm_sq

plt.loglog(k_integration, pk_unnorm)
plt.title("Unnormalized power spectrum (Eisenstein & Hu, 1997)" )
plt.xlabel('k')
plt.ylabel('P(k)')
plt.show()