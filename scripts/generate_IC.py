import numpy as np
import matplotlib.pyplot as plt
import cosmosim
from cosmosim.simulation import Universe
from cosmosim.utils import *
from astropy import units as u


#######################################
# MODEL PARAMETERS -> global variables

n_particles = 32
n_cells = 64
a_ini = 0.05
delta_a = 0.001
L_box = 64.0  # Mpc/h


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
# Let's assume a standard observational target like sigma_8 = 0.81
sigma8_target = 0.81


# FS wave parameters and arrays
dk = 2.0 * np.pi / L_box # define fundamental wave mode with physical units

k_axis = np.fft.fftfreq(n_cells) * n_cells * dk
kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing='ij')
k_grid = np.sqrt(kx**2 + ky**2 + kz**2)
k_grid_safe = np.where(k_grid == 0.0, 1.0, k_grid)  # Prevent 0/0 division


# Amplitude calculation
def tophat_window(k, R=8.0):
    x = k * R
    # Handle the limit x -> 0 smoothly to avoid 0/0 errors
    return np.where(x < 1e-3, 1.0, 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3))

# 1. Create a fine 1D array of physical k values for integration
k_integration = np.logspace(-4, 2, 2000) # h/Mpc

# 2. Evaluate your unnormalized spectrum (A=1) along this 1D array
pk_unnorm = eh97_power_spectrum(k_integration, omega_m=omega_0m, omega_b=0.045, h=h, A=1.0, omega_nu=0, n=ns, N_nu=1, T_cmb = T_cmb)

# 3. Integrate to find the unnormalized sigma_8
window = tophat_window(k_integration, R=8.0)
integrand = (k_integration**2) * pk_unnorm * (window**2) / (2.0 * np.pi**2)
sigma8_unnorm_sq = np.trapezoid(integrand, k_integration) #type: ignore

# 4. Compute the true physical normalization constant A !!!!-> For z=0! 
A_true_now = (sigma8_target**2) / sigma8_unnorm_sq

# 5. Divide with growth factor scaling to get amplitude for a <1 !!!
D2 = (growth_factor(a_ini, omega_0m, omega_0k, omega_0lamb) / growth_factor(1, omega_0m, omega_0k, omega_0lamb))**2
A_true_then = A_true_now / D2

#############################

# calculate power spectrum with analytical formula
powerspectrum_EdS = eh97_power_spectrum(
    k_grid,
    h=h,
    omega_m=omega_0m,
    omega_b=omega_0b,
    omega_nu=0,
    n=ns,
    A=A_true_then,
    N_nu=1,
    T_cmb=T_cmb
)


def alpha_norm(n_cells, box_length): #TODO: check if works for different cell units (when 1 cell is not 1 Mpc)
    V = box_length**3
    a = n_cells**3 / np.sqrt(V)
    return a


# generate random numbers and initialize three grids
rng = np.random.default_rng(20260610)
gauss_noise_real = (np.sqrt(powerspectrum_EdS) * rng.normal(0, 1, size=k_grid.shape)) / k_grid_safe**2
gauss_noise_imag = (np.sqrt(powerspectrum_EdS) * rng.normal(0, 1, size=k_grid.shape)) / k_grid_safe**2
c_k = 0.5 * (gauss_noise_real - 1j * gauss_noise_imag)

alpha = alpha_norm(n_cells, L_box)
S_x = np.real(np.fft.ifftn(kx * c_k)) * alpha
S_y = np.real(np.fft.ifftn(ky * c_k)) * alpha
S_z = np.real(np.fft.ifftn(kz * c_k)) * alpha


# now apply Zeldovich Approximation to regular grid 
# 1D unperturbed grid coordinates
q_1d = np.linspace(0, n_cells, n_particles, endpoint=False)

# Compute the momentum scaling factor
a_half = a_ini - 0.5 * delta_a
p_factor = a_half**2 * growth_factor_deriv(a_half, omega_0m=omega_0m, omega_0k=omega_0k, omega_0lamb=omega_0lamb)

positions = []
momenta = []
# Single nested loop to sample the 3D S_x, S_y, S_z grids
for i in range(n_particles):
    for j in range(n_particles):
        for k in range(n_particles):
            # Calculate displaced positions
            x = q_1d[i] + growth_factor(a_ini, omega_0m, omega_0k, omega_0lamb) * S_x[i, j, k]
            y = q_1d[j] + growth_factor(a_ini, omega_0m, omega_0k, omega_0lamb) * S_y[i, j, k]
            z = q_1d[k] + growth_factor(a_ini, omega_0m, omega_0k, omega_0lamb) * S_z[i, j, k]
            positions.append([x, y, z])
            
            # Calculate initial momenta (matching the displacement sign!)
            px = p_factor * S_x[i, j, k]
            py = p_factor * S_y[i, j, k]
            pz = p_factor * S_z[i, j, k]
            momenta.append([px, py, pz])
positions = np.array(positions)
momenta = np.array(momenta)
positions %= n_cells

# Load simulation object
test = Universe(
    n_particles=n_particles, 
    n_cells=n_cells, 
    scale_factor=a_ini, 
    delta_a=delta_a,
    omega_0m=omega_0m,
    omega_0lamb=omega_0lamb
    )


test.positions = positions
test.momenta = momenta
# test.plot()
test.run(steps=900, numba=True, store=False)
test.plot()
test.plot_colour(thickness=1)
Power_Spectrum, k_bins = test._calculate_power_spectrum()

plt.scatter(k_bins, Power_Spectrum, label='Measured P(k)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.legend()
plt.title('Power Spectrum')
plt.grid()
plt.show()

Power_Spectrum2, k_bins2 = test._calculate_power_spectrum(cic_correction=False)

plt.scatter(k_bins2, Power_Spectrum2, label='P(k) without CIC correction')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.title('Power Spectrum')
plt.legend()
plt.grid()
plt.grid()
plt.show()


# A_test = amplitude_physical(
#     a_ini,
#     h,
#     As, 
#     omega_0m,
#     omega_0k,
#     omega_0lamb,
#     ns
# )