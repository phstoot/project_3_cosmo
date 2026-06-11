import numpy as np
import matplotlib.pyplot as plt
import cosmosim
from cosmosim.simulation import Universe
from cosmosim.utils import *
from astropy import units as u

# This file is free to edit interactively


# k_1d = np.logspace(-3, np.log10(5), 100)

# # calculate power spectrum with analytical formula
# powerspectrum_test = eh97_power_spectrum(
#     k_1d,
#     0.7,
#     1,
#     0.05,
#     0,
#     1,
#     1,
#     1,
#     2.7255 # Fixsen 2009
# )

# Om0= 0.3111
# Ode0= 0.6889
# H0= 67.70*u.km/u.s/u.Mpc
# from scipy import integrate
# def H(a):
#     # Assumes flat Universe
#     return H0*np.sqrt(Om0/a**3.+Ode0)
# def growth_factor2(a):
#     # Direct calculation
#     return (5.*Om0/2.*H(a)/H0\
#         *integrate.quad(lambda ap: 1./ap**3./(H(ap)/H0)**3.,
#                         0.,a)[0]).to_value(u.dimensionless_unscaled)

# a = np.logspace(-4, 0)
# D = growth_factor(a, 1, 0, 0)
# # D_t = growth_factor_deriv(a, 1, 1, 0, 0)
# D_t = growth_factor_deriv(a, 1, 0.31, 0, 0.69)
# # plt.loglog(a, om, label=r'$\Omega_{0, m}$')
# # plt.plot(1/a, growth_factor(a, 1, 0, 0)/a, label=r'approx, $\Omega_{0,m}=1$')
# # plt.plot(1/a, growth_factor(a, 0.31, 0, 0.69)/a, label=r'approx, $\Omega_{0,m}=0.31$, $\Omega_{0, \Lambda}=0.69$')
# # plt.plot(1/a, [growth_factor2(ap)/ap for ap in a], label='numerical')
# plt.plot(1/a, D_t/a, label=r'$\dot{D}+(a)$')
# plt.plot(1/a, a**(-1.5), label='deriv matter only')
# # plt.legend()
# plt.xlabel('a')
# plt.ylabel('D+ / a')
# plt.xscale('log')
# plt.xlim(400, 0.8)
# plt.ylim(0, 10000)
# from matplotlib.ticker import FuncFormatter
# plt.gca().xaxis.set_major_formatter(FuncFormatter(
#                 lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(\
#                     np.maximum(-np.log10(y),0)))).format(y)))
# plt.legend(frameon=True,fontsize=14.,loc='upper left',
#        facecolor='w',edgecolor='w')
# # plt.yscale('log')
# plt.show()

ns = 0.967

#####################################

# # final part: normalizing sigma8

# def tophat_window(k, R=8.0):
#     x = k * R
#     # Handle the limit x -> 0 smoothly to avoid 0/0 errors
#     return np.where(x < 1e-3, 1.0, 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3))

# # 1. Create a fine 1D array of physical k values for integration
# k_integration = np.logspace(-4, 2, 2000) # h/Mpc

# # 2. Evaluate your unnormalized spectrum (A=1) along this 1D array
# pk_unnorm = eh97_power_spectrum(k_integration, omega_m=0.3, omega_b=0.045, h=0.7, A=1.0, omega_nu=0, n=ns, N_nu=1, T_cmb = 2.7255)

# # 3. Integrate to find the unnormalized sigma_8
# window = tophat_window(k_integration, R=8.0)
# integrand = (k_integration**2) * pk_unnorm * (window**2) / (2.0 * np.pi**2)
# sigma8_unnorm_sq = np.trapezoid(integrand, k_integration)

# # 4. Compute the true physical normalization constant A
# # Let's assume a standard observational target like sigma_8 = 0.81
# sigma8_target = 0.81
# A_true = (sigma8_target**2) / sigma8_unnorm_sq








#####################################

def amplitude(a, H0, As, omega_0m, omega_0k, omega_0lamb):
    c = 2.998e5 # km/s
    D2 = (growth_factor(a, omega_0m, omega_0k, omega_0lamb) / growth_factor(1, omega_0m, omega_0k, omega_0lamb))**2
    A = 8 * np.pi**2 / 25 * As * c**4 / (omega_0m**2 * H0**4) * D2
    return A

###########################

As = 2.105e-9



n_particles = 64
n_cells = 128
a_ini = 0.1
delta_a = 0.001

L_box = 32.0  # Mpc/h
dk = 2.0 * np.pi / n_cells

k_axis = np.fft.fftfreq(n_cells) * n_cells * dk
kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing='ij')
k_grid = np.sqrt(kx**2 + ky**2 + kz**2)
k_grid_safe = np.where(k_grid == 0.0, 1.0, k_grid)  # Prevent 0/0 division


# calculate power spectrum with analytical formula
powerspectrum_EdS = eh97_power_spectrum(
    k_grid,
    0.7,
    1,
    0.05,
    0,
    1,
    amplitude(0.1, 67, As, 0.31, 0, 0.68),
    ns,
    2.7255 # Fixsen 2009
)

def alpha_norm(n_cells, box_length):
    V = box_length**3
    a = n_cells**3 / np.sqrt(V)
    return a


# generate random numbers and initialize three grids
rng = np.random.default_rng(20260610)
gauss_noise_real = (np.sqrt(powerspectrum_EdS) * rng.normal(0, 1, size=k_grid.shape)) / k_grid_safe**2
gauss_noise_imag = (np.sqrt(powerspectrum_EdS) * rng.normal(0, 1, size=k_grid.shape)) / k_grid_safe**2
c_k = 0.5 * (gauss_noise_real - 1j * gauss_noise_imag)

alpha = alpha_norm(n_cells, n_cells)
S_x = np.real(np.fft.ifftn(kx * c_k)) * alpha
S_y = np.real(np.fft.ifftn(ky * c_k)) * alpha
S_z = np.real(np.fft.ifftn(kz * c_k)) * alpha


# now apply Zeldovich Approximation to regular grid 

# 1D unperturbed grid coordinates
q_1d = np.linspace(0, n_cells, n_particles, endpoint=False)

# Compute the momentum scaling factor (keeping the physical sign matching positions)
a_half = a_ini - 0.5 * delta_a
p_factor = a_half**(1.5)

positions = []
momenta = []

# Single nested loop to safely sample the 3D S_x, S_y, S_z grids
for i in range(n_particles):
    for j in range(n_particles):
        for k in range(n_particles):
            # 1. Unperturbed particle position on the grid
            q_x = q_1d[i]
            q_y = q_1d[j]
            q_z = q_1d[k]
            
            # 2. Extract the specific 3D displacement value for this voxel
            # S_x, S_y, S_z must be your (32,32,32) arrays from the 3D IFFT
            disp_x = S_x[i, j, k]
            disp_y = S_y[i, j, k]
            disp_z = S_z[i, j, k]
            
            # 3. Calculate displaced positions
            x = q_x + a_ini * disp_x
            y = q_y + a_ini * disp_y
            z = q_z + a_ini * disp_z
            positions.append([x, y, z])
            
            # 4. Calculate initial momenta (matching the displacement sign!)
            px = p_factor * disp_x
            py = p_factor * disp_y
            pz = p_factor * disp_z
            momenta.append([px, py, pz])

# Convert to standard numpy arrays
positions = np.array(positions)
momenta = np.array(momenta)

# Enforce periodic boundaries
positions %= n_cells

# Load into your simulation object
test = Universe(n_particles=n_particles, n_cells=n_cells, scale_factor=a_ini, delta_a=delta_a)
test.positions = positions
test.momenta = momenta
# test.plot()
test.run(steps=900, numba=True, store=False)
test.plot()
test.plot_colour(thickness=1)