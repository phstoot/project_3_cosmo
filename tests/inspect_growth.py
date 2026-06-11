import numpy as np
import matplotlib.pyplot as plt
import cosmosim
from cosmosim.simulation import Universe
from cosmosim.utils import *
from astropy import units as u
from scipy.differentiate import derivative

# inspect growth factor formula's for checking

Om0= 0.3111
Ode0= 0.6889
H0= 67.70*u.km/u.s/u.Mpc #type: ignore
from scipy import integrate
def H(a):
    # Assumes flat Universe
    return H0*np.sqrt(Om0/a**3.+Ode0)
def growth_factor2(a):
    # Direct calculation
    return (5.*Om0/2.*H(a)/H0\
        *integrate.quad(lambda ap: 1./ap**3./(H(ap)/H0)**3.,
                        0.,a)[0]).to_value(u.dimensionless_unscaled)

a = np.logspace(-4, 0)
D = growth_factor(a, 1, 0, 0)
# plt.loglog(a, om, label=r'$\Omega_{0, m}$')
plt.plot(1/a, growth_factor(a, 1, 0, 0)/a, label=r'approx, $\Omega_{0,m}=1$')
plt.plot(1/a, growth_factor(a, 0.31, 0, 0.69)/a, label=r'approx, $\Omega_{0,m}=0.31$, $\Omega_{0, \Lambda}=0.69$')
plt.plot(1/a, [growth_factor2(ap)/ap for ap in a], label='numerical')
plt.xlabel('1/a (= z + 1)')
plt.ylabel('D+(a) / a')
plt.xscale('log')
plt.xlim(400, 0.8)
plt.ylim(0, 1.2)
from matplotlib.ticker import FuncFormatter
plt.gca().xaxis.set_major_formatter(FuncFormatter(
                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(\
                    np.maximum(-np.log10(y),0)))).format(y)))
plt.legend(frameon=True,fontsize=14.,loc='lower left',
       facecolor='w',edgecolor='w')
plt.show()




D_deriv_de = growth_factor_deriv(a, 0.31, 0, 0.69)
D_deriv_0 = growth_factor_deriv(a, 1, 0, 0)
plt.axhline(1, color='black', linestyle='--', label=r'$\dot{D}+(a)$ = 1, matter only')
plt.plot(a, D_deriv_de, label=r'$\dot{D}+(a)$ deriv, with DE')
plt.plot(a, D_deriv_0, label=r'$\dot{D}+(a)$ deriv, matter only')
plt.xlabel('a')
plt.ylabel('dD/dt')
plt.xscale('log')
from matplotlib.ticker import FuncFormatter
plt.gca().xaxis.set_major_formatter(FuncFormatter(
                lambda y,pos: (r'${{:.{:1d}f}}$'.format(int(\
                    np.maximum(-np.log10(y),0)))).format(y)))
plt.legend(frameon=True,fontsize=14.,loc='upper left',
       facecolor='w',edgecolor='w')
plt.show()