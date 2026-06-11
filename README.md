### Computational Physics b
# Cosmological Particle-Mesh N-Body simulation

## Overview
This project is an effort to perform a small cosmological simulation as a study of a ΛCDM universe. Starting from initial conditions based on observational data, the simulation calculated the evolution of the particles under the influence of gravity. In particular, the power spectrum is being studied.

The code is structured as a small Python package with separate modules, along with scripts to reproduce the results used in the report.

The code includes animations, highlighting the formation of filaments in the simulated universe. Some exemplary gifs are store under animation/.



---

## Installation

### For development (collaborators)

Install in editable mode inside your dedicated environment:

```bash
pip install -e .
```

### For assessment / general use

Install in a your dedicated environment:

```bash
pip install .
```

---

## Model and Simulation Details

For a generic analysis of the ΛCDM universe, we simulate the universe with the following choices:

* Particle Number per dim: ( $n_p = 32$ )
* Lattice size: ( 64 $\times$ 64 $\times$ 64 )
* redshift: ( $z = 1$ )
* Initial Scale Factor: ( $a = 0.1$ )
* Matter fraction: ( $\Omega_{0m} = 0.31$ )
* Baryon fraction: ( $\Omega_{0b} = 0.045$ )
* Dark Energy fraction: ( $\Omega_{0lamb} = 0.69$ )
* Curvature: ( $\Omega_{0k} = 0$ )
* "Little h": ( $h = 0.67$ )
* Boundary conditions: periodic
* Density Interpolation Method: Cloud in Cell
* Spin representation: angles ( $\theta \in [0, 2\pi]$ )


### Implementation and reproducability notes

* The simulation is accelerated using **Numba JIT compilation**
* The density interpolation necessary for the Poisson solver is done with Cloud in Cell.
* One can store simulation parameters like density, scale_factor by choosing store=True in run().
* By default, particles are distributed evenly over the grid and displaced by a random step. However, one can also manually implement the initial parameters, as was done for the ΛCDM model.
* The outcome as one approaches modern age is highly sensitive to the initial conditions, even regarding the general form of clustering. Fine tuning is therefore required to properly replicated the data.  
The universe is initialized 


---

## Project Structure

```text
project/
│
├── src/cosmosim
│   ├── simulation.py   # The main simulation file
│   └── utils.py        # Utilities
│
├── scripts/
│   └── 1D_sinewavetest.py  # Use package interactively
│   └── analyse_data.py  # Use package interactively
│   └── example_amplitudes.py  # Use package interactively
│   └── example_eds.py  # Use package interactively
│   └── example_LambdaCDM.py  # Use package interactively
│   └── example_random.py  # Use package interactively
│
├── data/             # Output data (generated)
├── results/          # post-processed results (generated)
└── README.md
```


---

## Reproducing Results
To reproduce the results from the presentation, here are some files to run:

### 1. Basic Results
1D_sinewavetest.py
analyse_data.py: Analyze data from longer data run.
example_amplitudes.py
example_eds.py
example_LambdaCDM.py
example_random.py


### 2. Conduct a larger simultion of the LambdaCDM model
Please keep in mind that this might require some time.


---

## General Usage

The package can also be used outside of the pre-made scripts, by simply importing the module when it is installed in your environment. See the simple example below:

### Example

```python

from cosmosim.simulation import Universe

# Initialize simulation
sim = Universe(n_particles=32, n_cells=64)

# Run simulation
sim.run(steps=900, store=True)

# Plot animated version of the evolution, quicken animation by altering batch_interval
sim.plot_animation(three_D=True, gridoff=True, batch_interval=5) # 3D with colours for density
sim.plot_animation() # 2D slice

# Access recorded data
density_data = sim.density_hist
positions_data = sim.positions_hist

# Plot current state of the universe (slice and 3D distribution)
sim.plot()
sim.plot_colour()

# Observables (Power Spectrum)
Power_spectrum, k_bins = test._calculate_power_spectrum()


```
From this point on more supplemental analysis (mean, standard deviation) can be set up by the user in scripts.

---

## Authors

Nils Thiessen, 
Philip Stoot

---