### Computational Physics b
# Cosmological Particle-Mesh N-Body simulation

## Overview
This project is an effort to perform a small cosmological simulation as a study of a ΛCDM universe. Starting from initial conditions based on observational data, the simulation calculated the evolution of the particles under the influence of gravity. In particular, the power spectrum is being studied.

The code is structured as a small Python package with separate modules, along with scripts to reproduce the results used in the report.

The code includes animations, highlighting the formation of filaments in the simulated universe.



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

To do: write text here


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
│   └── interactive.py  # Use package interactively
│
├── data/             # Output data (generated)
├── results/          # post-processed results (generated)
└── README.md
```


---

## Reproducing Results

To reproduce the results from the presentation, ...


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