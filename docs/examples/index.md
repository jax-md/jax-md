# Examples

Executable examples demonstrating JAX MD features. Each example is a
self-contained Python script that can be run locally or viewed as a
rendered notebook here.

---

## Simulations

- [NVE](nve_simulation.ipynb) -- Constant energy simulation with Verlet integration
- [NVT](nvt_simulation.ipynb) -- Constant temperature with Nose-Hoover thermostat
- [NPT](npt_simulation.ipynb) -- Constant pressure and temperature
- [NVE + Neighbor Lists](nve_neighbor_list.ipynb) -- Efficient short-range interactions
- [FIRE Minimization](fire_minimization.ipynb) -- Energy minimization via FIRE algorithm
- [ReaxFF NVE](reaxff_nve_simulation.ipynb) -- Reactive force field simulation
- [Multi-Image NL](custom_nl.ipynb) -- Neighbor lists for small periodic boxes

## Neural Network Potentials

- [Graph Network](neural_networks.ipynb) -- Invariant message-passing GNN on Silicon DFT data
- [NequIP](equivariant_neural_networks.ipynb) -- E(3)-equivariant GNN with tensor products

## Units

- [NVE](units/nve_si_sw.ipynb) -- Stillinger-Weber Silicon, constant energy
- [NVT](units/nvt_si_sw.ipynb) -- Nose-Hoover thermostat
- [NPT](units/npt_si_sw.ipynb) -- Nose-Hoover barostat
- [NVK](units/nvk_si_sw.ipynb) -- Constant kinetic energy
- [Berendsen](units/berendsen_si_sw.ipynb) -- Weak coupling thermostat
- [Velocity Rescaling](units/velocity_rescale_si_sw.ipynb) -- Stochastic rescaling
- [CSVR](units/csvr_si_sw.ipynb) -- Canonical sampling (Bussi)
- [NPT Melting](units/npt_melting_si_sw.ipynb) -- High-temperature melting simulation
