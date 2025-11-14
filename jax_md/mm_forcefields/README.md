# Molecular Mechanics (`jax_md.mm_forcefields`)

Utilities for building molecular mechanics (MM) force fields live under `jax_md.mm_forceilds`. 

## Package Guide

### `base.py`
- `Topology`: Connectivity information (bonds, angles, torsions, impropers) plus dense exclusion and 1–4 scaling masks that the rest of the stack consumes.  The optional `molecule_id` array lets you scope exclusions/scalings to intramolecular pairs.
- `BondedParameters` / `NonbondedParameters`: Per-interaction parameter bundles kept in kcal/mol and Å units to match common biomolecular force fields.
- `NonbondedOptions`: Runtime knobs for Lennard-Jones variants (soft-core capping or shifted potential), cutoff/skin distances, and 1–4 scaling factors shared across Coulomb and LJ terms.

### `neighbor.py`
- `create_neighbor_list(...)`: Thin wrapper over `jax_md.partition.neighbor_list` that defaults to the masked format typically expected for MM calculations.
- `make_exclusion_mask(...)` and `make_14_table(...)`: Construct dense boolean lookups for bonded/angle exclusions and 1–4 pairs; both respect `molecule_id` to avoid suppressing intermolecular interactions.
- `safe_norm`, `safe_arccos`, `normalize`: Numerically stable primitives reused by the bonded energy code to keep gradients finite.

### `io/`
- `io/charmm.py` implements lightweight parsers for CHARMM RTF/PRM files and a minimal PDB reader (`parse_pdb_simple`). This code works for now but is not very maintainble. Other solutions should be coinsidered.
- `oplsaa/io.py` stitches pieces together: it infers angles/torsions from the RTF bonds, maps parameters from atom types, converts CHARMM LJ radii (Rmin/2) into $\sigma$, and returns `(positions, Topology, Parameters)` types ready for simulation.

### `nonbonded/`
All Coulomb handlers share the same interface (`CoulombHandler.energy(positions, charges, box, exclusion_mask, pair_14_mask, nlist, scale_14)`), work in kcal/mol, and honor the dense masks emitted by `Topology`.

- `CutoffCoulomb`: Direct-space 1/r with optional `erfc` damping; ideal for short cutoffs or implicit solvent work.
- `EwaldCoulomb`: Real + reciprocal Ewald sums with configurable damping (`alpha`), k-space resolution (`kmax`), and cutoff.
- `PMECoulomb`: Particle-Mesh Ewald that deposits charges on a regular grid (simple 2-point linear assignment) and uses FFTs for the reciprocal term.
- `COULOMB_CONSTANT`: Shared `332.06371 kcal·Å/(mol·e²)` conversion factor.

### `oplsaa/`
The OPLS-AA implementation is split into composable layers:

- `topology.py`: Creates validated `Topology` objects and auto-builds exclusion/1–4 masks via the neighbor helpers.
- `params.py`: Wraps bonded/nonbonded arrays into a `Parameters` NamedTuple and validates shapes.
- `energy.py`: Canonical energy assembly that wires bonded terms, LJ, and a supplied Coulomb handler together.  Returns `(energy_fn, neighbor_fn, displacement_fn)`, so the caller can manage neighbor list updates externally.
- `energy_oplsaa.py` & `modular_Ewald.py`: Legacy/experimental optimized variants kept for benchmarking; they expose similar functionality but are not used by the public API.
- `__init__.py`: Public surface that re-exports `energy`, topology/parameter helpers, `NonbondedOptions`, and `load_charmm_system`.
- `README.md`: Force-field specific notes; complements, rather than duplicates, this summary.

## Typical Workflow

```python
import jax.numpy as jnp
from jax_md.mm import oplsaa
from jax_md.mm_forcefields.nonbonded.electrostatics import PMECoulomb

# 1. Parse input files (CHARMM-style in this case)
positions, topology, params = oplsaa.load_charmm_system(
    "molecule.pdb", "forcefield.prm", "topology.rtf"
)

# 2. Choose long-range electrostatics + LJ configuration
coulomb = PMECoulomb(grid_size=48, alpha=0.28, r_cut=11.0)
nb_options = oplsaa.NonbondedOptions(
    r_cut=11.0,
    dr_threshold=0.6,
    use_shift_lj=True,
    scale_14_lj=0.5,
    scale_14_coul=0.5,
)

# 3. Build energy + neighbor functions
energy_fn, neighbor_fn, displacement_fn = oplsaa.energy(
    topology, params, box=jnp.array([40.0, 40.0, 40.0]), coulomb=coulomb, nb_options=nb_options
)

# 4. Allocate/update neighbor list externally
nlist = neighbor_fn.allocate(jnp.asarray(positions))
energies = energy_fn(jnp.asarray(positions), nlist)
print(float(energies["total"]))
```

`energy_fn` returns a dict with individual components (`bond`, `angle`, `torsion`, `improper`, `lj`, `coulomb`, `total`)