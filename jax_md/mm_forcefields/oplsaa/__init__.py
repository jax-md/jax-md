"""OPLSAA forcefield for JAX-MD.

This module provides a complete implementation of the OPLSAA forcefield
for molecular dynamics simulations in JAX.

Example usage:
    >>> from jax_md.mm import oplsaa
    >>> from jax_md.mm_forcefields.nonbonded.electrostatics import PMECoulomb
    >>> from jax_md.mm_forcefields.base import NonbondedOptions
    >>> import jax.numpy as jnp
    >>>
    >>> # Load system from files
    >>> positions, topology, parameters = oplsaa.load_charmm_system(
    ...     'molecule.pdb', 'params.prm', 'topology.rtf'
    ... )
    >>>
    >>> # Setup energy function
    >>> box = jnp.array([50.0, 50.0, 50.0])
    >>> coulomb = PMECoulomb(grid_size=32, alpha=0.3, r_cut=12.0)
    >>> nb_options = NonbondedOptions(r_cut=12.0, scale_14_lj=0.5, scale_14_coul=0.5)
    >>>
    >>> energy_fn, neighbor_fn, displacement_fn = oplsaa.energy(
    ...     topology, parameters, box, coulomb, nb_options
    ... )
    >>>
    >>> # Initialize and compute
    >>> nlist = neighbor_fn.allocate(positions)
    >>> E = energy_fn(positions, nlist)
    >>> print(f"Total energy: {E['total']:.3f} kcal/mol")
"""

from jax_md.mm_forcefields.oplsaa.energy import energy
from jax_md.mm_forcefields.oplsaa.topology import (
  create_topology,
  validate_topology,
)
from jax_md.mm_forcefields.oplsaa.params import (
  create_parameters,
  validate_parameters,
  Parameters,
)
from jax_md.mm_forcefields.oplsaa.io import load_charmm_system
from jax_md.mm_forcefields.base import Topology, NonbondedOptions

__all__ = [
  'energy',
  'Topology',
  'create_topology',
  'validate_topology',
  'Parameters',
  'create_parameters',
  'validate_parameters',
  'NonbondedOptions',
  'load_charmm_system',
]
