from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import jax.numpy as jnp
from jax import vmap

from jax_md.util import safe_norm
from jax_md.mm_forcefields.martini.top_file_parser import (
  GromacsTopFile,
  LinearVirtualSite,
  NAtomCOMVirtualSite,
)


def _parse_vsites_from_topology(
  top: GromacsTopFile,
  excl_pairs: List[Tuple[int, int]],
):
  """
  Parse virtual sites from a GROMACS topology and build an application function.

  This function flattens and resolves all linear virtual sites across all molecules in the
  topology. It also accumulates excluded bonded pairs involving virtual sites.

  Args:
    top: Parsed GROMACS topology containing molecules, atom types, and virtual sites.
    excl_pairs: List of excluded atom pairs to be updated in-place with
      virtual-site-related exclusions.

  Returns:
    A tuple of:
      - apply_vsites: Callable that constructs virtual site positions from coordinates.
      - excl_pairs: Updated exclusion pair list including virtual-site exclusions.
  """
  masses: List[float] = []
  base_atom_index = 0

  linear_vsites: Dict[int, Tuple] = {}
  oop_vsites: Dict[int, Tuple] = {}
  all_three_fd_vsites: Dict[int, Tuple] = {}
  all_two_fd_vsites: Dict[int, Tuple] = {}

  for mol in top._molecules:
    mol_type = top._moleculeTypes[mol.name]
    n_atoms = len(mol_type.atoms)

    for _ in range(mol.count):
      offset = base_atom_index - 1

      for atom in mol_type.atoms:
        atom_type = top._atomTypes[atom.type]
        mass = atom.mass if atom.mass is not None else atom_type.mass
        masses.append(mass)

      for com_vsite in mol_type.n_com_vsites:
        mol_type.linear_vsites[com_vsite.index] = convert_com_to_linear_vsite(
          com_vsite, offset, masses
        )

      flattened_vsites = dict(mol_type.linear_vsites)

      for linear_vsite_idx in mol_type.linear_vsites:
        linear_vsite = flatten_vsite(
          flattened_vsites[linear_vsite_idx], flattened_vsites
        )
        linear_vsites[linear_vsite_idx + offset] = (linear_vsite, offset)
        if len(linear_vsite.atom_weights) == 1:
          excl_pairs.append(
            (
              linear_vsite_idx + offset,
              list(linear_vsite.atom_weights.keys())[0] + offset,
            )
          )

      for two_fd_vsite in mol_type.two_fd_vsites:
        all_two_fd_vsites[two_fd_vsite.index + offset] = (
          two_fd_vsite,
          offset,
        )

      for three_fd_vsite in mol_type.three_fd_vsites:
        all_three_fd_vsites[three_fd_vsite.index + offset] = (
          three_fd_vsite,
          offset,
        )

      for three_out_vsite in mol_type.three_out_vsites:
        oop_vsites[three_out_vsite.index + offset] = (
          three_out_vsite,
          offset,
        )

      base_atom_index += n_atoms

  return (
    _make_apply_vsites(
      linear_vsites=linear_vsites,
      oop_vsites=oop_vsites,
      all_three_fd_vsites=all_three_fd_vsites,
      all_two_fd_vsites=all_two_fd_vsites,
    ),
    excl_pairs,
  )


def convert_com_to_linear_vsite(
  com_site: NAtomCOMVirtualSite, offset: int, masses: List[float]
):
  """
  Convert a center-of-mass virtual site definition into a linear virtual site.

  Args:
    com_site: Center-of-mass virtual site definition from topology.
    offset: Global atom index offset for the current molecule instance.
    masses: List of atomic masses indexed by global atom index.

  Returns:
    A LinearVirtualSite with normalized mass-based weights.
  """
  atom_masses = [masses[atom + offset] for atom in com_site.atoms]
  total_mass = sum(atom_masses)
  weights = [m / total_mass for m in atom_masses]
  return LinearVirtualSite(
    atom_weights={
      com_site.atoms[i]: weights[i] for i in range(len(com_site.atoms))
    }
  )


def flatten_vsite(
  linear_site: LinearVirtualSite,
  all_sites: Dict[int, LinearVirtualSite],
) -> LinearVirtualSite:
  """
  Recursively flatten nested linear virtual site definitions into atomic weights.

  Some virtual sites depend on other virtual sites. This function resolves
  the full dependency chain and expresses the virtual site purely in terms of
  real atoms.

  Args:
    linear_site: A virtual site potentially defined in terms of other sites.
    all_sites: Mapping from site index to LinearVirtualSite definitions.

  Returns:
    A fully expanded LinearVirtualSite depending only on real atom indices.
  """
  atom_weights: Dict[int, float] = {}
  for index, weight in linear_site.atom_weights.items():
    if index in all_sites:
      flattened = flatten_vsite(all_sites[index], all_sites)
      all_sites[index] = flattened
      for f_atom, f_weight in flattened.atom_weights.items():
        atom_weights[f_atom] = atom_weights.get(f_atom, 0.0) + weight * f_weight
    else:
      atom_weights[index] = atom_weights.get(index, 0.0) + weight
  return LinearVirtualSite(atom_weights)


def _make_apply_vsites(
  linear_vsites: Dict,
  oop_vsites: Dict,
  all_three_fd_vsites: Dict,
  all_two_fd_vsites: Dict,
) -> Callable:
  """
  Construct a function that applies all virtual site types to positions.

  This builds JAX-friendly static arrays for all virtual site definitions and
  returns a single function that updates particle coordinates by evaluating:
    - Linear (mass-weighted) virtual sites
    - Out-of-plane virtual sites
    - 3-point distance-form virtual sites
    - 2-point distance-form virtual sites

  Args:
    linear_vsites: Mapping of linear virtual sites and offsets.
    oop_vsites: Out-of-plane virtual site definitions.
    all_three_fd_vsites: 3-body virtual sites.
    all_two_fd_vsites: 2-body virtual sites.

  Returns:
    apply_vsites function with signature:
      (positions, displacement_fn, shift_fn) -> updated positions
  """
  v_idx, f_idx, wts = build_linear_vsite_arrays(linear_vsites)

  oop_data = {
    'vsite_indices': jnp.array([k for k in oop_vsites.keys()]),
    'all_atom_i': jnp.array(
      [s.atom_i + offset for (s, offset) in oop_vsites.values()]
    ),
    'all_atom_j': jnp.array(
      [s.atom_j + offset for (s, offset) in oop_vsites.values()]
    ),
    'all_atom_k': jnp.array(
      [s.atom_k + offset for (s, offset) in oop_vsites.values()]
    ),
    'a': jnp.array([s.a for (s, offset) in oop_vsites.values()]),
    'b': jnp.array([s.b for (s, offset) in oop_vsites.values()]),
    'c': jnp.array([s.c for (s, offset) in oop_vsites.values()]),
  }

  three_fd_data = {
    'vsite_indices': jnp.array([k for k in all_three_fd_vsites.keys()]),
    'all_atom_i': jnp.array(
      [s.atom_i + offset for (s, offset) in all_three_fd_vsites.values()]
    ),
    'all_atom_j': jnp.array(
      [s.atom_j + offset for (s, offset) in all_three_fd_vsites.values()]
    ),
    'all_atom_k': jnp.array(
      [s.atom_k + offset for (s, offset) in all_three_fd_vsites.values()]
    ),
    'a': jnp.array([s.a for (s, offset) in all_three_fd_vsites.values()]),
    'd': jnp.array([s.d for (s, offset) in all_three_fd_vsites.values()]),
  }

  two_fd_data = {
    'vsite_indices': jnp.array([k for k in all_two_fd_vsites.keys()]),
    'all_atom_i': jnp.array(
      [s.atom_i + offset for (s, offset) in all_two_fd_vsites.values()]
    ),
    'all_atom_j': jnp.array(
      [s.atom_j + offset for (s, offset) in all_two_fd_vsites.values()]
    ),
    'd': jnp.array([s.d for (s, offset) in all_two_fd_vsites.values()]),
  }

  def apply_vsites(positions, displacement_fn, shift_fn):
    if v_idx.shape[0] > 0:
      positions = apply_linear_vsites(positions, v_idx, f_idx, wts, displacement_fn, shift_fn)
    if oop_vsites:
      positions = apply_out_of_plane_vsites(
        positions,
        displacement_fn=displacement_fn,
        shift_fn=shift_fn,
        **oop_data,
      )
    if all_two_fd_vsites:
      positions = apply_two_fd_vsites(
        positions,
        displacement_fn=displacement_fn,
        shift_fn=shift_fn,
        **two_fd_data,
      )
    if all_three_fd_vsites:
      positions = apply_three_fd_vsites(
        positions,
        displacement_fn=displacement_fn,
        shift_fn=shift_fn,
        **three_fd_data,
      )
    return positions

  return apply_vsites


def build_linear_vsite_arrays(flat_vsites: dict, max_contributors=None):
  """
  Convert linear virtual site definitions into padded JAX arrays.

  This formats variable-length atom-weight relationships into fixed-size
  arrays suitable for JAX vectorization.

  Args:
    flat_vsites: Mapping of virtual site index to (LinearVirtualSite, offset).
    max_contributors: Maximum number of atoms contributing to any site.
      If None, inferred from input.

  Returns:
    Tuple of:
      - vsite_indices: (V,) indices of virtual sites
      - from_indices: (V, K) contributing atom indices (padded)
      - weights: (V, K) contribution weights (padded)
  """
  if not flat_vsites:
    return (
      jnp.zeros((0,), dtype=jnp.int32),
      jnp.zeros((0,), dtype=jnp.int32),
      jnp.zeros((0,), dtype=jnp.float64),
    )
  if max_contributors is None:
    max_contributors = max(
      len(site.atom_weights) for (site, _) in flat_vsites.values()
    )
  vsite_indices, from_indices, weights = [], [], []
  for vsite_idx, (site, offset) in flat_vsites.items():
    vsite_indices.append(vsite_idx)
    atoms = [atom + offset for atom in site.atom_weights.keys()]
    ws = list(site.atom_weights.values())
    pad = max_contributors - len(atoms)
    from_indices.append(atoms + [0] * pad)
    weights.append(ws + [0.0] * pad)
  return (
    jnp.array(vsite_indices),
    jnp.array(from_indices),
    jnp.array(weights),
  )


def apply_linear_vsites(positions, vsite_indices, from_indices, weights, displacement_fn, shift_fn):
  """
  Compute and apply linear (mass-weighted) virtual site positions.

  Each virtual site is defined as a weighted sum of contributing atoms.

  Args:
    positions: Atomic positions, shape (n_atoms, 3).
    vsite_indices: Indices of virtual sites to update, shape (V,).
    from_indices: Up to K contributing atom indices per virtual site,
          shape (V, K) (pad with 0 and use weight=0 for unused slots)
    weights: Contribution weights, shape (V, K).

  Returns:
    Updated positions with virtual sites overwritten.
  """
  # contributing = positions[from_indices]
  # new_positions = jnp.sum(weights[:, :, None] * contributing, axis=1)
  # return positions.at[vsite_indices].set(new_positions)

  ref_idx = from_indices[:, 0]          # (V,)  anchor atom per vsite
  r_ref = positions[ref_idx]            # (V, 3)
  contributing = positions[from_indices]  # (V, K, 3)

  disp = vmap(vmap(displacement_fn, in_axes=(0, None)), in_axes=(0, 0))(
      contributing, r_ref
  )  # (V, K, 3)

  delta = jnp.sum(weights[:, :, None] * disp, axis=1)  # (V, 3)

  new_positions = shift_fn(r_ref, delta)
  return positions.at[vsite_indices].set(new_positions)


def apply_out_of_plane_vsites(
  positions,
  vsite_indices,
  all_atom_i,
  all_atom_j,
  all_atom_k,
  a,
  b,
  c,
  displacement_fn,
  shift_fn,
):
  """
  Compute out-of-plane virtual sites using a 3-atom geometric construction.

  The virtual site is defined as:
    r = r1 + a*(r2-r1) + b*(r3-r1) + c*((r2-r1) x (r3-r1))

  Args:
    positions: Atomic positions, shape (n_atoms, 3).
    vsite_indices: Indices of virtual sites to update.
    all_atom_i: Central atom indices.
    all_atom_j: First reference atom indices.
    all_atom_k: Second reference atom indices.
    a: Linear coefficient for r2 contribution.
    b: Linear coefficient for r3 contribution.
    c: Cross-product coefficient.
    displacement_fn: Function computing minimum-image displacement.
    shift_fn: Function for calculating new positions.

  Returns:
    Updated positions with out-of-plane virtual sites applied.
  """
  r1 = positions[all_atom_i]
  r2 = positions[all_atom_j]
  r3 = positions[all_atom_k]

  d12 = vmap(displacement_fn)(r2, r1)
  d13 = vmap(displacement_fn)(r3, r1)
  cross = jnp.cross(d12, d13)
  delta = a[:, None] * d12 + b[:, None] * d13 + c[:, None] * cross

  new_pos = shift_fn(r1, delta)

  return positions.at[vsite_indices].set(new_pos)


def apply_three_fd_vsites(
  positions,
  vsite_indices,
  all_atom_i,
  all_atom_j,
  all_atom_k,
  a,
  d,
  displacement_fn,
  shift_fn,
):
  """
  Compute 3-point distance-form virtual sites (GROMACS type 3fd).

  The site is placed at a fixed distance along a weighted direction:
    r = r1 + d * normalize(a*(r2-r1) + (1-a)*(r3-r1))

  Args:
    positions: Atomic positions, shape (n_atoms, 3).
    vsite_indices: Virtual site indices.
    all_atom_i: First atom indices.
    all_atom_j: Second atom indices.
    all_atom_k: Third atom indices.
    a: Mixing coefficient between directions.
    d: Distance from reference atom.
    displacement_fn: Minimum-image displacement function.
    shift_fn: Function for calculating new positions.

  Returns:
    Updated positions with 3fd virtual sites applied.
  """
  r1 = positions[all_atom_i]
  r2 = positions[all_atom_j]
  r3 = positions[all_atom_k]
  d12 = vmap(displacement_fn)(r2, r1)
  d13 = vmap(displacement_fn)(r3, r1)
  direction = (1 - a[:, None]) * d12 + a[:, None] * d13
  unit_dir = direction / safe_norm(direction, axis=-1, keepdims=True)
  delta = d[:, None] * unit_dir
  new_pos = shift_fn(r1, delta)
  return positions.at[vsite_indices].set(new_pos)


def apply_two_fd_vsites(
  positions, vsite_indices, all_atom_i, all_atom_j, d, displacement_fn, shift_fn
):
  """
  Compute 2-point distance-form virtual sites (GROMACS type 2fd).

  The virtual site is placed along a bond direction:
    r = r1 + d * normalize(r2 - r1)

  Args:
    positions: Atomic positions, shape (n_atoms, 3).
    vsite_indices: Virtual site indices to update.
    all_atom_i: Reference atom indices.
    all_atom_j: Target atom indices.
    d: Distance from reference atom.
    displacement_fn: Minimum-image displacement function.
    shift_fn: Function for calculating new positions.

  Returns:
    Updated positions with 2fd virtual sites applied.
  """
  r1 = positions[all_atom_i]
  r2 = positions[all_atom_j]

  direction = vmap(displacement_fn)(r2, r1)
  unit_dir = direction / safe_norm(direction, axis=-1, keepdims=True)
  delta = d[:, None] * unit_dir
  new_pos = shift_fn(r1, delta)
  return positions.at[vsite_indices].set(new_pos)
