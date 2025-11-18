"""Energy functions for OPLSAA forcefield."""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import vmap

from jax_md import space
from jax_md.mm_forcefields import neighbor
from jax_md.mm_forcefields.base import NonbondedOptions, Topology
from jax_md.mm_forcefields.nonbonded.electrostatics import CoulombHandler
from jax_md.mm_forcefields.oplsaa.params import Parameters
from jax_md.partition import NeighborList, NeighborListFns
from jax_md.util import Array


def energy(
  topology: Topology,
  params: Parameters,
  box: Array,
  coulomb: CoulombHandler,
  nb_options: NonbondedOptions = NonbondedOptions(),
) -> Tuple[
  Callable[[Array, NeighborList], dict[str, Array]],
  NeighborListFns,
  space.DisplacementFn,
]:
  """Create OPLSAA energy function.

  Args:
      topology: Molecular topology.
      params: Force field parameters.
      box: Simulation box (scalar or array).
      coulomb: Coulomb energy handler (CutoffCoulomb, EwaldCoulomb, or PMECoulomb).
      nb_options: Nonbonded interaction options.

  Returns:
      energy_fn: Function that computes energy given positions and neighbor list.
          Signature: energy_fn(positions, nlist) -> energy_dict
      neighbor_fn: Function to build/update neighbor lists.
      displacement_fn: Displacement function for the given box.

  Example:
      >>> from jax_md.mm import oplsaa
      >>> from jax_md.mm_forcefields.nonbonded.electrostatics import PMECoulomb
      >>>
      >>> top = oplsaa.Topology(...)
      >>> params = oplsaa.Parameters(...)
      >>> coulomb = PMECoulomb(grid_size=32, alpha=0.3, r_cut=12.0)
      >>> nb_opts = oplsaa.NonbondedOptions(r_cut=12.0, dr_threshold=0.5)
      >>>
      >>> energy_fn, neighbor_fn, disp_fn = oplsaa.energy(
      ...     top, params, box, coulomb, nb_opts
      ... )
      >>>
      >>> # Initialize neighbor list
      >>> nlist = neighbor_fn.allocate(positions)
      >>>
      >>> # Compute energy
      >>> E = energy_fn(positions, nlist)
  """
  displacement_fn, shift_fn = space.periodic(box)

  # TODO fix type error
  neighbor_fn: NeighborListFns = neighbor.create_neighbor_list(
    displacement_fn, box, nb_options.r_cut, nb_options.dr_threshold
  )  # type: ignore

  # Extract parameters for convenience
  bonded = params.bonded
  nonbonded = params.nonbonded

  # Bonded energy functions
  def bond_energy(positions: Array) -> Array:
    """Harmonic bond energy: E = k * (r - r0)^2"""
    if topology.bonds.shape[0] == 0:
      return jnp.array(0.0)

    i, j = topology.bonds[:, 0], topology.bonds[:, 1]
    disp = vmap(displacement_fn)(positions[i], positions[j])
    r = neighbor.safe_norm(disp)
    return jnp.sum(bonded.bond_k * (r - bonded.bond_r0) ** 2)

  def angle_energy(positions: Array) -> Array:
    """Harmonic angle energy: E = k * (theta - theta0)^2"""
    if topology.angles.shape[0] == 0:
      return jnp.array(0.0)

    i, j, k = (
      topology.angles[:, 0],
      topology.angles[:, 1],
      topology.angles[:, 2],
    )
    rij = vmap(displacement_fn)(positions[i], positions[j])
    rkj = vmap(displacement_fn)(positions[k], positions[j])

    # Compute angle
    rij_norm = neighbor.normalize(rij)
    rkj_norm = neighbor.normalize(rkj)
    cos_theta = jnp.sum(rij_norm * rkj_norm, axis=-1)
    theta = neighbor.safe_arccos(cos_theta)

    return jnp.sum(bonded.angle_k * (theta - bonded.angle_theta0) ** 2)

  def torsion_energy(positions: Array) -> Array:
    """Proper dihedral energy: E = k * (1 + cos(n*phi - gamma))"""
    if topology.torsions.shape[0] == 0:
      return jnp.array(0.0)

    idx = topology.torsions

    def compute_dihedral(p0, p1, p2, p3):
      """Compute dihedral angle using cross products."""
      b0 = displacement_fn(p1, p0)
      b1 = displacement_fn(p2, p1)
      b2 = displacement_fn(p3, p2)

      n1 = jnp.cross(b0, b1)
      n2 = jnp.cross(b1, b2)

      n1 = neighbor.normalize(n1)
      n2 = neighbor.normalize(n2)

      cos_phi = jnp.sum(n1 * n2)
      phi = neighbor.safe_arccos(cos_phi)

      return phi

    phi = vmap(compute_dihedral)(
      positions[idx[:, 0]],
      positions[idx[:, 1]],
      positions[idx[:, 2]],
      positions[idx[:, 3]],
    )

    return jnp.sum(
      bonded.torsion_k
      * (1 + jnp.cos(bonded.torsion_n * phi - bonded.torsion_gamma))
    )

  def improper_energy(positions: Array) -> Array:
    """Improper dihedral energy: E = k * (1 + cos(n*psi - gamma))"""
    if topology.impropers.shape[0] == 0:
      return jnp.array(0.0)

    idx = topology.impropers

    def compute_dihedral_signed(p0, p1, p2, p3):
      """Compute signed dihedral angle."""
      b0 = displacement_fn(p1, p0)
      b1 = displacement_fn(p2, p1)
      b2 = displacement_fn(p3, p2)

      b1_norm = neighbor.normalize(b1)

      v = b0 - jnp.sum(b0 * b1_norm, axis=-1, keepdims=True) * b1_norm
      w = b2 - jnp.sum(b2 * b1_norm, axis=-1, keepdims=True) * b1_norm

      x = jnp.sum(v * w, axis=-1)
      y = jnp.sum(jnp.cross(b1_norm, v) * w, axis=-1)

      return jnp.arctan2(y, x)

    psi = vmap(compute_dihedral_signed)(
      positions[idx[:, 0]],
      positions[idx[:, 1]],
      positions[idx[:, 2]],
      positions[idx[:, 3]],
    )

    return jnp.sum(
      bonded.improper_k
      * (1 + jnp.cos(bonded.improper_n * psi - bonded.improper_gamma))
    )

  def lennard_jones_energy(positions: Array, nlist: NeighborList) -> Array:
    """Lennard-Jones 12-6 potential with exclusions and 1-4 scaling."""
    n_atoms = positions.shape[0]
    max_neighbors = nlist.idx.shape[1]

    # Prepare neighbor indices
    idx_i = jnp.repeat(jnp.arange(n_atoms)[:, None], max_neighbors, axis=1)
    idx_j = nlist.idx

    # Valid neighbor mask
    valid = (idx_j >= 0) & (idx_j < n_atoms)
    idx_j_safe = jnp.where(valid, idx_j, 0)
    idx_i_safe = jnp.where(valid, idx_i, 0)

    # Get positions
    ri = positions[idx_i_safe]
    rj = positions[idx_j_safe]

    # Compute distances
    batched_disp = vmap(vmap(displacement_fn, in_axes=(0, 0)), in_axes=(0, 0))
    disp = batched_disp(ri, rj)
    r_sq = jnp.sum(disp**2, axis=-1)
    r_sq_safe = jnp.maximum(r_sq, 1e-4)
    r = jnp.sqrt(r_sq_safe)

    # Get LJ parameters (geometric mean combination rules)
    sigma_i = nonbonded.sigma[idx_i_safe]
    sigma_j = nonbonded.sigma[idx_j_safe]
    epsilon_i = nonbonded.epsilon[idx_i_safe]
    epsilon_j = nonbonded.epsilon[idx_j_safe]

    sigma_ij = jnp.sqrt(sigma_i * sigma_j)
    epsilon_ij = jnp.sqrt(epsilon_i * epsilon_j)

    # LJ potential
    def lj_potential(r_sq, r, sigma, epsilon):
      sr = sigma / jnp.sqrt(r_sq)
      sr6 = sr**6
      lj_val = 4.0 * epsilon * (sr6**2 - sr6)

      if nb_options.use_soft_lj:
        return nb_options.lj_cap * jnp.tanh(lj_val / nb_options.lj_cap)
      elif nb_options.use_shift_lj:
        sr_cut = sigma / nb_options.r_cut
        sr6_cut = sr_cut**6
        lj_cut = 4.0 * epsilon * (sr6_cut**2 - sr6_cut)
        return lj_val - lj_cut
      else:
        return lj_val

    lj_val = lj_potential(r_sq_safe, r, sigma_ij, epsilon_ij)

    # Check exclusions and scaling
    same = idx_i_safe == idx_j_safe
    excluded = vmap(lambda i, j: topology.exclusion_mask[i, j])(
      idx_i_safe, idx_j_safe
    )
    is_14 = vmap(lambda i, j: topology.pair_14_mask[i, j])(
      idx_i_safe, idx_j_safe
    )

    # Include mask
    include = valid & (~same) & (~excluded) & (r < nb_options.r_cut)

    # Apply 1-4 scaling
    scale = jnp.where(is_14, nb_options.scale_14_lj, 1.0)

    # Compute energy
    energy = jnp.where(include, scale * lj_val, 0.0)

    # Factor of 0.5 to avoid double counting
    return 0.5 * jnp.sum(energy)

  def total_energy(positions: Array, nlist: NeighborList) -> dict[str, Array]:
    """Compute total OPLSAA energy.

    Args:
        positions: Atomic positions, shape (n_atoms, 3).
        nlist: Neighbor list.

    Returns:
        Dictionary with energy components:
            - 'bond': Bond energy
            - 'angle': Angle energy
            - 'torsion': Torsion energy
            - 'improper': Improper energy
            - 'lj': Lennard-Jones energy
            - 'coulomb': Coulomb energy
            - 'total': Total energy
    """
    E_bond = bond_energy(positions)
    E_angle = angle_energy(positions)
    E_torsion = torsion_energy(positions)
    E_improper = improper_energy(positions)
    E_lj = lennard_jones_energy(positions, nlist)
    E_coulomb = coulomb.energy(
      positions,
      nonbonded.charges,
      box,
      topology.exclusion_mask,
      topology.pair_14_mask,
      nlist,
      nb_options.scale_14_coul,
    )

    E_total = E_bond + E_angle + E_torsion + E_improper + E_lj + E_coulomb

    return {
      'bond': E_bond,
      'angle': E_angle,
      'torsion': E_torsion,
      'improper': E_improper,
      'lj': E_lj,
      'coulomb': E_coulomb,
      'total': E_total,
    }

  return total_energy, neighbor_fn, displacement_fn
