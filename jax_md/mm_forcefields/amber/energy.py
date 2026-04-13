"""
This module provides an AMBER-like energy-function generator that builds a
JAX-compatible potential energy function along with a neighbor-list function.

The implementation is intended to be interoperate with systems converted from
OpenMM through the tools available in openmm.py.

Conventions (unless otherwise stated):
  - Positions are in A, or fractional coordinates in [0, 1) when using
    space.periodic_general(box, fractional_coordinates=True)
  - Energies are in kcal/mol
  - Charges are in units of the elementary charge (e)
  - Angles/dihedrals are in radians
"""

from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
import numpy as onp
import scipy
from jax.scipy.special import erfc

from jax_md import dataclasses, partition, smap, space, util
from jax_md.mm_forcefields import neighbor
from jax_md.mm_forcefields.base import (
  NonbondedOptions,
  Topology,
  combine_berthelot,
  combine_lorentz,
  combine_product,
  compute_angle,
  compute_dihedral,
  force_switch,
  hard_cutoff,
)
from jax_md.mm_forcefields.nonbonded.electrostatics import (
  CoulombHandler,
  CutoffCoulomb,
  EwaldCoulomb,
  PMECoulomb,
)
from jax_md.mm_forcefields.oplsaa.params import Parameters

# Types

f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat

COULOMB_CONSTANT = 332.06371  # kcal*A/(mol*e^2)


# TODO rethink the cleanest way to format this with some enums
# TODO also move this somewhere more suitable
@dataclasses.dataclass
class FEOptions(object):
  """
  Free-energy (alchemical) options.

  This is currently unstable but will contain the options needed to
  perform coupling/decoupling in FE simulations. The exact choice of
  topology scheme and conventions for alchemical transformations
  are undecided.

  Attributes:
    vdw_scaling: Optional string describing the vdW scaling scheme
    coul_scaling: Optional string describing the Coulomb scaling scheme
    ti_mask: Mask values for thermodynamic integration regions
    sc_mask: Mask values marking softcore/alchemical atoms
  """

  vdw_scaling: Optional[str] = None
  coul_scaling: Optional[str] = None
  ti_mask: Array | float = 1.0
  sc_mask: Array | float = 1.0
  softcore_lrc: bool = False
  softcore_lrc_points: int = 96


def space_selector(
  nb_options: NonbondedOptions,
  box_vectors: Array | float | None,
) -> tuple[space.DisplacementFn, space.ShiftFn]:
  """
  Pick the displacement/shift functions based on NB/PBC settings
  """
  use_pbc = nb_options.use_pbc
  use_periodic_general = nb_options.use_periodic_general
  fractional_coordinates = nb_options.fractional_coordinates
  wrapped_space = nb_options.wrapped_space

  if not use_pbc:
    disp_fn, shift_fn = space.free()
  elif use_periodic_general:
    disp_fn, shift_fn = space.periodic_general(
      box_vectors,
      fractional_coordinates=fractional_coordinates,
      wrapped=wrapped_space,
    )
  else:
    disp_fn, shift_fn = space.periodic(box_vectors, wrapped=wrapped_space)
  return disp_fn, shift_fn


def _create_dense_mask(
  n_atoms: int, exc_pairs: Array
) -> Callable[[Array], Array]:
  receivers = jnp.concatenate((exc_pairs[:, 0], exc_pairs[:, 1]))
  senders = jnp.concatenate((exc_pairs[:, 1], exc_pairs[:, 0]))
  idx = jnp.argsort(senders)
  receivers = receivers[idx]
  senders = senders[idx]

  N = n_atoms
  count = jax.ops.segment_sum(jnp.ones(len(receivers), jnp.int32), receivers, N)
  max_count = jnp.max(count)
  offset = jnp.tile(jnp.arange(max_count), N)[: len(senders)]
  hashes = senders * max_count + offset
  dense_idx = N * jnp.ones((N * max_count,), jnp.int32)
  exclusions_dense = dense_idx.at[hashes].set(receivers).reshape((N, max_count))

  def mask_fn(idx: Array) -> Array:
    """
    Mask neighbor indices that are excluded or treated as 1-4 pairs

    This function is passed to the neighbor-list builder to remove pairs
    that should not appear in the main nonbonded neighbor list

    Args:
      idx: Dense candidate neighbor index array of shape (N, M)

    Returns:
      An array of the same shape where excluded entries are replaced with
      topology.n_atoms
    """
    # TODO unique is assumed here, not clear from the algorithm if the junk fill values matter
    idx = jax.vmap(
      lambda idx_r, mask_r: jnp.where(jnp.isin(idx_r, mask_r), n_atoms, idx_r),
      in_axes=(0, 0),
    )(idx, exclusions_dense)
    return idx

  return mask_fn


def _create_sparse_mask(
  n_atoms: int, exc_pairs: Array
) -> Callable[[Array], Array]:
  receivers = jnp.concatenate((exc_pairs[:, 0], exc_pairs[:, 1]))
  senders = jnp.concatenate((exc_pairs[:, 1], exc_pairs[:, 0]))
  idx = jnp.argsort(senders)
  receivers = receivers[idx]
  senders = senders[idx]

  N = n_atoms
  count = jax.ops.segment_sum(jnp.ones(len(receivers), jnp.int32), receivers, N)
  max_count = jnp.max(count)
  offset = jnp.tile(jnp.arange(max_count), N)[: len(senders)]
  hashes = senders * max_count + offset
  dense_idx = N * jnp.ones((N * max_count,), jnp.int32)
  exclusions_dense = dense_idx.at[hashes].set(receivers).reshape((N, max_count))

  exclusions_dense_sorted = jnp.sort(exclusions_dense, axis=1)

  def mask_fn(idx: Array) -> Array:
    receiver = idx[0]
    sender = idx[1]

    mask_r = exclusions_dense_sorted[sender]

    pos = jax.vmap(lambda row, x: jnp.searchsorted(row, x))(mask_r, receiver)
    pos = jnp.minimum(pos, max_count - jnp.int32(1))
    is_excluded = (
      jnp.take_along_axis(mask_r, pos[:, None], axis=1)[:, 0] == receiver
    )

    receiver_out = jnp.where(is_excluded, N, receiver)
    sender_out = jnp.where(is_excluded, N, sender)
    return jnp.stack((receiver_out, sender_out), axis=0)

  return mask_fn


def energy(
  params: Parameters,
  topology: Topology,
  box_vectors: Array | float | None,
  coulomb_options: CoulombHandler = CoulombHandler(),
  nb_options: NonbondedOptions = NonbondedOptions(),
  fe_options: FEOptions = FEOptions(),
  precision: Optional[str] = None,
  dense_mask_format: bool = True,
) -> tuple[
  Callable[..., dict[str, Array]],
  NeighborFn,
  space.DisplacementFn,
  space.ShiftFn,
]:
  """
  Generate a JAX compatible potential energy function to evaluate the AMBER
  forcefield.

  Args:
    params: Forcefield parameters (mm_forcefields params structure)
    topology: Topology container (mm_forcefields topology structure)
    box_vectors: Simulation box. May be scalar, shape (3,), or shape (3, 3)
      (triclinic). Used when periodic boundary conditions are enabled.
    coulomb_options: Coulomb handler implementing .energy(...) (e.g. cutoff,
      Ewald, PME)
    nb_options: Nonbonded options controlling cutoffs, neighbor list format,
      and PBC handling
    fe_options: Free-energy/alchemical option
    precision: "single" or "double" or "mixed" # TODO currently not implemented

  Returns:
    A tuple (energy_fn, neighbor_fn, displacement_fn, shift_fn) where:
      - energy_fn(positions, params, neighbor_list, cl_lambda=0.0) returns a
        dict[str, Array] containing per-term energies and a total etotal.
      - neighbor_fn is a partition.NeighborListFn compatible with
        jax_md.partition neighbor list objects.
      - displacement_fn, shift_fn are the space functions used by the model.
  """
  # TODO Flesh this out with some kind of treemap check to make sure all
  # leaves of passed dataclasses are correct type

  if precision is not None:
    raise NotImplementedError(
      'Precision handling is currently not implemented, double precision assumed'
    )
  # if precision == "double":
  #   wp_float = np.float64
  #   wp_int = np.int32
  # elif precision == "single":
  #   wp_float = np.float32
  #   wp_int = np.int32
  # else:
  #   raise ValueError("Invalid option provided for precision")

  bonded = params.bonded
  nonbonded = params.nonbonded

  use_softcore_vdw = (fe_options.vdw_scaling or '').lower() == 'softcore'
  use_linear_coul = (fe_options.coul_scaling or '').lower() == 'linear'
  use_softcore_lrc = bool(getattr(fe_options, 'softcore_lrc', False))
  softcore_lrc_points = int(getattr(fe_options, 'softcore_lrc_points', 96))
  if use_softcore_vdw:
    sc_atom_mask = jnp.asarray(fe_options.sc_mask)
    if sc_atom_mask.ndim == 0:
      # Scalar fallback for convenience; explicit length-N mask is preferred
      sc_atom_mask = jnp.ones((topology.n_atoms,), dtype=jnp.bool_) * (
        sc_atom_mask != 0
      )
    else:
      sc_atom_mask = jnp.asarray(sc_atom_mask, dtype=jnp.bool_)

  disp_coef_main_alch = None
  if use_softcore_vdw and (nb_options.r_cut is not None):

    def _eval_switch_integral_jnp(
      r: Array | float, rs: float, rc: float, sigma: Array | float
    ) -> Array:
      A = 1.0 / (rc - rs)
      A2 = A * A
      A3 = A2 * A
      sig2 = sigma * sigma
      sig6 = sig2 * sig2 * sig2
      rs2 = rs * rs
      rs3 = rs * rs2
      r2 = r * r
      r3 = r * r2
      r4 = r * r3
      r5 = r * r4
      r6 = r * r5
      r9 = r3 * r6
      return (
        sig6
        * A3
        * (
          (
            sig6
            * (
              +rs3 * 28.0 * (6.0 * rs2 * A2 + 15.0 * rs * A + 10.0)
              - r * rs2 * 945.0 * (rs2 * A2 + 2.0 * rs * A + 1.0)
              + r2 * rs * 1080.0 * (2.0 * rs2 * A2 + 3.0 * rs * A + 1.0)
              - r3 * 420.0 * (6.0 * rs2 * A2 + 6.0 * rs * A + 1.0)
              + r4 * 756.0 * (2.0 * rs * A2 + A)
              - r5 * 378.0 * A2
            )
            - r6
            * (
              +rs3 * 84.0 * (6.0 * rs2 * A2 + 15.0 * rs * A + 10.0)
              - r * rs2 * 3780.0 * (rs2 * A2 + 2.0 * rs * A + 1.0)
              + r2 * rs * 7560.0 * (2.0 * rs2 * A2 + 3.0 * rs * A + 1.0)
            )
          )
          / (252.0 * r9)
          - jnp.log(r) * 10.0 * (6.0 * rs2 * A2 + 6.0 * rs * A + 1.0)
          + r * 15.0 * (2.0 * rs * A2 + A)
          - r2 * 3.0 * A2
        )
      )

    def _calc_disp_coef_subset(
      sigma_subset: Array,
      epsilon_subset: Array,
      n_atoms_full: int,
      r_cut: float,
      r_switch: float | None,
    ) -> Array:
      if sigma_subset.size == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)
      disp_coefs = jnp.stack([sigma_subset, epsilon_subset], axis=1)
      values, count = jnp.unique(disp_coefs, axis=0, return_counts=True)

      sigma2 = values[:, 0] * values[:, 0]
      sigma6 = sigma2 * sigma2 * sigma2
      count_s = count * (count + 1) / 2
      sum1 = jnp.sum(count_s * values[:, 1] * sigma6 * sigma6)
      sum2 = jnp.sum(count_s * values[:, 1] * sigma6)

      sig_mesh = jnp.triu(
        jnp.array(jnp.meshgrid(values[:, 0], values[:, 0])), k=1
      ).T.reshape(-1, 2)
      eps_mesh = jnp.triu(
        jnp.array(jnp.meshgrid(values[:, 1], values[:, 1])), k=1
      ).T.reshape(-1, 2)
      count_mesh = jnp.triu(
        jnp.array(jnp.meshgrid(count, count)), k=1
      ).T.reshape(-1, 2)

      sig_c = 0.5 * jnp.sum(sig_mesh, axis=1)
      eps_c = jnp.sqrt(eps_mesh[:, 0] * eps_mesh[:, 1])
      count_c = jnp.prod(count_mesh, axis=1)

      sigma2 = sig_c * sig_c
      sigma6 = sigma2 * sigma2 * sigma2
      sum1 = sum1 + jnp.sum(count_c * eps_c * sigma6 * sigma6)
      sum2 = sum2 + jnp.sum(count_c * eps_c * sigma6)

      sum3 = jnp.asarray(0.0, dtype=jnp.float64)
      if r_switch is not None:
        sum3 = jnp.sum(
          count_s
          * values[:, 1]
          * (
            _eval_switch_integral_jnp(r_cut, r_switch, r_cut, values[:, 0])
            - _eval_switch_integral_jnp(r_switch, r_switch, r_cut, values[:, 0])
          )
        )
        sum3 = sum3 + jnp.sum(
          count_c
          * eps_c
          * (
            _eval_switch_integral_jnp(r_cut, r_switch, r_cut, sig_c)
            - _eval_switch_integral_jnp(r_switch, r_switch, r_cut, sig_c)
          )
        )

      denom = (n_atoms_full * (n_atoms_full + 1)) / 2
      sum1 = sum1 / denom
      sum2 = sum2 / denom
      sum3 = sum3 / denom
      n_full = jnp.asarray(float(n_atoms_full), dtype=jnp.float64)
      return (
        8.0
        * n_full
        * n_full
        * jnp.pi
        * (sum1 / (9.0 * (r_cut**9)) - sum2 / (3.0 * (r_cut**3)) + sum3)
      )

    sc_mask_jnp = jnp.asarray(sc_atom_mask, dtype=jnp.bool_)
    cc_mask_jnp = ~sc_mask_jnp
    sigma_jnp = jnp.asarray(nonbonded.sigma)
    epsilon_jnp = jnp.asarray(nonbonded.epsilon)
    rs = float(nb_options.r_switch) if nb_options.r_switch is not None else None
    disp_coef_main_alch = _calc_disp_coef_subset(
      sigma_jnp[cc_mask_jnp],
      epsilon_jnp[cc_mask_jnp],
      int(topology.n_atoms),
      float(nb_options.r_cut),
      rs,
    )

  disp_fn, shift_fn = space_selector(nb_options, box_vectors)

  if isinstance(coulomb_options, PMECoulomb):
    coulomb_method = 'PME'
  elif isinstance(coulomb_options, EwaldCoulomb):
    coulomb_method = 'Ewald'
  elif isinstance(coulomb_options, CutoffCoulomb):
    coulomb_method = 'Cutoff'
  else:
    coulomb_method = 'NoCutoff'
  coulomb = coulomb_options

  # Precompute CMAP coefficients with OpenMM's conventions
  cmap_precomp = None
  if getattr(bonded.cmap_maps, 'size', 0) != 0:
    cmap_precomp = cmap_setup(bonded.cmap_maps)

  ### Mapped function creation
  # NOTE initial values for kwargs must be passed to smap for correct
  # handling when passed to the mapped function and to store combinators

  bond_energy_mapped = smap.bond(
    harmonic_bond,
    space.canonicalize_displacement_or_metric(disp_fn),
    static_bonds=topology.bonds,
    ignore_unused_parameters=True,
    k=None,
    l=None,
  )

  angle_energy_mapped = smap.angle(
    harmonic_angle,
    disp_fn,
    static_angles=topology.angles,
    ignore_unused_parameters=True,
    k=None,
    theta0=None,
  )

  torsion_energy_mapped = smap.torsion(
    periodic_torsion,
    disp_fn,
    static_torsions=topology.torsions,
    ignore_unused_parameters=True,
    k=None,
    phase=None,
    period=None,
  )

  improper_energy_mapped = smap.torsion(
    harmonic_improper,
    disp_fn,
    static_torsions=topology.impropers,
    ignore_unused_parameters=True,
    k=None,
    theta0=None,
  )

  if nb_options.r_switch is not None:
    vdw_cutoff_fn = partial(
      force_switch, r_on=nb_options.r_switch, r_off=nb_options.r_cut
    )
    ele_cutoff_fn = partial(hard_cutoff, r_cut=nb_options.r_cut)
  elif nb_options.r_cut is not None:
    vdw_cutoff_fn = partial(hard_cutoff, r_cut=nb_options.r_cut)
    ele_cutoff_fn = partial(hard_cutoff, r_cut=nb_options.r_cut)
  else:
    vdw_cutoff_fn = lambda fn: fn
    ele_cutoff_fn = lambda fn: fn

  # Create fused LJ + coulomb function to avoid 2 passes over smap
  # NOTE probably not the best way to flag if the ab form is used
  use_ab = len(topology.nbfix_atom_type) != 0
  if use_softcore_vdw and use_ab:
    raise NotImplementedError(
      'Softcore vdW is currently implemented only for sigma/epsilon LJ (no NBFIX path yet).'
    )
  use_erfc = isinstance(coulomb_options, (EwaldCoulomb, PMECoulomb))

  def coulomb_fn(
    dr: Array,
    charge_sq: Array,
    alpha: Array,
    use_erfc: bool,
    **unused_kwargs: Any,
  ) -> Array:
    energy = COULOMB_CONSTANT * (charge_sq / dr)
    if use_erfc:
      energy *= erfc(alpha * dr)
    return energy

  def lj_fn(
    dr: Array, lj1: Array, lj2: Array, use_ab: bool, **unused_kwargs: Any
  ) -> Array:
    if use_ab:
      dr2 = dr * dr
      dr6 = dr2 * dr2 * dr2
      energy = (lj1 / dr6) ** 2 - (lj2 / dr6)
    else:
      idr = lj1 / dr
      idr2 = idr * idr
      idr6 = idr2 * idr2 * idr2
      idr12 = idr6 * idr6
      energy = 4.0 * lj2 * (idr12 - idr6)
    return energy

  def fused_nb_fn_generator():
    lj_wrapped = vdw_cutoff_fn(jax.tree_util.Partial(lj_fn, use_ab=use_ab))
    coul_wrapped = ele_cutoff_fn(
      jax.tree_util.Partial(coulomb_fn, use_erfc=use_erfc)
    )

    def _fused_fn(dr, lj1, lj2, charge_sq, alpha, **unused_kwargs):
      dr = jnp.where(jnp.isclose(dr, 0.0), 1, dr)
      lj_energy = lj_wrapped(dr, lj1, lj2, **unused_kwargs)
      coul_energy = coul_wrapped(dr, charge_sq, alpha, **unused_kwargs)
      # return lj_energy + coul_energy
      return jnp.stack([lj_energy, coul_energy], axis=-1)  # (..., 2)

    return _fused_fn

  fused_nb_fn = fused_nb_fn_generator()

  if use_ab:
    lj1_rule = None
    lj2_rule = None
    species_rule = topology.nbfix_atom_type
  else:
    lj1_rule = (combine_lorentz, None)
    lj2_rule = (combine_berthelot, None)
    species_rule = None

  # NOTE the alpha value is closed over during generation which
  # may be an issue if the grid layout changes during dynamics
  pair_lj_edge_fn = smap.pair_neighbor_list(
    fused_nb_fn,
    space.canonicalize_displacement_or_metric(disp_fn),
    reduce_axis=(),  # return edge-wise terms to support SC/CC masking
    ignore_unused_parameters=True,
    species=species_rule,
    lj1=lj1_rule,
    lj2=lj2_rule,
    charge_sq=(combine_product, None),
    alpha=coulomb.alpha if use_erfc else 0.0,
  )

  pair_coul_1r_edge_fn = None
  if use_softcore_vdw and use_linear_coul:
    coul_1r_wrapped = ele_cutoff_fn(
      lambda dr, charge_sq, **unused_kwargs: COULOMB_CONSTANT
      * (charge_sq / jnp.where(jnp.isclose(dr, 0.0), 1.0, dr))
    )
    pair_coul_1r_edge_fn = smap.pair_neighbor_list(
      coul_1r_wrapped,
      space.canonicalize_displacement_or_metric(disp_fn),
      reduce_axis=(),
      ignore_unused_parameters=True,
      charge_sq=(combine_product, None),
    )

  softcore_lj_edge_fn = None
  if use_softcore_vdw:
    softcore_wrapped = vdw_cutoff_fn(
      jax.tree_util.Partial(lennard_jones_softcore)
    )
    softcore_lj_edge_fn = smap.pair_neighbor_list(
      softcore_wrapped,
      space.canonicalize_displacement_or_metric(disp_fn),
      reduce_axis=(),
      ignore_unused_parameters=True,
      species=species_rule,
      sigma=(combine_lorentz, None),
      epsilon=(combine_berthelot, None),
      cl_lambda=None,
    )

  bond_lj_fn = smap.bond(
    lennard_jones,
    space.canonicalize_displacement_or_metric(disp_fn),
    sigma=None,
    epsilon=None,
  )

  # TODO can charges, box vectors, and charge prod be separated from this?
  mapped_coul_fns = coulomb.prepare_smap(
    nonbonded.charges,
    box_vectors,
    nonbonded.exc_charge_prod,
    displacement_fn=disp_fn,
    cutoff_fn=ele_cutoff_fn,
    fractional_coordinates=nb_options.fractional_coordinates,
  )

  ### Creating dense mask for neighbor exclusions
  # NOTE it might help to organize masking and conversion primitives
  # to handle any case of dense, sparse, (N,N), orderedsparse, etc
  # TODO consider if forcing 32 bit here will improve performance
  if dense_mask_format:
    mask_fn = _create_dense_mask(topology.n_atoms, topology.exc_pairs)
  else:
    mask_fn = _create_sparse_mask(topology.n_atoms, topology.exc_pairs)

  neighbor_fn = neighbor.create_neighbor_list(
    disp_fn,
    box_vectors if nb_options.use_pbc else None,
    nb_options.r_cut,
    nb_options.dr_threshold,
    custom_mask_function=mask_fn,
    fractional_coordinates=nb_options.fractional_coordinates,
    format=nb_options.nb_format,
    disable_cell_list=not nb_options.use_pbc,
    dense_candidate_format=dense_mask_format,
  )

  def _sc_edge_masks(nbr_list: NeighborList) -> tuple[Array, Array, Array]:
    if nb_options.nb_format in (
      NeighborListFormat.Sparse,
      NeighborListFormat.OrderedSparse,
    ):
      receiver = nbr_list.idx[0]
      sender = nbr_list.idx[1]
      valid = (receiver < topology.n_atoms) & (sender < topology.n_atoms)
    else:
      sender = jnp.arange(topology.n_atoms, dtype=jnp.int32)[:, None]
      receiver = nbr_list.idx
      valid = receiver < topology.n_atoms
      receiver = jnp.where(valid, receiver, 0)

    sender_is_sc = sc_atom_mask[sender]
    receiver_is_sc = sc_atom_mask[receiver]
    sc_sc = valid & sender_is_sc & receiver_is_sc
    cc_cc = valid & (~sender_is_sc) & (~receiver_is_sc)
    sc_cc = valid & ~(sc_sc | cc_cc)
    return sc_sc, sc_cc, cc_cc

  # Optional numerical softcore LRC approximation for SC-CC split force.
  _soft_lrc_ready = False
  _soft_lrc_sigma = None
  _soft_lrc_epsilon = None
  _soft_lrc_pair_weight = None
  _soft_lrc_tail_x = None
  _soft_lrc_tail_w = None
  _soft_lrc_sw_x = None
  _soft_lrc_sw_w = None
  _soft_lrc_rc = None
  _soft_lrc_rs = None
  if (
    use_softcore_vdw
    and use_softcore_lrc
    and (nb_options.r_cut is not None)
    and topology.n_atoms > 0
  ):
    sc_mask_jnp = jnp.asarray(sc_atom_mask, dtype=jnp.bool_)
    sc_idx = jnp.where(sc_mask_jnp)[0]
    cc_idx = jnp.where(~sc_mask_jnp)[0]
    if sc_idx.size > 0 and cc_idx.size > 0:
      sigma_jnp = jnp.asarray(nonbonded.sigma)
      epsilon_jnp = jnp.asarray(nonbonded.epsilon)
      # Match modify_alchemical_system safeguard for zero sigma in softcore force.
      sigma_jnp = jnp.where(jnp.isclose(sigma_jnp, 0.0), 3.0, sigma_jnp)

      sc_vals, sc_counts = jnp.unique(
        jnp.stack([sigma_jnp[sc_idx], epsilon_jnp[sc_idx]], axis=1),
        axis=0,
        return_counts=True,
      )
      cc_vals, cc_counts = jnp.unique(
        jnp.stack([sigma_jnp[cc_idx], epsilon_jnp[cc_idx]], axis=1),
        axis=0,
        return_counts=True,
      )
      pair_counts = sc_counts[:, None] * cc_counts[None, :]
      sigma_pair = 0.5 * (sc_vals[:, None, 0] + cc_vals[None, :, 0])
      epsilon_pair = jnp.sqrt(sc_vals[:, None, 1] * cc_vals[None, :, 1])

      n_atoms_f = float(topology.n_atoms)
      num_interactions = (n_atoms_f * (n_atoms_f + 1.0)) / 2.0

      # NOTE: OpenMM CustomNonbondedForce LRC uses iterative midpoint/Richardson integration;
      # I use one-shot Gauss quadrature here for a simpler approximation
      gx, gw = np.polynomial.legendre.leggauss(softcore_lrc_points)
      x_tail = 0.5 * (gx + 1.0)
      w_tail = 0.5 * gw

      _soft_lrc_sigma = jnp.asarray(sigma_pair.reshape(-1), dtype=jnp.float64)
      _soft_lrc_epsilon = jnp.asarray(
        epsilon_pair.reshape(-1), dtype=jnp.float64
      )
      _soft_lrc_pair_weight = jnp.asarray(
        pair_counts.reshape(-1) / num_interactions, dtype=jnp.float64
      )
      _soft_lrc_tail_x = jnp.asarray(x_tail, dtype=jnp.float64)
      _soft_lrc_tail_w = jnp.asarray(w_tail, dtype=jnp.float64)
      _soft_lrc_rc = jnp.asarray(float(nb_options.r_cut), dtype=jnp.float64)

      if nb_options.r_switch is not None:
        _soft_lrc_rs = jnp.asarray(
          float(nb_options.r_switch), dtype=jnp.float64
        )
        _soft_lrc_sw_x = jnp.asarray(x_tail, dtype=jnp.float64)
        _soft_lrc_sw_w = jnp.asarray(w_tail, dtype=jnp.float64)

      _soft_lrc_ready = True

  def _softcore_lrc_sc_cc(cl_lambda_val: Array | float, volume: Array) -> Array:
    if not _soft_lrc_ready:
      return jnp.asarray(0.0, dtype=jnp.float64)

    rc = _soft_lrc_rc
    x = _soft_lrc_tail_x
    w = _soft_lrc_tail_w

    r_tail = rc / x[None, :]
    u_tail = lennard_jones_softcore(
      r_tail,
      _soft_lrc_sigma[:, None],
      _soft_lrc_epsilon[:, None],
      cl_lambda_val,
    )
    i_tail = jnp.sum(w[None, :] * u_tail * (rc**3) / (x[None, :] ** 4), axis=1)

    i_switch = jnp.asarray(0.0, dtype=jnp.float64)
    if _soft_lrc_rs is not None:
      rs = _soft_lrc_rs
      xs = _soft_lrc_sw_x
      ws = _soft_lrc_sw_w
      r_sw = rs + xs[None, :] * (rc - rs)
      s = xs**3 * (10.0 + xs * (-15.0 + xs * 6.0))
      u_sw = lennard_jones_softcore(
        r_sw,
        _soft_lrc_sigma[:, None],
        _soft_lrc_epsilon[:, None],
        cl_lambda_val,
      )
      i_switch = jnp.sum(
        ws[None, :] * s[None, :] * u_sw * (r_sw**2) * (rc - rs), axis=1
      )

    i_pair = i_tail + i_switch
    sum_weighted = jnp.sum(_soft_lrc_pair_weight * i_pair)
    n_atoms_f = jnp.asarray(float(topology.n_atoms), dtype=jnp.float64)
    coeff = 2.0 * jnp.pi * n_atoms_f * n_atoms_f * sum_weighted
    return coeff / volume

  def energy_fn(
    positions: Array,
    nbr_list: NeighborList,
    cl_lambda: Array | float = 0.0,
    lambda_q: Array | float = 1.0,
    **kwargs: Any,
  ) -> dict[str, Array]:
    """
    Evaluate the total potential energy and return a per-term breakdown.

    Args:
      positions: Particle positions. Must be consistent with the selected
        space parameterization. If nb_options.fractional_coordinates=True,
        positions are expected to lie in the unit cube [0, 1).
      nbr_list: A neighbor list object created/updated by neighbor_fn.
      cl_lambda: Alchemical lambda value used by free-energy
        scaling paths.
      lambda_q: Electrostatic alchemical lambda for linear charge scaling.
      kwargs: kwargs to pass through dynamic structures. Note that this
        may not always be safe and can cause recompilation or runtime
        errors.

    Returns:
      A dictionary of energies in kcal/mol.
    """
    _params = kwargs.pop('params', params)
    _box = kwargs.pop('box', box_vectors)
    perturbation = kwargs.pop('perturbation', None)
    bonded = _params.bonded
    nonbonded = _params.nonbonded
    result_dict = dict()

    # periodic and periodic_general accept different arguments for this
    # TODO I think the current code is set up to only allow NPT simulation
    # with periodic_general. It might be more correct with the current interface
    # to only allow box as a kwarg and throw an error if side is passed

    # TODO NOTE For NPT, perturbation must affect
    # (1) displacement/metric distances
    # (2) any box-dependent terms (PME reciprocal, tail corrections via V).
    # in addition, the grid arrangement for PME is computed according
    # to the original box size; if a big enough perturbation takes place
    # additional error may occur. It isn't clear if other packages change
    # the grid arangement/alpha during dynamics or if the discontinuities from doing
    # this cause issues. It also isn't clear if there's a tolerance beyond which
    # an error should be thrown.
    if nb_options.use_periodic_general:
      box_kwarg = {'box': _box}
    else:
      box_kwarg = {'side': _box}
    space_kwarg = dict(box_kwarg)
    if perturbation is not None:
      space_kwarg['perturbation'] = perturbation

    # NOTE most of these functions should work in the case there are no entries
    # into the respective indexing array, but if there are dummy entries as
    # in the case where parameter sets are aligned and padded, issues may occur

    ### Bonded interactions
    bond_pot = bond_energy_mapped(
      positions, k=bonded.bond_k, l=bonded.bond_r0, **space_kwarg
    )
    result_dict['bond_pot'] = bond_pot

    angle_pot = angle_energy_mapped(
      positions, k=bonded.angle_k, theta0=bonded.angle_theta0, **space_kwarg
    )
    result_dict['angle_pot'] = angle_pot

    torsion_pot = torsion_energy_mapped(
      positions,
      k=bonded.torsion_k,
      phase=bonded.torsion_gamma,
      period=bonded.torsion_n,
      **space_kwarg,
    )
    result_dict['torsion_pot'] = torsion_pot

    improper_pot = improper_energy_mapped(
      positions,
      k=bonded.improper_k,
      theta0=bonded.improper_gamma,
      **space_kwarg,
    )
    result_dict['improper_pot'] = improper_pot

    cmap_pot = cmap_energy(
      positions,
      topology.cmap_atoms,
      topology.cmap_map_idx,
      cmap_precomp,
      partial(disp_fn, **space_kwarg),
    )
    result_dict['cmap_pot'] = cmap_pot

    ### Coulomb Interaction
    coul_pot, coul_exc_pot = coulomb.energy_smap(
      positions,
      nonbonded.charges,
      nbr_list,
      space_kwarg,
      topology.exc_pairs,
      nonbonded.exc_charge_prod,
      coulomb_fns=mapped_coul_fns,
      return_components=True,  # TODO need to work out smap wiring for this
    )

    ### Lennard Jones Interaction
    if use_ab:
      lj1 = nonbonded.nbfix_acoef
      lj2 = nonbonded.nbfix_bcoef
    else:
      lj1 = nonbonded.sigma
      lj2 = nonbonded.epsilon

    edge_terms = pair_lj_edge_fn(
      positions,
      nbr_list,
      lj1=lj1,
      lj2=lj2,
      charge_sq=nonbonded.charges,
      **space_kwarg,
    )
    lj_edge = edge_terms[..., 0]
    coul_dir_edge = edge_terms[..., 1]

    coul_dir_pot = jnp.sum(coul_dir_edge)

    if use_softcore_vdw and use_linear_coul:
      assert pair_coul_1r_edge_fn is not None
      sc_sc_mask, _, _ = _sc_edge_masks(nbr_list)
      sc_sc_coul_1r_edge = pair_coul_1r_edge_fn(
        positions,
        nbr_list,
        charge_sq=params.nonbonded.charges,
        **space_kwarg,
      )
      sc_sc_coul_full = jnp.sum(jnp.where(sc_sc_mask, sc_sc_coul_1r_edge, 0.0))
      sc_sc_coul_addback = (1.0 - lambda_q * lambda_q) * sc_sc_coul_full
      coul_dir_pot = coul_dir_pot + sc_sc_coul_addback
      result_dict['sc_sc_coul_1r_full_pot'] = sc_sc_coul_full
      result_dict['sc_sc_coul_addback_pot'] = sc_sc_coul_addback

    if use_softcore_vdw:
      assert softcore_lj_edge_fn is not None
      soft_lj_edge = softcore_lj_edge_fn(
        positions,
        nbr_list,
        sigma=nonbonded.sigma,
        epsilon=nonbonded.epsilon,
        cl_lambda=cl_lambda,
        **space_kwarg,
      )
      sc_sc_mask, sc_cc_mask, cc_cc_mask = _sc_edge_masks(nbr_list)

      lj_sc_sc = jnp.sum(jnp.where(sc_sc_mask, lj_edge, 0.0))
      lj_sc_cc = jnp.sum(jnp.where(sc_cc_mask, lj_edge, 0.0))
      lj_cc_cc = jnp.sum(jnp.where(cc_cc_mask, lj_edge, 0.0))

      sc_lj_sc_sc = jnp.sum(jnp.where(sc_sc_mask, soft_lj_edge, 0.0))
      sc_lj_sc_cc = jnp.sum(jnp.where(sc_cc_mask, soft_lj_edge, 0.0))
      sc_lj_cc_cc = jnp.sum(jnp.where(cc_cc_mask, soft_lj_edge, 0.0))

      # HFE-style default: only SC-CC uses softcore, CC-CC and SC-SC remain normal.
      lj_pot = lj_cc_cc + lj_sc_sc + sc_lj_sc_cc

      result_dict['lj_cc_cc_pot'] = lj_cc_cc
      result_dict['lj_sc_cc_pot'] = lj_sc_cc
      result_dict['lj_sc_sc_pot'] = lj_sc_sc
      result_dict['lj_softcore_cc_cc_pot'] = sc_lj_cc_cc
      result_dict['lj_softcore_sc_cc_pot'] = sc_lj_sc_cc
      result_dict['lj_softcore_sc_sc_pot'] = sc_lj_sc_sc
    else:
      lj_pot = jnp.sum(lj_edge)

    lj_exc_pot = bond_lj_fn(
      positions,
      topology.exc_pairs,
      sigma=nonbonded.exc_sigma,
      epsilon=nonbonded.exc_epsilon,
      **space_kwarg,
    )

    coul_pot = coul_pot + coul_dir_pot

    # Calculate Dispersion Correction term
    # TODO note that this will be incorrect if parameters change
    # TODO switching function correction as well as softcore/common core
    # alchemical corrections are implemented, but tail correction for
    # nbfix and other custom nonbonded terms is not yet implemented
    # examples for how to implement these corrections in a more general
    # way are in NonbondedForceImpl.cpp and CustomNonbondedForceImpl.cpp
    if coulomb_method == 'PME':
      box_for_volume = _box
      if perturbation is not None:
        box_for_volume = box_for_volume * perturbation
      if box_for_volume.ndim == 2:
        V = jnp.linalg.det(box_for_volume)
      else:
        V = jnp.prod(box_for_volume)
      disp_coef_main = (
        disp_coef_main_alch
        if (use_softcore_vdw and disp_coef_main_alch is not None)
        else nb_options.disp_coef
      )
      disp_main_pot = disp_coef_main / V
      disp_softcore_sc_cc_pot = _softcore_lrc_sc_cc(cl_lambda, V)
      disp_pot = disp_main_pot + disp_softcore_sc_cc_pot
    else:
      disp_main_pot = jnp.float32(0.0)
      disp_softcore_sc_cc_pot = jnp.float32(0.0)
      disp_pot = jnp.float32(0.0)

    result_dict['lj_pot'] = lj_pot
    result_dict['coul_pot'] = coul_pot
    result_dict['disp_main_pot'] = disp_main_pot
    result_dict['disp_softcore_sc_cc_pot'] = disp_softcore_sc_cc_pot
    result_dict['disp_pot'] = disp_pot
    result_dict['lj_exc_pot'] = lj_exc_pot
    result_dict['coul_exc_pot'] = coul_exc_pot

    nb_pot = lj_pot + coul_pot + lj_exc_pot + coul_exc_pot
    result_dict['nb_pot'] = nb_pot

    etotal = (
      bond_pot
      + angle_pot
      + torsion_pot
      + improper_pot
      + cmap_pot
      + nb_pot
      + disp_pot
    )
    result_dict['etotal'] = etotal

    return result_dict

  return energy_fn, neighbor_fn, disp_fn, shift_fn


def harmonic_bond(dist: Array, k: Array, l: Array) -> Array:
  """
  Harmonic bond energy function

  Args:
    dist: Bond distance
    k: Force constant
    l: Equilibrium distance

  Returns:
    Bond energy contribution
  """
  return 0.5 * k * jnp.power((dist - l), 2)


def harmonic_angle(
  dr_ij: Array, dr_kj: Array, k: Array, theta0: Array
) -> Array:
  """
  Harmonic angle energy function

  Args:
    dr_ij: Displacement vector from atom j to i
    dr_kj: Displacement vector from atom j to k
    k: Force constant
    theta0: Equilibrium angle

  Returns:
    Angle energy contribution
  """
  # TODO not a stable choice for angle or dihedral
  # shuld smap support 3 or 4 body metrics in addition to 2 body?
  theta = jax.vmap(compute_angle)(dr_ij, dr_kj)
  return 0.5 * k * jnp.power((theta - theta0), 2)


def periodic_torsion(
  dr_ij: Array,
  dr_jk: Array,
  dr_kl: Array,
  k: Array,
  phase: Array,
  period: Array,
) -> Array:
  """
  Periodic torsion (proper dihedral) energy.

  Args:
    dr_ij: Displacement vector i-j
    dr_jk: Displacement vector j-k
    dr_kl: Displacement vector k-l
    k: Torsion amplitude
    phase: Phase offset
    period: Periodicity. Non integer values may cause issues.

  Returns:
    Torsion energy contribution
  """
  theta = jax.vmap(compute_dihedral)(dr_ij, dr_jk, dr_kl)
  return k * (1.0 + jnp.cos(period * theta - phase))


def harmonic_improper(
  dr_ij: Array,
  dr_jk: Array,
  dr_kl: Array,
  k: Array,
  theta0: Array,
) -> Array:
  """CHARMM-style improper torsion energy (harmonic in the dihedral angle).

  OpenMM's CHARMM PSF loader implements impropers via a CustomTorsionForce with
  an energy of the form:

    dtheta = abs(theta - theta0)
    dtheta = min(dtheta, 2*pi - dtheta)
    E = k * dtheta^2

  Args:
    dr_ij: Displacement vector p1-p0.
    dr_jk: Displacement vector p2-p1.
    dr_kl: Displacement vector p3-p2.
    k: Force constant (kcal/mol / rad^2).
    theta0: Equilibrium dihedral angle (radians).

  Returns:
    Improper torsion energy contribution.
  """
  theta = jax.vmap(compute_dihedral)(dr_ij, dr_jk, dr_kl)
  dtheta = jnp.abs(theta - theta0)
  dtheta = jnp.minimum(dtheta, 2.0 * jnp.pi - dtheta)
  return k * dtheta * dtheta


def lennard_jones(
  dr: Array, sigma: Array, epsilon: Array, **unused_kwargs: Any
) -> Array:
  """Standard 12-6 Lennard-Jones pair potential.

  Args:
    dr: Pair distance
    sigma: Lennard-Jones sigma
    epsilon: Lennard-Jones epsilon

  Returns:
    LJ energy
  """
  # TODO is this masking necessary, or should it be handled by smap?
  dr = jnp.where(jnp.isclose(dr, 0.0), 1, dr)
  idr = sigma / dr
  idr2 = idr * idr
  idr6 = idr2 * idr2 * idr2
  idr12 = idr6 * idr6
  return 4.0 * epsilon * (idr12 - idr6)


def lennard_jones_ab(
  dr: Array, a: Array, b: Array, **unused_kwargs: Any
) -> Array:
  """Standard 12-6 Lennard-Jones pair potential.

  Args:
    dr: Pair distance
    a: Lennard-Jones A coefficient
    b: Lennard-Jones B coefficient

  Returns:
    LJ energy
  """
  # TODO is this masking necessary, or should it be handled by smap?
  dr = jnp.where(jnp.isclose(dr, 0.0), 1, dr)
  dr2 = dr * dr
  dr6 = dr2 * dr2 * dr2
  return (a / dr6) ** 2 - (b / dr6)


def lennard_jones_softcore(
  dr: Array,
  sigma: Array,
  epsilon: Array,
  cl_lambda: float,
  alpha: float = 0.5,
) -> Array:
  """
  Beutler/AMBER-style softcore Lennard-Jones

  References:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3187911/

  Args:
    dr: Pair distance
    sigma: Lennard-Jones sigma
    epsilon: Lennard-Jones epsilon
    cl_lambda: Alchemical lambda (0=fully interacting, 1=decoupled)
    alpha: Softcore alpha parameter

  Returns:
    Softcore LJ energy (kcal/mol)
  """
  # TODO is this masking necessary, or should it be handled by smap?
  dr = jnp.where(jnp.isclose(dr, 0.0), 1, dr)
  idr = dr / sigma
  idr2 = idr * idr
  idr6 = idr2 * idr2 * idr2
  lidr6 = alpha * cl_lambda + idr6
  lidr12 = lidr6 * lidr6
  return 4.0 * epsilon * (1.0 - cl_lambda) * ((1.0 / lidr12) - (1.0 / lidr6))


def coulomb_softcore(
  dr: Array,
  charge_sq: Array,
  cl_lambda: float,
  alpha: float = 0.5,
) -> Array:
  """
  Softcore Coulomb for alchemical transformations

  References:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3187911/

  Args:
    dr: Pair distance
    charge_sq: Product q_i * q_j
    cl_lambda: Alchemical lambda (0=fully interacting, 1=decoupled)
    alpha: Softcore alpha parameter

  Returns:
    Softcore Coulomb energy (in reduced units; multiply by Coulomb constant for
    kcal/mol if needed).
  """
  # TODO is this masking necessary, or should it be handled by smap?
  dr = jnp.where(jnp.isclose(dr, 0.0), 1, dr)
  soft_r = jnp.sqrt(dr * dr + alpha * (1.0 - cl_lambda) ** 2)
  return cl_lambda * charge_sq / soft_r


def _wrap_angle(theta: Array) -> Array:
  """
  Wrap an angle into (-pi, pi]
  """
  two_pi = jnp.asarray(2.0 * onp.pi, dtype=theta.dtype)
  pi = two_pi * 0.5
  return jnp.mod(theta + pi, two_pi) - pi


def _wrap_angle_0_2pi(theta: Array) -> Array:
  """
  Wrap an angle into [0, 2pi)
  """
  two_pi = jnp.asarray(2.0 * onp.pi, dtype=theta.dtype)
  return jnp.mod(theta + two_pi, two_pi)


def cmap_setup(cmap_maps: Array) -> tuple[Array, Array, int]:
  """
  Precompute OpenMM style bicubic CMAP coefficients

  This follows the same structure as OpenMM's CMAPTorsionForceImpl
  - compute dE/dphi at knots by fitting a periodic cubic spline along phi
  - compute dE/dpsi at knots by fitting a periodic cubic spline along psi
  - compute d^2E/(dphi dpsi) by fitting a periodic spline to dE/dpsi along phi
  - convert the four corner values + derivatives into 16 bicubic coefficients
    per patch using the bicubic Hermite basis (A = H @ G @ H.T)

  This setup will generally only be run once before dynamics in a JIT context.
  If JIT compatibility is needed, an analogue to SplineFitter is needed. This
  implementation is made to follow OpenMM's Reference platform approach as
  closely as possible, but there may be a more succinct approach with an existing
  library for someone more familiar with spline interpolation theory.

  Notes:
    - This function uses Scipy (scipy.interpolate.CubicSpline) and is not
      JIT-compatible. It is intended to run once during setup, not inside a
      compiled step function
    - The returned coefficients are evaluated by _cmap_eval_bicubic()

  Args:
    cmap_maps: Array of CMAP grids with shape (n_maps, size, size) or (size, size)

  Returns:
    A tuple (coeff, delta, size) where:
      - coeff has shape (n_maps, size*size, 16)
      - delta is the grid spacing in radians
      - size is the number of grid points along one dimension
  """

  maps_np = onp.asarray(cmap_maps, dtype=onp.float64)
  if maps_np.ndim == 2:
    maps_np = maps_np[None, ...]
  n_maps, n_x, n_y = maps_np.shape
  if n_x != n_y:
    raise ValueError('CMAP maps must be square.')

  size = int(n_x)
  delta = float(2.0 * onp.pi / size)

  # Match OpenMM's energy indexing convention
  maps_np = onp.transpose(maps_np, (0, 2, 1))

  x = onp.asarray(
    [i * 2.0 * onp.pi / size for i in range(size + 1)], dtype=onp.float64
  )

  # General form of the 4x4 Hermite basis matrix
  # e.g. https://mrl.cs.nyu.edu/~perlin/courses/spring2020/2020_04_02/
  # This is equivalent to the construction of the 16x16 wt[] matrix
  # but is arranged in a way that is more explicit
  H = onp.array(
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [-3.0, 3.0, -2.0, -1.0],
      [2.0, -2.0, 1.0, 1.0],
    ],
    dtype=onp.float64,
  )

  coeff_np = onp.zeros((n_maps, size * size, 16), dtype=onp.float64)
  y = onp.empty((size + 1,), dtype=onp.float64)

  for m in range(n_maps):
    energy_grid = maps_np[m]
    d1 = onp.zeros((size, size), dtype=onp.float64)
    d2 = onp.zeros((size, size), dtype=onp.float64)
    d12 = onp.zeros((size, size), dtype=onp.float64)

    # dE/dphi at knots (periodic spline along phi for each psi_j)
    for j in range(size):
      y[:size] = energy_grid[:, j]
      y[size] = energy_grid[0, j]
      cs = scipy.interpolate.CubicSpline(x, y, bc_type='periodic')
      d1[:, j] = cs(x[:size], 1)

    # dE/dpsi at knots (periodic spline along psi for each phi_i)
    for i in range(size):
      y[:size] = energy_grid[i, :]
      y[size] = energy_grid[i, 0]
      cs = scipy.interpolate.CubicSpline(x, y, bc_type='periodic')
      d2[i, :] = cs(x[:size], 1)

    # Cross derivative d^2E/(dphi dpsi): spline dE/dpsi along phi
    for j in range(size):
      y[:size] = d2[:, j]
      y[size] = d2[0, j]
      cs = scipy.interpolate.CubicSpline(x, y, bc_type='periodic')
      d12[:, j] = cs(x[:size], 1)

    # Finally create the bicubic patch coefficients
    for i in range(size):
      nexti = (i + 1) % size
      for j in range(size):
        nextj = (j + 1) % size

        f00 = energy_grid[i, j]
        f10 = energy_grid[nexti, j]
        f01 = energy_grid[i, nextj]
        f11 = energy_grid[nexti, nextj]

        fx00 = d1[i, j] * delta
        fx10 = d1[nexti, j] * delta
        fx01 = d1[i, nextj] * delta
        fx11 = d1[nexti, nextj] * delta

        fy00 = d2[i, j] * delta
        fy10 = d2[nexti, j] * delta
        fy01 = d2[i, nextj] * delta
        fy11 = d2[nexti, nextj] * delta

        fxy00 = d12[i, j] * delta * delta
        fxy10 = d12[nexti, j] * delta * delta
        fxy01 = d12[i, nextj] * delta * delta
        fxy11 = d12[nexti, nextj] * delta * delta

        G = onp.array(
          [
            [f00, f01, fy00, fy01],
            [f10, f11, fy10, fy11],
            [fx00, fx01, fxy00, fxy01],
            [fx10, fx11, fxy10, fxy11],
          ],
          dtype=onp.float64,
        )
        A = H @ G @ H.T
        patch = i + size * j
        coeff_np[m, patch, :] = A.reshape((16,), order='C')

  return (
    jnp.asarray(coeff_np, dtype=jnp.float64),
    f64(delta),
    int(size),
  )


def _cmap_eval_bicubic(
  map_id: Array | int,
  phi: Array,
  psi: Array,
  cmap_precomp: tuple[Array, Array, int],
) -> Array:
  """
  Evaluate CMAP energy using OpenMM-compatible bicubic coefficients

  Args:
    map_id: Integer map index
    phi: First dihedral angle in radians, wrapped into [0, 2pi)
    psi: Second dihedral angle in radians, wrapped into [0, 2pi)
    cmap_precomp: Output of cmap_setup()

  Returns:
    CMAP correction energy for a single torsion pair
  """
  coeff, delta, size = cmap_precomp
  delta = jnp.asarray(delta, dtype=coeff.dtype)
  u = phi / delta
  v = psi / delta
  s = jnp.minimum(u.astype(jnp.int32), size - 1)
  t = jnp.minimum(v.astype(jnp.int32), size - 1)
  da = u - s.astype(coeff.dtype)
  db = v - t.astype(coeff.dtype)
  patch = s + size * t
  c = coeff[map_id, patch]  # [16]

  # Horner evaluation from ReferenceCMAPTorsionIxn
  e = jnp.asarray(0.0, dtype=coeff.dtype)
  for i in range(3, -1, -1):
    base = ((c[i * 4 + 3] * db + c[i * 4 + 2]) * db + c[i * 4 + 1]) * db + c[
      i * 4 + 0
    ]
    e = da * e + base
  return e


# rough analogue to the implementation in CMAPTorsionForce
def cmap_energy(
  positions: Array,
  cmap_atoms: Optional[Array],
  cmap_map_id: Optional[Array],
  cmap_precomp: Optional[tuple[Array, Array, int]],
  disp_fn: Callable[..., Array],
) -> Array:
  """
  Compute CMAP torsion correction energy

  Args:
    positions: Particle positions
    cmap_atoms: Array of CMAP torsion atom indices with shape (M, 8)
    cmap_map_id: Optional array of length M selecting which map to use for
      each torsion
    cmap_precomp: Output of cmap_setup()
    disp_fn: Displacement function consistent with the simulation space

  Returns:
    Total CMAP energy in kcal/mol
  """
  # TODO no longer robust to check none
  if cmap_atoms is None:
    return f64(0.0)
  if getattr(cmap_atoms, 'size', 0) == 0:
    return f64(0.0)
  if cmap_precomp is None:
    return f64(0.0)

  ids = cmap_map_id
  if ids is None:
    ids = jnp.zeros((cmap_atoms.shape[0],), dtype=jnp.int32)

  atoms_a = cmap_atoms[:, 0:4]
  atoms_b = cmap_atoms[:, 4:8]

  pa = positions[atoms_a]  # [M,4,3]
  pb = positions[atoms_b]

  # TODO this is essentially pairs of 4 body interactions
  # is there an idiomatic way of expressing this with smap?

  d0a = jax.vmap(disp_fn)(pa[:, 0], pa[:, 1])
  d1a = jax.vmap(disp_fn)(pa[:, 2], pa[:, 1])
  d2a = jax.vmap(disp_fn)(pa[:, 2], pa[:, 3])
  phi = jax.vmap(compute_dihedral)(d0a, d1a, d2a)

  d0b = jax.vmap(disp_fn)(pb[:, 0], pb[:, 1])
  d1b = jax.vmap(disp_fn)(pb[:, 2], pb[:, 1])
  d2b = jax.vmap(disp_fn)(pb[:, 2], pb[:, 3])
  psi = jax.vmap(compute_dihedral)(d0b, d1b, d2b)

  # OpenMM wraps CMAP angles into [0, 2pi) before patch lookup
  phi_omm = _wrap_angle_0_2pi(phi)
  psi_omm = _wrap_angle_0_2pi(psi)

  e = jax.vmap(_cmap_eval_bicubic, in_axes=(0, 0, 0, None))(
    ids, phi_omm, psi_omm, cmap_precomp
  )

  return util.high_precision_sum(e)
