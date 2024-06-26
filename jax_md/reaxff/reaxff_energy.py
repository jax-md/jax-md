"""
Contains energy related functions for ReaxFF

Author: Mehmet Cagri Kaymak
"""
from __future__ import annotations
import numpy as onp
import jax.numpy as jnp
import jax
from jax.scipy.sparse import linalg
from jax_md import util
from jax_md.util import safe_mask
from jax_md.util import high_precision_sum
from jax_md.reaxff.reaxff_helper import vectorized_cond, safe_sqrt
from jax_md.reaxff.reaxff_forcefield import ForceField
# to resolve circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from jax_md.reaxff.reaxff_interactions import ReaxFFNeighborLists
from jax import custom_jvp

# Types
f32 = util.f32
f64 = util.f64
Array = util.Array

c1c = 332.0638  # Coulomb energy conversion
rdndgr = 180.0/onp.pi
dgrrdn = 1.0/rdndgr

def calculate_reaxff_energy(species: Array,
                            atomic_numbers: Array,
                            nbr_lists: 'ReaxFFNeighborLists',
                            close_nbr_dists: Array,
                            far_nbr_dists: Array,
                            body_3_angles: Array,
                            body_4_angles: Array,
                            hb_ang_dist: Array,
                            force_field: ForceField,
                            init_charges: Array = None,
                            total_charge: float = 0.0,
                            tol: float = 1e-06,
                            max_solver_iter: int = 500,
                            backprop_solve: bool = False,
                            tors_2013: bool = False,
                            tapered_reaxff: bool = False,
                            solver_model: str = "EEM"):
  '''
  Calculate full ReaxFF potential
  Args:
    species: An ndarray of shape `[n, ]` for the atom types.
    atomic_numbers:  An ndarray of shape `[n, ]` for the atomic numbers
      of the atoms.
    nbr_lists: Contains the interaction lists for ReaxFF potential.
    close_nbr_dists: An ndarray of shape `[n,m]` for bonded interaction
      distances
    far_nbr_dists: An ndarray of shape `[n,m]` for non-bonded interaction
      distances (vdw and Coulomb)
    body_3_angles, body_4_angles, hb_ang_dist: Angles and distances for
      many-body interactions
    force_field: ReaxFF parameters
    init_charges: Initial charges for the iterative solver
      An ndarray of shape `[n, ]` or None
    total_charge: Total charge of the system (float)
    tol: Tolarence for the charge solver
    max_solver_iter: Maximum number of solver iterations
      If set to -1, use direct solve
    backprop_solve: Control variable to decide whether to do a solve to
      calculate the gradients of the charges wrt positions. By definition,
      the gradients should be 0 but if the solver tolerance is high,
      the gradients might be non-ignorable.
    tors_2013: Control variable to decide whether to use more stable
      version of the torsion interactions
    tapered_reaxff: Control variable to decide whether to use tapered cutoffs
      for various bonded interactions, causes computational overhead
    solver_model: Control variable for the solver model
      ("EEM" or "ACKS")

  Returns:
    System energy in kcal/mol
  '''
  cou_pot = 0
  vdw_pot = 0
  charge_pot = 0
  cov_pot = 0
  lone_pot = 0
  val_pot = 0
  total_penalty = 0
  total_conj = 0
  overunder_pot = 0
  tor_conj = 0
  torsion_pot = 0
  hb_pot = 0
  coulomb_acks2 = 0
  charge_pot_acks2 = 0
  self_energy = 0

  result_dict = dict()



  N = len(species)
  atom_mask = species >= 0
  self_energy = jnp.sum(force_field.self_energies[species] * atom_mask) + force_field.shift[0]
  result_dict['self_energy'] = self_energy
  far_nbr_inds = nbr_lists.far_nbrs.idx
  far_neigh_types = species[far_nbr_inds]
  close_nbr_inds = nbr_lists.close_nbrs.idx
  atom_inds = jnp.arange(N).reshape(-1,1)
  close_nbr_inds = nbr_lists.close_nbrs.idx[atom_inds,
                                                nbr_lists.filter2.idx]
  close_nbr_inds = jnp.where(nbr_lists.filter2.idx != -1,
                                 close_nbr_inds,
                                 N)
  body_3_inds = nbr_lists.filter3.idx
  body_4_inds = nbr_lists.filter4.idx
  if nbr_lists.filter_hb != None:
    hb_inds = nbr_lists.filter_hb.idx
  else:
    hb_inds = None

  # shared accross charge calc, coulomb, and vdw
  # + 1e-15 # for numerical issues
  far_nbr_mask = (far_nbr_inds != N) & (atom_mask.reshape(-1,1)
                                        & atom_mask[far_nbr_inds])
  far_nbr_dists = far_nbr_dists * far_nbr_mask
  tapered_dists = taper(far_nbr_dists, 0.0, 10.0)
  tapered_dists = jnp.where((far_nbr_dists > 10.0) | (far_nbr_dists < 0.001),
                            0.0,
                            tapered_dists)

  # shared accross charge calc and coulomb
  gamma = jnp.power(force_field.gamma.reshape(-1, 1), 3/2)
  gamma_mat = gamma * gamma.transpose()
  gamma_mat = gamma_mat[far_neigh_types, species.reshape(-1, 1)]
  hulp1_mat = far_nbr_dists ** 3 + (1/gamma_mat)
  hulp2_mat = jnp.power(hulp1_mat, 1.0/3.0) * far_nbr_mask

  if solver_model == "EEM":
    charges = calculate_eem_charges(species,
                                    atom_mask,
                                    far_nbr_inds,
                                    hulp2_mat,
                                    tapered_dists,
                                    force_field.idempotential,
                                    force_field.electronegativity,
                                    init_charges,
                                    total_charge,
                                    backprop_solve,
                                    tol,
                                    max_solver_iter)
  else:
    xcut = force_field.softcut_2d[far_neigh_types, species.reshape(-1, 1)]
    d = far_nbr_dists / xcut
    bond_softness = force_field.par_35 * (d**3) * ((1-d)**6)
    bond_softness = safe_mask(far_nbr_dists < xcut,
                              lambda x: x, bond_softness, 0.0)
    self_mask = jnp.arange(N).reshape(-1,1) == far_nbr_inds
    bond_softness = jnp.where(self_mask == 1, 0, bond_softness)
    charges,effpot = calculate_acks2_charges(species,
                                    atom_mask,
                                    far_nbr_inds,
                                    hulp2_mat,
                                    bond_softness,
                                    tapered_dists,
                                    force_field.idempotential,
                                    force_field.electronegativity,
                                    total_charge,
                                    backprop_solve,
                                    tol)

    charge_pot_acks2 = calculate_charge_energy_acks2(charges[:-1],
                                                     effpot)

    coulomb_acks2 = calculate_acks2_coulomb_pot(far_nbr_inds,
                                                atom_mask,
                                                effpot,
                                                bond_softness)
  result_dict['charges'] = charges[:-1]

  cou_pot = calculate_coulomb_pot(far_nbr_inds,
                                  atom_mask,
                                  hulp2_mat,
                                  tapered_dists,
                                  charges[:-1])
  cou_pot += coulomb_acks2
  result_dict['E_coulomb'] = cou_pot

  charge_pot = calculate_charge_energy(species,
                                       charges[:-1],
                                       force_field.idempotential,
                                       force_field.electronegativity)
  charge_pot += charge_pot_acks2
  result_dict['E_charge'] = charge_pot

  vdw_pot = calculate_vdw_pot(species,
                              far_nbr_mask,
                              far_nbr_inds,
                              far_nbr_dists,
                              tapered_dists,
                              force_field)
  result_dict['E_vdw'] = vdw_pot

  atomic_num1 = atomic_numbers.reshape(-1, 1)
  atomic_num2 = atomic_numbers[close_nbr_inds]
  # O: 8, C:6
  triple_bond1 = jnp.logical_and(atomic_num1 == 8, atomic_num2 == 6)
  triple_bond2 = jnp.logical_and(atomic_num1 == 6, atomic_num2 == 8)
  triple_bond = jnp.logical_or(triple_bond1, triple_bond2)
  covbon_mask = (close_nbr_inds != N) & (atom_mask.reshape(-1,1)
                                         & atom_mask[close_nbr_inds])
  [cov_pot, bo, bopi, bopi2, abo] = calculate_covbon_pot(close_nbr_inds,
                                                         close_nbr_dists,
                                                         covbon_mask,
                                                         species,
                                                         triple_bond,
                                                         force_field)
  result_dict['E_covalent'] = cov_pot

  [lone_pot, vlp] = calculate_lonpar_pot(species,
                                         atom_mask,
                                         abo,
                                         force_field)

  result_dict['E_lone_pair'] = lone_pot
  overunder_pot = calculate_ovcor_pot(species,
                                      atomic_numbers,
                                      atom_mask,
                                      close_nbr_inds,
                                      close_nbr_dists,
                                      close_nbr_inds != N,
                                      bo, bopi, bopi2, abo, vlp,
                                      force_field)

  result_dict['E_over_under'] = overunder_pot

  [val_pot,
   total_penalty,
   total_conj] = calculate_valency_pot(species,
                                       body_3_inds,
                                       body_3_angles,
                                       body_3_inds[:,0] != -1,
                                       close_nbr_inds,
                                       vlp,
                                       bo,bopi, bopi2, abo,
                                       force_field)

  result_dict['E_valency'] = val_pot
  result_dict['E_valency_penalty'] = total_penalty
  result_dict['E_valency_conj'] = total_conj

  [torsion_pot,
   tor_conj] = calculate_torsion_pot(species,
                                     body_4_inds,
                                     body_4_angles,
                                     body_4_inds[:,0] != -1,
                                     close_nbr_inds,
                                     bo,bopi,abo,
                                     force_field,
                                     tors_2013)

  result_dict['E_torsion'] = torsion_pot
  result_dict['E_torsion_conj'] = tor_conj
  result_dict['E_hbond'] = 0.0

  if hb_inds != None:

    hb_mask = (hb_inds[:,1] != -1) & (hb_inds[:,2] != -1)

    hb_pot = calculate_hb_pot(species,
                             hb_inds,
                             hb_ang_dist,
                             hb_mask,
                             close_nbr_inds,
                             far_nbr_inds,
                             bo,
                             force_field)
    result_dict['E_hbond'] = hb_pot

  return (cou_pot + vdw_pot + charge_pot
          + cov_pot + lone_pot + val_pot
          + total_penalty + total_conj
          + overunder_pot + tor_conj
            + torsion_pot + hb_pot +
            self_energy), charges

def calculate_eem_charges(species: Array,
                          atom_mask: Array,
                          nbr_inds: Array,
                          hulp2_mat: Array,
                          tapered_dists: Array,
                          idempotential: Array,
                          electronegativity: Array,
                          init_charges: Array = None,
                          total_charge: float = 0.0,
                          backprop_solve: bool = False,
                          tol: float = 1e-06,
                          max_solver_iter: int = 500):
  '''
  EEM charge solver
  If max_solver_iter is set to -1, use direct solve
  Returns:
    an array of shape [n+1,] where first n entries are the charges and
    last entry is the electronegativity equalization value
  '''

  if backprop_solve == False:
    tapered_dists = jax.lax.stop_gradient(tapered_dists)
    hulp2_mat = jax.lax.stop_gradient(hulp2_mat)
  prev_dtype = tapered_dists.dtype
  N = len(species)
  # might cause nan issues if 0s not handled well
  A = safe_mask(hulp2_mat != 0, lambda x: tapered_dists * 14.4 / x, hulp2_mat, 0.0)
  my_idemp = idempotential[species]
  my_elect = electronegativity[species] * atom_mask

  def to_dense():
    '''
    Create a dense matrix
    '''
    A_ = jax.vmap(lambda j: jax.vmap(lambda i: jnp.sum(A[i] * (nbr_inds[i] == j)))(jnp.arange(N)))(jnp.arange(N))
    A_ =  A_.at[jnp.diag_indices(N)].add(2.0 * my_idemp)
    matrix = jnp.zeros(shape=(N+1,N+1),dtype=prev_dtype)
    matrix = matrix.at[:N,:N].set(A_)
    matrix = matrix.at[N,:N].set(atom_mask)
    matrix = matrix.at[:N,N].set(atom_mask)
    matrix = matrix.at[N,N].set(0.0)
    return matrix

  mask = (nbr_inds != N)

  def SPMV_dense(vec):
    '''
    Matrix-free mat-vec
    '''
    res = jnp.zeros(shape=(N+1,), dtype=jnp.float64)
    s_vec = vec.astype(prev_dtype)[nbr_inds] * mask
    vals = jax.vmap(jnp.dot)(A, s_vec) + \
        (my_idemp * 2.0) * vec[:N] + vec[N]
    res = res.at[:N].set(vals * atom_mask)
    res = res.at[N].set(jnp.sum(vec[:N] * atom_mask))  # sum of charges
    return res

  b = jnp.zeros(shape=(N+1,), dtype=jnp.float64)
  b = b.at[:N].set(-1 * my_elect)
  b = b.at[N].set(total_charge)
  if max_solver_iter == -1:
    charges = jnp.linalg.solve(to_dense(), b)
  else:
    charges, conv_info = linalg.cg(SPMV_dense, b, x0=init_charges, tol=tol, maxiter=max_solver_iter)
  charges = charges.astype(prev_dtype)
  charges = charges.at[:-1].multiply(atom_mask)
  return charges

def calculate_acks2_charges(species: Array,
                          atom_mask: Array,
                          nbr_inds: Array,
                          hulp2_mat: Array,
                          bond_softness: Array,
                          tapered_dists: Array,
                          idempotential: Array,
                          electronegativity: Array,
                          total_charge: float,
                          backprop_solve: bool = False,
                          tol: float = 1e-06):
  if backprop_solve == False:
    tapered_dists = jax.lax.stop_gradient(tapered_dists)
    hulp2_mat = jax.lax.stop_gradient(hulp2_mat)
    bond_softness = jax.lax.stop_gradient(bond_softness)
  prev_dtype = tapered_dists.dtype
  N = len(species)
  # might cause nan issues if 0s not handled well
  A = jnp.where(hulp2_mat == 0, 0.0, tapered_dists * 14.4 / hulp2_mat)
  my_idemp = idempotential[species]
  my_elect = electronegativity[species]
  B = bond_softness

  def to_dense():
    a_inds = jnp.arange(N)
    A_ = jax.vmap(lambda j:
                  jax.vmap(lambda i:
                     jnp.sum(A[i] * (nbr_inds[i] == j)))(a_inds))(a_inds)
    diag_inds = jnp.diag_indices(N)
    A_ =  A_.at[diag_inds].add(2.0 * my_idemp)

    B_ = jax.vmap(lambda j:
                  jax.vmap(lambda i:
                     jnp.sum(B[i] * ((nbr_inds[i] == j)
                                     & (i != j))))(a_inds))(a_inds)
    diags_B = jnp.sum(B_, axis=0)
    B_ =  B_.at[diag_inds].add(-1 * diags_B)

    matrix = jnp.zeros(shape=(2*N+2,2*N+2),dtype=prev_dtype)
    matrix = matrix.at[:N,:N].set(A_)
    matrix = matrix.at[N:2*N,N:2*N].set(B_)

    matrix = matrix.at[N:2*N,:N].set(jnp.eye(N))
    matrix = matrix.at[:N,N:2*N].set(jnp.eye(N))

    matrix = matrix.at[2*N,N:2*N].set(atom_mask)
    matrix = matrix.at[N:2*N,2*N].set(atom_mask)

    matrix = matrix.at[2*N+1,:N].set(atom_mask)
    matrix = matrix.at[:N,2*N+1].set(atom_mask)

    return matrix


  def SPMV_dense(vec):
    res = jnp.zeros(shape=(N+1,), dtype=jnp.float64)
    s_vec = vec.astype(prev_dtype)[nbr_inds] * (nbr_inds != N)
    vals = jax.vmap(jnp.dot)(A, s_vec) + \
        (my_idemp * 2.0) * vec[:N] + vec[N]
    res = res.at[:N].set(vals)
    res = res.at[N].set(jnp.sum(vec[:N]))  # sum of charges
    return res

  b = jnp.zeros(shape=(2*N+2,), dtype=jnp.float64)
  b = b.at[:N].set(-1 * my_elect)
  b = b.at[N:2*N].set(total_charge / N)
  b = b.at[2*N].set(0.0)
  b = b.at[2*N+1].set(total_charge)
  #matrix = to_dense()
  #print(matrix)
  #charges = jnp.linalg.solve(matrix, b)
  charges, conv_info = linalg.cg(SPMV_dense, b, tol=tol, maxiter=9999)
  charges = charges.astype(prev_dtype)
  return charges[:N] * atom_mask, charges[N:2*N] * atom_mask


def calculate_coulomb_pot(nbr_inds: Array,
                          atom_mask: Array,
                          hulp2_mat: Array,
                          tapered_dists: Array,
                          charges: Array):
  N = len(atom_mask)
  mask = (atom_mask.reshape(-1, 1) * atom_mask[nbr_inds]) * (nbr_inds != N)
  charge_mat = charges.reshape(-1, 1) * charges[nbr_inds]
  eph_mat = safe_mask(mask,
                      lambda x: c1c * charge_mat / (x + 1e-20), hulp2_mat, 0.0)
  ephtap_mat = eph_mat * tapered_dists * mask
  total_pot = high_precision_sum(ephtap_mat) / 2.0

  return total_pot


def calculate_charge_energy(species: Array,
                            charges: Array,
                            idempotential: Array,
                            electronegativity: Array):

  ech = high_precision_sum(23.02 * (electronegativity[species]
                                    * charges
                                    + idempotential[species]
                                    * jnp.square(charges)))
  return ech

def calculate_acks2_coulomb_pot(nbr_inds: Array,
                                atom_mask: Array,
                                effpot: Array,
                                bond_softness: Array):
  hulp2 = effpot.reshape(-1, 1) - effpot[nbr_inds]
  eph = (-0.25 * 23.02) * bond_softness * hulp2 * hulp2

  return high_precision_sum(eph)

def calculate_charge_energy_acks2(charges: Array,
                                  effpot: Array):
  ech_acks2 = high_precision_sum(23.02 * charges * effpot)
  return ech_acks2


def calculate_vdw_pot(species: Array,
                      far_nbr_mask: Array,
                      nbr_inds: Array,
                      dists: Array,
                      tapered_dists: Array,
                      force_field: ForceField):
  N = len(species)
  neigh_types = species[nbr_inds]
  vop = jnp.power(force_field.vop.reshape(-1, 1), force_field.vdw_shiedling/2.0)
  gamwh_mat = vop * vop.transpose()
  #gamwh_mat = (1.0 / gamwh_mat) ** force_field.vdw_shiedling
  gamwh_mat = 1.0 / gamwh_mat
  gamwco_mat = gamwh_mat[neigh_types, species.reshape(-1, 1)]
  # select the required values
  p1_mat = force_field.p1co[neigh_types, species.reshape(-1, 1)]
  p2_mat = force_field.p2co[neigh_types, species.reshape(-1, 1)]
  p3_mat = force_field.p3co[neigh_types, species.reshape(-1, 1)]
  hulpw_mat = safe_mask(dists > 0, lambda x: x ** force_field.vdw_shiedling, dists, 0.0) + gamwco_mat
  rrw_mat = jnp.power(hulpw_mat, (1.0 / force_field.vdw_shiedling))
  # if p = 0 -> gradient will be 0
  temp_val2 = p3_mat * ((1.0 - rrw_mat / p1_mat))
  # gradient nan issue fix
  h1_mat = jnp.exp(temp_val2)
  h2_mat = jnp.exp(0.5 * temp_val2)
  ewh_mat = p2_mat * (h1_mat - 2.0 * h2_mat)
  ewhtap_mat = ewh_mat * tapered_dists
  ewhtap_mat = ewhtap_mat * far_nbr_mask
  total_pot = high_precision_sum(ewhtap_mat) / 2.0

  return total_pot


def calculate_bo(nbr_inds: Array,
                 nbr_dist: Array,
                 species: Array,
                 species_AN: Array,
                 force_field: ForceField):
  '''
  Usage:
      first update/allocate neighborlist will be called
      then the info will be passed to this function

      for now, assume the format is "Dense"
  '''
  N = len(species)
  atomic_num1 = species_AN.reshape(-1, 1)
  atomic_num2 = species_AN[nbr_inds]
  # O: 8, C:6
  triple_bond1 = jnp.logical_and(atomic_num1 == 8, atomic_num2 == 6)
  triple_bond2 = jnp.logical_and(atomic_num1 == 6, atomic_num2 == 8)
  triple_bond = jnp.logical_or(triple_bond1, triple_bond2)

  [cov_pot, bo, bopi, bopi2, abo] = calculate_covbon_pot(nbr_inds,
                                                         nbr_dist,
                                                         nbr_inds != N,
                                                         species,
                                                         triple_bond,
                                                         force_field)

  return bo


def calculate_covbon_pot(nbr_inds: Array,
                         nbr_dist: Array,
                         nbr_mask: Array,
                         species: Array,
                         triple_bond: Array,
                         force_field: ForceField,
                         tapered_reaxff: bool = False):
  N = len(species)
  nbr_mask = nbr_mask & (nbr_dist > 0)

  neigh_types = species[nbr_inds]
  atom_inds = jnp.arange(N).reshape(-1, 1)
  species = species.reshape(-1, 1)
  # save the chosen dtype
  dtype = nbr_dist.dtype
  #symm = (atom_inds == nbr_inds).astype(dtype) + 1
  #symm = 1.0 / symm
  # since we store the close nbr list full, we later divide the summation by 2
  # to compansate double counting, the self bonds are not double counted
  # so they will be multipled by 0.5 as expected
  symm = 1.0
  my_rob1 = force_field.rob1[neigh_types, species]
  my_rob2 = force_field.rob2[neigh_types, species]
  my_rob3 = force_field.rob3[neigh_types, species]
  my_ptp = force_field.ptp[neigh_types, species]
  my_pdp = force_field.pdp[neigh_types, species]
  my_popi = force_field.popi[neigh_types, species]
  my_pdo = force_field.pdo[neigh_types, species]
  my_bop1 = force_field.bop1[neigh_types, species]
  my_bop2 = force_field.bop2[neigh_types, species]
  my_de1 = force_field.de1[neigh_types, species]
  my_de2 = force_field.de2[neigh_types, species]
  my_de3 = force_field.de3[neigh_types, species]
  my_psp = force_field.psp[neigh_types, species]
  my_psi = force_field.psi[neigh_types, species]

  # TODO: tempo fix, due to numerical problems in this function
  # use double precision then cast it back to the original type
  nbr_dist = nbr_dist.astype(jnp.float64)

  rhulp = safe_mask(my_rob1 > 0, lambda x: nbr_dist / (x+1e-10), my_rob1, 1e-7)
  rhulp2 = safe_mask(my_rob2 > 0, lambda x: nbr_dist / (x+1e-10), my_rob2, 1e-7)
  rhulp3 = safe_mask(my_rob3 > 0, lambda x: nbr_dist / (x+1e-10), my_rob3, 1e-7)
  rh2p = rhulp2 ** my_ptp
  ehulpp = jnp.exp(my_pdp * rh2p)

  rh2pp = rhulp3 ** my_popi
  ehulppp = jnp.exp(my_pdo * rh2pp)

  rh2 = rhulp ** my_bop2
  ehulp = (1 + force_field.cutoff) * jnp.exp(my_bop1 * rh2)

  mask1 = (my_rob1 > 0) & nbr_mask
  mask2 = (my_rob2 > 0) & nbr_mask
  mask3 = (my_rob3 > 0) & nbr_mask
  full_mask = mask1 | mask2 | mask3

  ehulp = safe_mask(mask1, lambda x: x, ehulp, 0)
  ehulpp = safe_mask(mask2, lambda x: x, ehulpp, 0)
  ehulppp = safe_mask(mask3, lambda x: x, ehulppp, 0)

  bor = ehulp + ehulpp + ehulppp

  bopi = ehulpp
  bopi2 = ehulppp
  if tapered_reaxff:
    bo = (taper_inc(bor, force_field.cutoff, 4.0*force_field.cutoff)
          * (bor - force_field.cutoff))
  else:
    bo = bor - force_field.cutoff
  bo = jnp.where(bo <= 0, 0.0, bo)
  abo = jnp.sum(bo, axis=1)

  bo, bopi, bopi2 = calculate_boncor_pot(nbr_inds,
                                         nbr_mask,
                                         species.flatten(),
                                         bo, bopi, bopi2, abo,
                                         force_field)

  abo = jnp.sum(bo * nbr_mask, axis=1)

  bosia = bo - bopi - bopi2
  bosia = jnp.clip(bosia, 0, float('inf'))
  de1h = symm * my_de1
  de2h = symm * my_de2
  de3h = symm * my_de3
  # add 1e-20 so that ln(a) is not nan
  bopo1 = safe_mask((bosia != 0), lambda x: (x + 1e-20) ** my_psp, bosia, 0)
  exphu1 = jnp.exp(my_psi * (1.0 - bopo1))
  ebh = -de1h * bosia * exphu1 - de2h * bopi - de3h * bopi2
  ebh = jnp.where(bo <= 0, 0.0, ebh)
  # Stabilisation terminal triple bond in CO
  ba = (bo - 2.5) * (bo - 2.5)
  exphu = jnp.exp(-force_field.trip_stab8 * ba)

  abo_j2 = abo[nbr_inds]
  abo_j1 = abo[atom_inds]

  obo_a = abo_j1 - bo
  obo_b = abo_j2 - bo

  exphua1 = jnp.exp(-force_field.trip_stab4*obo_a)
  exphub1 = jnp.exp(-force_field.trip_stab4*obo_b)

  my_aval = force_field.aval[species] + force_field.aval[neigh_types]

  triple_bond = jnp.where(bo < 1.0, 0.0, triple_bond)
  ovoab = abo_j1 + abo_j2 - my_aval
  exphuov = jnp.exp(force_field.trip_stab5 * ovoab)

  hulpov = 1.0/(1.0+25.0*exphuov)

  estriph = force_field.trip_stab11*exphu*hulpov*(exphua1+exphub1)

  eb = (ebh + estriph * triple_bond)
  eb = safe_mask(full_mask, lambda x: x, eb, 0)

  cov_pot = high_precision_sum(eb) / 2.0
  # cast the arrays back to the original dtype
  cov_pot = cov_pot.astype(dtype)
  #bo = bo.astype(dtype)
  #bopi = bopi.astype(dtype)
  #bopi2 = bopi2.astype(dtype)
  #abo = abo.astype(dtype)

  symm = (atom_inds == nbr_inds).astype(dtype) + 1
  symm = 1.0 / symm
  # to correct for self bonds, multiply by 0.5
  #bo = bo * symm
  #bopi = bopi * symm
  #bopi2 = bopi2 * symm

  return [cov_pot, bo, bopi, bopi2, abo]


def calculate_boncor_pot(nbr_inds: Array,
                         nbr_mask: Array,
                         species: Array,
                         bo: Array,
                         bopi: Array,
                         bopi2: Array,
                         abo: Array,
                         force_field: ForceField):

  neigh_types = species[nbr_inds]
  species = species.reshape(-1, 1)

  abo_j2 = abo[nbr_inds]
  abo_j1 = abo.reshape(-1, 1)

  aval_j2 = force_field.aval[neigh_types]
  aval_j1 = force_field.aval[species]

  vp131 = safe_sqrt(force_field.bo131[species] * force_field.bo131[neigh_types])
  vp132 = safe_sqrt(force_field.bo132[species] * force_field.bo132[neigh_types])
  vp133 = safe_sqrt(force_field.bo133[species] * force_field.bo133[neigh_types])

  my_ovc = force_field.ovc[neigh_types, species]

  ov_j1 = abo_j1 - aval_j1
  ov_j2 = abo_j2 - aval_j2

  exp11 = jnp.exp(-force_field.over_coord1*ov_j1)
  exp21 = jnp.exp(-force_field.over_coord1*ov_j2)
  exphu1 = jnp.exp(-force_field.over_coord2*ov_j1)
  exphu2 = jnp.exp(-force_field.over_coord2*ov_j2)
  exphu12 = (exphu1+exphu2)

  ovcor = -(1.0/force_field.over_coord2) * jnp.log(0.50*exphu12)
  huli = aval_j1+exp11+exp21
  hulj = aval_j2+exp11+exp21

  corr1 = huli/(huli+ovcor)
  corr2 = hulj/(hulj+ovcor)
  corrtot = 0.50*(corr1+corr2)

  corrtot = jnp.where(my_ovc > 0.001, corrtot, 1.0)

  my_v13cor = force_field.v13cor[neigh_types, species]

  # update vval3 based on amas value
  vval3 = jnp.where(force_field.amas < 21.0,
                   force_field.valf,
                   force_field.vval3)

  vval3_j1 = vval3[species]
  vval3_j2 = vval3[neigh_types]
  ov_j11 = abo_j1 - vval3_j1
  ov_j22 = abo_j2 - vval3_j2
  cor1 = vp131 * bo * bo - ov_j11
  cor2 = vp131 * bo * bo - ov_j22

  exphu3 = jnp.exp(-vp132 * cor1 + vp133)
  exphu4 = jnp.exp(-vp132 * cor2 + vp133)
  bocor1 = 1.0/(1.0+exphu3)
  bocor2 = 1.0/(1.0+exphu4)

  bocor1 = jnp.where(my_v13cor > 0.001, bocor1, 1.0)
  bocor2 = jnp.where(my_v13cor > 0.001, bocor2, 1.0)

  bo = bo * corrtot * bocor1 * bocor2
  threshold = 0.0 # fortran threshold: 1e-10
  bo = safe_mask(nbr_mask & (bo > threshold), lambda x: x, bo, 0)
  corrtot2 = corrtot*corrtot
  bopi = bopi*corrtot2*bocor1*bocor2
  bopi2 = bopi2*corrtot2*bocor1*bocor2

  bopi = safe_mask(nbr_mask & (bopi > threshold), lambda x: x, bopi, 0)
  bopi2 = safe_mask(nbr_mask & (bopi2 > threshold), lambda x: x, bopi2, 0)

  return bo, bopi, bopi2


def smooth_lone_pair_casting(number,
                             p_lambda=0.9999,
                             l1=-1.3,
                             l2=-0.3,
                             r1=0.3,
                             r2=1.3):

  part_2 = (1/jnp.pi)*(jnp.arctan(p_lambda *
                                  jnp.sin(2*jnp.pi * number) /
                                  (p_lambda * jnp.cos(2*jnp.pi * number) - 1)))

  f_R = number - 1/2 -part_2

  f_L = number + 1/2 - part_2

  result = jnp.where(number < l1, f_L,
                     jnp.where(number < l2, f_L * taper(number, l1, l2),
                               jnp.where(number < r1, 0,
                                         jnp.where(number <= r2,
                                                   f_R * taper_inc(number, r1, r2),
                                                   f_R))))

  return result


def calculate_lonpar_pot(species: Array,
                         atom_mask: Array,
                         abo: Array,
                         force_field: ForceField):
  # handle this part in double preicison
  prev_type = abo.dtype
  #abo = abo.astype(jnp.float64)
  # Determine number of lone pairs on atoms
  voptlp = 0.5 * (force_field.stlp[species] - force_field.aval[species])
  vund = abo - force_field.stlp[species]
  #vund_div2 = smooth_lone_pair_casting(vund/2.0) # (vund/2.0).astype(np.int32)
  vund_div2 = (vund/2.0).astype(jnp.int32).astype(prev_type)
  vlph = 2.0 * vund_div2
  vlpex = vund - vlph
  expvlp = jnp.exp(-force_field.par_16 * (2.0 + vlpex) * (2.0 + vlpex))
  vlp = expvlp - vund_div2

  # Calculate lone pair energy
  diffvlp = voptlp-vlp
  exphu1 = jnp.exp(-75.0*diffvlp)
  hulp1 = 1.0/(1.0+exphu1)
  elph = force_field.vlp1[species] * diffvlp * hulp1
  elph = safe_mask(atom_mask, lambda x: x, elph, 0)
  elp = high_precision_sum(elph)
  elp = elp.astype(prev_type)
  vlp = vlp.astype(prev_type)

  return [elp, vlp]


def calculate_ovcor_pot(species: Array,
                        atoms_AN: Array,
                        atom_mask: Array,
                        nbr_inds: Array,
                        nbr_dists: Array,
                        nbr_mask: Array,
                        bo: Array,
                        bopi: Array,
                        bopi2: Array,
                        abo: Array,
                        vlp: Array,
                        force_field: ForceField):
  my_stlp = force_field.stlp[species]
  my_aval = force_field.aval[species]
  my_amas = force_field.amas[species]
  my_valp1 = force_field.valp1[species]
  my_vovun = force_field.vovun[species]
  neigh_types = species[nbr_inds]
  # this function is numerically sensitive so use double precision
  prev_type = nbr_dists.dtype
  #bo = bo.astype(jnp.float64)
  #bopi = bopi.astype(jnp.float64)
  #bopi2 = bopi2.astype(jnp.float64)
  #abo = abo.astype(jnp.float64)


  vlptemp = jnp.where(my_amas > 21.0, 0.50*(my_stlp-my_aval), vlp)
  dfvl = jnp.where(my_amas > 21.0, 0.0, 1.0)
  #  Calculate overcoordination energy
  #  Valency is corrected for lone pairs
  voptlp = 0.50*(my_stlp-my_aval)
  vlph = (voptlp-vlptemp)
  diffvlph = dfvl*vlph
  diffvlp2 = dfvl.reshape(-1,1) * vlph[nbr_inds]
  # Determine coordination neighboring atoms
  part_1 = bopi + bopi2
  part_2 = abo[nbr_inds] - force_field.aval[neigh_types] - diffvlp2
  sumov = jnp.sum(part_1 * part_2, axis=1)
  mult_vov_de1 = force_field.vover * force_field.de1
  my_mult_vov_de1 = mult_vov_de1[species.reshape(-1, 1), neigh_types]

  sumov2 = jnp.sum(my_mult_vov_de1 * bo, axis=1)
  # Gradient non issue fix
  exphu1 = jnp.exp(force_field.par_32 * sumov)
  vho = 1.0 / (1.0+force_field.par_33*exphu1)
  diffvlp = diffvlph * vho

  vov1 = abo - my_aval - diffvlp
  # to solve the nan issue
  exphuo = jnp.exp(my_vovun*vov1)
  hulpo = 1.0/(1.0+exphuo)

  hulpp = (1.0/(vov1+my_aval+1e-08))

  eah = sumov2*hulpp*hulpo*vov1

  ea = high_precision_sum(eah * atom_mask)

  # Calculate undercoordination energy
  # Gradient non issue fix
  exphu2 = jnp.exp(force_field.par_10*sumov)
  vuhu1 = 1.0+force_field.par_9*exphu2
  hulpu2 = 1.0/vuhu1

  exphu3 = -jnp.exp(force_field.par_7*vov1)
  hulpu3 = -(1.0+exphu3)

  dise2 = my_valp1
  # Gradient non issue fix
  exphuu = jnp.exp(-my_vovun*vov1)
  hulpu = 1.0/(1.0+exphuu)
  eahu = dise2*hulpu*hulpu2*hulpu3

  eahu = jnp.where(my_valp1 < 0, 0, eahu)
  eahu = safe_mask(atom_mask, lambda x: x, eahu, 0)
  ea = ea + high_precision_sum(eahu * atom_mask)
  # cast the result back to the original type
  ea = ea.astype(prev_type)

  #Correction for C2 PART effecting (eplh)
  # TODO: Most FFs do not activate this part, so I should use lax.cond
  # to decide if the computation is needed, commented off for now
  '''
  par6_mask = jnp.abs(force_field.par_6) > 0.001
  src_C_mask = atoms_AN == 6
  dst_C_mask = atoms_AN[nbr_inds] == 6
  C_C_bonds_mask = src_C_mask.reshape(-1,1) & dst_C_mask
  C_C_bonds_mask = C_C_bonds_mask & nbr_mask & par6_mask
  vov4 = abo - my_aval
  vov4 = vov4[nbr_inds]
  vov3 = bo - vov4 - 0.040 * (vov4 ** 4)
  vov3_mask = vov3 > 3.0
  elph = force_field.par_6 * (vov3 -3.0)**2
  elph = elph * (vov3_mask & C_C_bonds_mask)
  c2_corr = high_precision_sum(elph)
  ea = ea + c2_corr
  '''
  return ea


def calculate_valency_pot(species: Array,
                          body_3_inds: Array,
                          body_3_angles: Array,
                          body_3_mask: Array,
                          nbr_inds: Array,
                          vlp: Array,
                          bo: Array,
                          bopi: Array,
                          bopi2: Array,
                          abo: Array,
                          force_field: ForceField,
                          tapered_reaxff: bool = False):
  prev_type = bo.dtype
  center = body_3_inds[:, 0]
  neigh1_lcl = body_3_inds[:, 1]
  neigh2_lcl = body_3_inds[:, 2]
  neigh1_glb = nbr_inds[center, neigh1_lcl]
  neigh2_glb = nbr_inds[center, neigh2_lcl]

  cent_types = species[center]
  neigh1_types = species[neigh1_glb]
  neigh2_types = species[neigh2_glb]

  val_angles = body_3_angles

  boa = bo[center, neigh1_lcl]
  bob = bo[center, neigh2_lcl]

  if tapered_reaxff:
    boa = (taper_inc(boa, force_field.cutoff2, 4.0*force_field.cutoff2)
           * (boa - force_field.cutoff2))
    bob = (taper_inc(bob, force_field.cutoff2, 4.0*force_field.cutoff2)
           * (bob - force_field.cutoff2))
    complete_mask = body_3_mask
  else:
    # Fortan comment: Scott Habershon recommendation March 2009
    mask = jnp.where(boa * bob < 0.00001, 0, 1)
    complete_mask = mask * body_3_mask
    boa = boa - force_field.cutoff2
    bob = bob - force_field.cutoff2
    complete_mask = complete_mask & (boa > 0) & (bob > 0)

  # thresholding
  boa = jnp.clip(boa, 0, float('inf'))
  bob = jnp.clip(bob, 0, float('inf'))
  # calculate SBO term
  # calculate sbo2 and vmbo for every atom in the sim.sys.
  sbo2 = jnp.sum(bopi, axis=1) + jnp.sum(bopi2, axis=1)
  vmbo = jnp.prod(jnp.exp(-bo ** 8),
                  dtype=jnp.float64, axis=1)#.astype(prev_type)

  my_abo = abo[center]
  # calculate for every atom
  exbo = abo - force_field.valf[species]
  my_exbo = exbo[center]
  # TODO: (REVISE LATER) cast the data to double to solve nan issue in division
  my_exbo = jnp.array(my_exbo, dtype=jnp.float64)
  my_vkac = force_field.vkac[neigh1_types, cent_types, neigh2_types]
  evboadj = 1.0  # why?
  # to solve the nan issue, clip the vlaues
  expun = jnp.exp(-my_vkac * my_exbo)
  expun2 = jnp.exp(force_field.val_par15 * my_exbo)

  htun1 = 2.0 + expun2
  htun2 = 1.0 + expun + expun2
  my_vval4 = force_field.vval4[cent_types]

  evboadj2 = my_vval4-(my_vval4-1.0)*(htun1/htun2)
  evboadj2 = jnp.array(evboadj2, dtype=prev_type)
  # calculate for every atom
  exlp1 = abo - force_field.stlp[species]
  exlp2 = 2.0 * ((exlp1/2.0).astype(jnp.int32))  # integer casting
  #exlp2 = 2.0 * smooth_lone_pair_casting(exlp1/2.0)
  exlp = exlp1 - exlp2
  vlpadj = jnp.where(exlp < 0.0, vlp, 0.0)  # vlp comes from lone pair
  # calculate for every atom
  sbo2 = sbo2 + (1 - vmbo) * (-exbo - force_field.val_par34 * vlpadj)
  sbo2 = jnp.clip(sbo2, 0, 2.0)
  # add 1e-20 so that ln(a) is not nan
  sbo2 = vectorized_cond(sbo2 < 1,
                         lambda x: (x  + 1e-15) ** force_field.val_par17,
                         lambda x: sbo2, sbo2)

  sbo2 = vectorized_cond(sbo2 >= 1,
                         lambda x: 2.0-(2.0-x + 1e-15)**force_field.val_par17,
                         lambda x: sbo2, sbo2)

  expsbo = jnp.exp(-force_field.val_par18*(2.0-sbo2))

  my_expsbo = expsbo[center]
  thba = force_field.th0[neigh1_types, cent_types, neigh2_types]

  thetao = 180.0 - thba * (1.0-my_expsbo)
  thetao = thetao * dgrrdn
  thdif = (thetao - val_angles)
  thdi2 = thdif * thdif

  my_vka = force_field.vka[neigh1_types, cent_types, neigh2_types]
  my_vka3 = force_field.vka3[neigh1_types, cent_types, neigh2_types]
  exphu = my_vka * jnp.exp(-my_vka3 * thdi2)

  exphu2 = my_vka - exphu
  # To avoid linear Me-H-Me angles (6/6/06)
  exphu2 = jnp.where(my_vka < 0.0, exphu2 - my_vka, exphu2)

  my_vval2 = force_field.vval2[neigh1_types, cent_types, neigh2_types]
  # add 1e-20 so that ln(a) is not nan
  boap = (boa + 1e-20) ** my_vval2
  bobp = (bob + 1e-20) ** my_vval2

  my_vval1 = force_field.vval1[cent_types]

  exa = jnp.exp(-my_vval1*boap)
  exb = jnp.exp(-my_vval1*bobp)

  exa2 = (1.0-exa)
  exb2 = (1.0-exb)

  evh = evboadj2*evboadj*exa2*exb2*exphu2
  evh = safe_mask(complete_mask, lambda x: x, evh, 0)

  total_pot = high_precision_sum(evh*complete_mask).astype(prev_type)

  # Calculate penalty for two double bonds in valency angle
  exbo = abo - force_field.aval[species]
  expov = jnp.exp(force_field.val_par22 * exbo)
  expov2 = jnp.exp(-force_field.val_par21 * exbo)

  htov1 = 2.0+expov2
  htov2 = 1.0+expov+expov2

  ecsboadj = htov1/htov2

  ecsboadj = jnp.array(ecsboadj, dtype=prev_type)
  my_ecsboadj = ecsboadj[center]  # for the center atom

  my_vkap = force_field.vkap[neigh1_types, cent_types, neigh2_types]
  exphu1 = jnp.exp(-force_field.val_par20*(boa-2.0)*(boa-2.0))
  exphu2 = jnp.exp(-force_field.val_par20*(bob-2.0)*(bob-2.0))
  epenh = my_vkap*my_ecsboadj*exphu1*exphu2

  epenh = safe_mask(complete_mask, lambda x: x, epenh, 0)
  total_penalty = high_precision_sum(epenh).astype(prev_type)

  # Calculate valency angle conjugation energy
  abo_i = abo[neigh1_glb]

  abo_k = abo[neigh2_glb]  # (i,j,k) will give abo for k

  unda = abo_i - boa
  ovb = my_abo - force_field.vval3[cent_types]

  undc = abo_k - bob
  ba = (boa-1.50)*(boa-1.50)
  bb = (bob-1.50)*(bob-1.50)

  exphua = jnp.exp(-force_field.val_par31*ba)
  exphub = jnp.exp(-force_field.val_par31*bb)
  exphuua = jnp.exp(-force_field.val_par39*unda*unda)
  exphuob = jnp.exp(force_field.val_par3*ovb)
  exphuob = jnp.array(exphuob,dtype=jnp.float64)
  exphuuc = jnp.exp(-force_field.val_par39*undc*undc)
  hulpob = 1.0/(1.0+exphuob)
  hulpob = jnp.array(hulpob, dtype=prev_type)
  my_vka8 = force_field.vka8[neigh1_types, cent_types, neigh2_types]
  ecoah = my_vka8*exphua*exphub*exphuua*exphuuc*hulpob

  ecoah = safe_mask(complete_mask, lambda x: x, ecoah, 0)
  total_conj = high_precision_sum(ecoah).astype(prev_type)
  return [total_pot, total_penalty, total_conj]


def calculate_torsion_pot(species: Array,
                          body_4_inds: Array,
                          body_4_angles: Array,
                          body_4_mask: Array,
                          nbr_inds: Array,
                          bo: Array,
                          bopi: Array,
                          abo: Array,
                          force_field: ForceField,
                          tapered_reaxff: bool = False,
                          tors_2013: bool = False):
  hsin = body_4_angles[0]  # hsin = sinhd * sinhe
  arg = body_4_angles[1]
  prev_type = hsin.dtype
  #bo = bo.astype(jnp.float64)
  # left : nbr_inds[ind2][n21]      or nbr_inds[center1][body_4_inds[:,1]]
  # center1: ind2                   or body_4_inds[:,0]
  # center2: neigh_inds[ind2][n22]  or nbr_inds[center1][body_4_inds[:,2]]
  # right: nbr_inds[center2][n31]   or nbr_inds[center2][body_4_inds[:,3]]
  center1_glb = body_4_inds[:, 0]
  left_lcl = body_4_inds[:, 1]  # local to center1
  left_glb = nbr_inds[center1_glb, left_lcl]
  center2_lcl = body_4_inds[:, 2]  # local to center1
  center2_glb = nbr_inds[center1_glb, center2_lcl]
  right_lcl = body_4_inds[:, 3]  # local to center2
  right_glb = nbr_inds[center2_glb, right_lcl]

  my_v1 = force_field.v1[species[left_glb],
                         species[center1_glb],
                         species[center2_glb],
                         species[right_glb]]
  my_v2 = force_field.v2[species[left_glb],
                         species[center1_glb],
                         species[center2_glb],
                         species[right_glb]]
  my_v3 = force_field.v3[species[left_glb],
                         species[center1_glb],
                         species[center2_glb],
                         species[right_glb]]
  my_v4 = force_field.v4[species[left_glb],
                         species[center1_glb],
                         species[center2_glb],
                         species[right_glb]]
  my_vconj = force_field.vconj[species[left_glb],
                         species[center1_glb],
                         species[center2_glb],
                         species[right_glb]]

  exbo1 = abo - force_field.valf[species]

  exbo1_2 = exbo1[center1_glb]  # center1
  exbo2_3 = exbo1[center2_glb]  # center2
  htovt = exbo1_2 + exbo2_3
  expov = jnp.exp(force_field.par_26 * htovt)
  expov2 = jnp.exp(-force_field.par_25 * htovt)
  htov1 = 2.0 + expov2
  htov2 = 1.0 + expov + expov2
  etboadj = htov1 / htov2
  etboadj = jnp.array(etboadj, dtype=prev_type)

  bo2t = 2.0 - bopi[center1_glb, center2_lcl] - etboadj
  bo2p = bo2t * bo2t

  bocor2 = jnp.exp(my_v4 * bo2p)

  arg2 = arg * arg
  ethhulp = (0.5 * my_v1 * (1.0 + arg) + my_v2 * bocor2 * (1.0 - arg2) +
             my_v3 * (0.5 + 2.0*arg2*arg - 1.5*arg))
  boa = bo[center1_glb, left_lcl]
  bob = bo[center1_glb, center2_lcl]
  boc = bo[center2_glb, right_lcl]

  mult_bo_mask = jnp.where(boa * bob * boc > force_field.cutoff2, 1, 0)

  complete_mask = body_4_mask * mult_bo_mask

  if tapered_reaxff:
    boa = (taper_inc(boa, force_field.cutoff2, 4.0*force_field.cutoff2)
           * (boa - force_field.cutoff2))
    bob = (taper_inc(bob, force_field.cutoff2, 4.0*force_field.cutoff2)
           * (bob - force_field.cutoff2))
    boc = (taper_inc(boc, force_field.cutoff2, 4.0*force_field.cutoff2)
           * (boc - force_field.cutoff2))
  else:
    boa = boa - force_field.cutoff2
    bob = bob - force_field.cutoff2
    boc = boc - force_field.cutoff2


  bo_mask = jnp.where(boa > 0, 1, 0)
  bo_mask = jnp.where(bob > 0, bo_mask, 0)
  bo_mask = jnp.where(boc > 0, bo_mask, 0)
  complete_mask = bo_mask * complete_mask

  if tors_2013:
    exphua = jnp.exp(-2*force_field.par_24 * boa**2)
    exphub = jnp.exp(-2*force_field.par_24 * bob**2)
    exphuc = jnp.exp(-2*force_field.par_24 * boc**2)
  else:
    exphua = jnp.exp(-force_field.par_24 * boa)
    exphub = jnp.exp(-force_field.par_24 * bob)
    exphuc = jnp.exp(-force_field.par_24 * boc)
  bocor4 = (1.0 - exphua) * (1.0 - exphub) * (1.0 - exphuc)

  eth = hsin * ethhulp * bocor4

  eth = safe_mask(complete_mask, lambda x: x, eth, 0)

  tors_pot = high_precision_sum(eth).astype(prev_type)
  # calculate conjugation pot
  ba = (boa-1.50)*(boa-1.50)
  bb = (bob-1.50)*(bob-1.50)
  bc = (boc-1.50)*(boc-1.50)

  exphua1 = jnp.exp(-force_field.par_28*ba)
  exphub1 = jnp.exp(-force_field.par_28*bb)
  exphuc1 = jnp.exp(-force_field.par_28*bc)
  sbo = exphua1*exphub1*exphuc1

  arghu0 = (arg2-1.0) * hsin  # hsin = sinhd*sinhe
  ehulp = my_vconj*(arghu0+1.0)

  ecoh = ehulp*sbo
  ecoh = safe_mask(complete_mask, lambda x: x, ecoh, 0)
  conj_pot = high_precision_sum(ecoh).astype(prev_type)
  return [tors_pot, conj_pot]

def calculate_hb_pot(species: Array,
                     hbond_inds: Array,
                     hbond_angles: Array,
                     hbond_mask: Array,
                     close_nbr_inds: Array,
                     far_nbr_inds: Array,
                     bo: Array,
                     force_field: ForceField,
                     tapered_reaxff: bool = False):
  # inds: donor ind, local acceptor ind (close neigh.), local ind_2 (far neigh)
  angles = hbond_angles[0,:]
  dists = hbond_angles[1,:]
  prev_type = dists.dtype
  glb_center = hbond_inds[:,0]
  lcl_close_nbr = hbond_inds[:,1]
  glb_close_nbr = close_nbr_inds[glb_center,lcl_close_nbr]
  lcl_far_nbr = hbond_inds[:,2]
  glb_far_nbr = far_nbr_inds[glb_center,lcl_far_nbr]

  cent_types = species[glb_center]
  close_nbr_types = species[glb_close_nbr]
  far_nbr_types = species[glb_far_nbr]



  my_rhb = force_field.rhb[close_nbr_types,cent_types,far_nbr_types]
  my_dehb = force_field.dehb[close_nbr_types,cent_types,far_nbr_types]
  my_vhb1 = force_field.vhb1[close_nbr_types,cent_types,far_nbr_types]
  my_vhb2 = force_field.vhb2[close_nbr_types,cent_types,far_nbr_types]
  bo = bo.astype(prev_type)
  boa = bo[glb_center,lcl_close_nbr]
  if tapered_reaxff:
    boa_mult = taper_inc(boa, 0.01, 4.0*0.01)
    dist_mult = taper(boa, 0.9 * 7.5, 7.5)
  else:
    boa_mult = 1.0
    dist_mult = 1.0
  boa = jnp.where(boa > 0.01, boa, 0.0)


  hbond_mask = hbond_mask & (dists < 7.5) & (dists > 0.0)
  my_rhb = my_rhb + 1e-10
  dists = dists + 1e-10
  # to not get divide by zero
  rhu1 = my_rhb / dists
  rhu2 = dists / my_rhb

  exphu1 = jnp.exp(-my_vhb1 * boa)
  exphu2 = jnp.exp(-my_vhb2 * (rhu1 + rhu2 - 2.0))

  ehbh = ((1.0-exphu1)
          * my_dehb * exphu2
          * jnp.power(jnp.sin((angles + 1e-10)/2.0), 4)) * boa_mult * dist_mult
  ehbh = safe_mask(hbond_mask, lambda x: x, ehbh, 0)
  hb_pot = high_precision_sum(ehbh).astype(prev_type)


  return hb_pot

def taper(value, low_tap_rad, up_tap_rad):
  '''
  Decreasing tapering function
  1 at low_tap_rad and 0 at up_tap_rad
  smoothly taper the value in between
  '''
  R = value - low_tap_rad
  up_tap_rad = up_tap_rad - low_tap_rad
  low_tap_rad = 0.0


  R2 = R * R
  R3 = R2 * R

  SWB = up_tap_rad
  SWA = low_tap_rad

  D1 = SWB-SWA
  D7 = D1**7.0
  SWA2 = SWA*SWA
  SWA3 = SWA2*SWA
  SWB2 = SWB*SWB
  SWB3 = SWB2*SWB

  SWC7 = 20.0
  SWC6 = -70.0*(SWA+SWB)
  SWC5 = 84.0*(SWA2+3.0*SWA*SWB+SWB2)
  SWC4 = -35.0*(SWA3+9.0*SWA2*SWB+9.0*SWA*SWB2+SWB3)
  SWC3 = 140.0*(SWA3*SWB+3.0*SWA2*SWB2+SWA*SWB3)
  SWC2 = -210.0*(SWA3*SWB2+SWA2*SWB3)
  SWC1 = 140.0*SWA3*SWB3
  SWC0 = (-35.0*SWA3*SWB2*SWB2+21.0*SWA2*SWB3*SWB2 -
          7.0*SWA*SWB3*SWB3+SWB3*SWB3*SWB)

  SW = (SWC7*R3*R3*R+SWC6*R3*R3+SWC5*R3*R2+SWC4*R2*R2+SWC3*R3+SWC2*R2 -
      SWC1*R+SWC0) / D7
  SW = jnp.where(R < low_tap_rad, 1.0,
                 jnp.where(R < up_tap_rad, SW, 0.0))

  return SW


def taper_inc(dist, low_tap_rad=0, up_tap_rad=10):
  '''
  Increasing tapering function
  0 at low_tap_rad and 1 at up_tap_rad
  smoothly taper the value in between
  '''
  return 1 - taper(dist, low_tap_rad, up_tap_rad)

