"""
Contains interaction list related functions for ReaxFF

Author: Mehmet Cagri Kaymak
"""
from jax_md import space, partition, util
from jax_md.reaxff.reaxff_energy import (calculate_bo, 
                                        calculate_reaxff_energy)
from typing import Callable, Any, Tuple
import jax
import jax.numpy as jnp
from jax_md import dataclasses
from jax_md.util import safe_mask

Array = util.Array
MaskFn = Callable
CandidateFn = Callable
# Types
f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat

@dataclasses.dataclass
class ReaxFFNeighborListFns:
  allocate: Callable = dataclasses.static_field()
  update: Callable = dataclasses.static_field()

  def __iter__(self):
    return iter((self.allocate, self.update))

@dataclasses.dataclass
class Filtration:
  '''
  Allows filtering out the masked indices from the interaction lists
  for better performance.
  Supports dense -> dense and sparse -> sparse filtering.

  Attributes:
    candidate_fn: A function that creates candidate list to be filtered.
    mask_fn: A function that decides which indices to mask.
             returns a boolean array.
    is_dense: An boolean flag to show if it is a dense->dense filter.
    idx: Index array.
    did_buffer_overflow: A boolean specifying whether or not the cell list
      exceeded the maximum allocated capacity.
  '''
  candidate_fn: CandidateFn = dataclasses.static_field()
  mask_fn: MaskFn = dataclasses.static_field()
  is_dense: bool = dataclasses.static_field()
  idx: Array
  did_buffer_overflow: Array

  def count(self,
            candidate_args):
    '''
    Count the number of interactions in the filtered array
    '''

    candidate_inds, candidate_vals = self.candidate_fn(*candidate_args)
    mask = self.mask_fn(candidate_vals)
    if self.is_dense:
      size = jnp.max(jnp.sum(mask,axis=1), initial=0)
    else:
      size = jnp.sum(mask)
    return size

  def allocate_fixed(self,
               candidate_args,
               capacity=0) -> 'Filtration':
    '''
    Allocate a fixed size idx array
    '''
    candidate_inds, candidate_vals = self.candidate_fn(*candidate_args)
    mask = self.mask_fn(candidate_vals)
    if self.is_dense:
      size = jnp.max(jnp.sum(mask,axis=1))
      mapped_argwhere = jax.vmap(lambda vec:
                                 jnp.argwhere(vec,
                                              size=capacity,
                                              fill_value=-1).flatten())
      idx = mapped_argwhere(mask)
      did_buffer_overflow = (self.did_buffer_overflow
                             | (size > capacity))


    else:
      size = jnp.sum(mask)
      selected_inds = jnp.argwhere(mask, size=capacity,fill_value=-1).flatten()
      idx = candidate_inds[selected_inds]
      idx = jnp.where((selected_inds == -1).reshape(-1,1), -1, idx)
      did_buffer_overflow = (self.did_buffer_overflow
                             | (size > capacity))
    return Filtration(self.candidate_fn,
                      self.mask_fn,
                      self.is_dense,
                      idx.astype(jnp.int32),
                      did_buffer_overflow)

  def allocate(self,
               candidate_args,
               capacity_multiplier=1.25,
               min_capacity=0) -> 'Filtration':
    '''
    Initial allocation
    '''
    candidate_inds, candidate_vals = self.candidate_fn(*candidate_args)
    mask = self.mask_fn(candidate_vals)
    if self.is_dense:
      size = int(jnp.max(jnp.sum(mask,axis=1)) * capacity_multiplier)
      size = max(min_capacity, size)
      mapped_argwhere = jax.vmap(lambda vec:
                                 jnp.argwhere(vec,
                                              size=size,
                                              fill_value=-1).flatten())
      idx = mapped_argwhere(mask)


    else:
      size = int(jnp.sum(mask) * capacity_multiplier)
      size = max(min_capacity, size)
      selected_inds = jnp.argwhere(mask, size=size,fill_value=-1).flatten()
      idx = candidate_inds[selected_inds]
      idx = jnp.where((selected_inds == -1).reshape(-1,1), -1, idx)
      #idx = candidate_inds[selected_inds] * (selected_inds != -1).reshape(-1,1)
    return Filtration(self.candidate_fn,
                      self.mask_fn,
                      self.is_dense,
                      idx.astype(jnp.int32),
                      jnp.bool_(False))

  def update(self, candidate_args) -> 'Filtration':
    '''
    Updates the filtered index array and overflow flag
    '''
    if self.idx is None:
      raise ValueError('Have to allocate first.')

    candidate_inds, candidate_vals = self.candidate_fn(*candidate_args)
    mask = self.mask_fn(candidate_vals)
    if self.is_dense:
      size = jnp.max(jnp.sum(mask,axis=1))

      mapped_argwhere = jax.vmap(lambda vec:
                                 jnp.argwhere(vec,
                                              size= self.idx.shape[1],
                                              fill_value=-1).flatten())
      idx = mapped_argwhere(mask)
      did_buffer_overflow = (self.did_buffer_overflow
                             | (size > self.idx.shape[1]))
    else:
      selected_inds = jnp.argwhere(mask,
                                   size=len(self.idx),
                                   fill_value=-1).flatten()
      idx = candidate_inds[selected_inds]
      idx = jnp.where((selected_inds == -1).reshape(-1,1), -1, idx)
      did_buffer_overflow = (self.did_buffer_overflow
                             | (jnp.sum(mask) > len(self.idx)))
    return Filtration(self.candidate_fn,
                      self.mask_fn,
                      self.is_dense,
                      idx.astype(jnp.int32),
                      did_buffer_overflow)

def filtration(candidate_fn:CandidateFn,
               mask_fn:MaskFn,
               is_dense=False) -> Filtration:
  '''
  Returns an empty Filtration object to be used
  '''
  return Filtration(candidate_fn,mask_fn, is_dense, None, jnp.bool_(False))

def calculate_all_angles_and_distances(R, nbr_lists, map_metric, map_disp):
  N = len(R)
  atom_inds = jnp.arange(N, dtype=jnp.int32).reshape(-1,1)
  filtered_close_idx = nbr_lists.close_nbrs.idx[atom_inds,nbr_lists.filter2.idx]
  filtered_close_idx = jnp.where(nbr_lists.filter2.idx != -1,
                                 filtered_close_idx,
                                 N)
  close_nbr_dist = map_metric(R, R[filtered_close_idx])
  far_nbr_dist = map_metric(R, R[nbr_lists.far_nbrs.idx])

  close_nbr_disps = map_disp(R, R[filtered_close_idx])


  center = nbr_lists.filter3.idx[:, 0]
  neigh1_lcl = nbr_lists.filter3.idx[:,1]
  neigh2_lcl = nbr_lists.filter3.idx[:,2]

  body_3_cos_angles = jax.vmap(calculate_angle)(close_nbr_disps[center,
                                                            neigh1_lcl],
                                            close_nbr_disps[center,
                                                            neigh2_lcl])
  body_3_angles = safe_mask((body_3_cos_angles < 1) & (body_3_cos_angles > -1),
                            jnp.arccos, body_3_cos_angles).astype(far_nbr_dist.dtype)

  body_4_angles = calculate_all_4_body_angles(nbr_lists.filter4.idx,
                                              filtered_close_idx,
                                              close_nbr_disps)

  if nbr_lists.filter_hb != None:
    hb_inds = nbr_lists.filter_hb.idx
    far_nbr_disps = map_disp(R, R[nbr_lists.far_nbrs.idx])
    hb_ang_dist = calculate_all_hbond_angles_and_dists(hb_inds,
                                                       close_nbr_disps,
                                                       far_nbr_disps)
  else:
    # filler arrays when there is no hbond
    hb_inds = jnp.zeros(shape=(0,3),dtype=center.dtype)
    hb_ang_dist = jnp.zeros(shape=(2,0),dtype=R.dtype)

  return (close_nbr_dist,
          far_nbr_dist,
          body_3_angles,
          body_4_angles,
          hb_ang_dist)

@dataclasses.dataclass
class ReaxFFNeighborLists:
  '''
  Stores the neighbor lists and filters required for ReaxFF
  '''
  close_nbrs: NeighborList
  far_nbrs: NeighborList
  filter2: Filtration
  filter3: Filtration
  filter34: Filtration
  filter4: Filtration
  filter_hb_close: Filtration
  filter_hb_far: Filtration
  filter_hb: Filtration
  did_buffer_overflow: Array

  def __iter__(self):
    return iter((self.close_nbrs,
                 self.far_nbrs,
                 self.filter2,
                 self.filter3,
                 self.filter34,
                 self.filter4,
                 self.filter_hb_close,
                 self.filter_hb_far,
                 self.filter_hb,
                 self.did_buffer_overflow))


def calculate_angle(disp12, disp32):
  '''
  Assume there are 3 atoms: atom 1,2 and 3 where 2 is the center
  disp12 = pos1 - pos2
  disp32 = pos3 - pos2
  '''

  prev_dtype = disp12.dtype
  if prev_dtype == jnp.float64:
      EPS = 1E-10
  else:
      EPS = 1E-6
  d12_sq = jnp.sum(disp12 * disp12)
  d32_sq = jnp.sum(disp32 * disp32)
  d12 = safe_mask(d12_sq > 0, jnp.sqrt, d12_sq)
  d32 = safe_mask(d32_sq > 0, jnp.sqrt, d32_sq)
  norm1 = d12 + EPS
  norm2 = d32 + EPS
  dot_prod = jnp.dot(disp12, disp32) / (norm1 * norm2)
  #dot_prod = safe_mask((dot_prod < 1.0) & (dot_prod > -1.0), lambda x: x, dot_prod)
  dot_prod = jnp.clip(dot_prod, -1.0 + EPS, 1.0 - EPS)
  return dot_prod.astype(prev_dtype)

def calculate_all_4_body_angles(body_4_inds, nbr_inds, nbr_disps):
  '''
  Calculates the angle related terms for the provided 4-body interaction list
  '''

  # [1-(2-3)-4] ---- (2-3 is the center)
  center1_glb = body_4_inds[:,0]
  left_lcl = body_4_inds[:,1] # local to center1
  disp21 = nbr_disps[center1_glb,left_lcl]
  disp12 = -disp21
  center2_lcl = body_4_inds[:,2] # local to center1
  disp23 = nbr_disps[center1_glb,center2_lcl]
  disp32 = -disp23
  center2_glb = nbr_inds[center1_glb,center2_lcl]
  right_lcl = body_4_inds[:,3] # local to center2
  disp34 = nbr_disps[center2_glb,right_lcl]
  disp43 = -disp34
  disp14 = disp12 + disp23 + disp34
  cos_angle_123 = jax.vmap(calculate_angle)(disp12,disp32)
  cos_angle_234 = jax.vmap(calculate_angle)(disp23,disp43)
  coshd = cos_angle_123
  coshe = cos_angle_234
  sinhd_sq = 1 - coshd**2
  sinhe_sq = 1 - coshe**2
  sinhd = safe_mask(sinhd_sq > 0, jnp.sqrt, sinhd_sq)
  sinhe = safe_mask(sinhe_sq > 0, jnp.sqrt, sinhe_sq)

  d142 = jnp.sum(disp14**2,axis=1)

  rla_sq = jnp.sum(disp12**2,axis=1)
  rlb_sq = jnp.sum(disp23**2,axis=1)
  rlc_sq = jnp.sum(disp34**2,axis=1)

  rla = safe_mask(rla_sq > 0, jnp.sqrt, rla_sq)
  rlb = safe_mask(rlb_sq > 0, jnp.sqrt, rlb_sq)
  rlc = safe_mask(rlc_sq > 0, jnp.sqrt, rlc_sq)

  tel = (rla*rla+rlb*rlb+rlc*rlc-d142-2.0*(rla*rlb*coshd-rla*rlc*
      coshd*coshe+rlb*rlc*coshe))
  poem = 2.0*rla*rlc*sinhd*sinhe
  #poem2=poem*poem
  poem = poem + 1e-20
  #poem = jnp.where(poem < 1e-20, 1e-20, poem)

  arg = tel/poem

  arg = jnp.clip(arg, -1.0, 1.0)

  return jnp.array([sinhd*sinhe,arg])


def calculate_all_hbond_angles_and_dists(hbond_inds,
                                         close_nbr_disps,
                                         far_nbr_disps):
  '''
  Calculates the angles and distances for the provided hydrogen-bond list
  '''
  disp12 = close_nbr_disps[hbond_inds[:,0],hbond_inds[:,1]]
  disp13 = far_nbr_disps[hbond_inds[:,0],hbond_inds[:,2]]
  cos_angles = jax.vmap(calculate_angle)(disp12,disp13)
  angles = safe_mask((cos_angles < 1) & (cos_angles > -1),
                            jnp.arccos, cos_angles).astype(disp12.dtype)
  dist13_sq = jnp.sum(disp13**2,axis=1)
  dists13 = safe_mask(dist13_sq > 1e-10, jnp.sqrt, dist13_sq)
  return jnp.array([angles, dists13])

def find_3_body_inter(N, ctr_ind, nbr_pot, nbr_inds, lcl_inds, cutoff2):
  '''
  Finds all potential 3-body interactions based on bonded potentials
  '''
  # filler values in filtered_lcl_inds is "-1", defined in the filter
  nbrs = nbr_inds
  nbr_pot_cond = nbr_pot > cutoff2
  vals,inds = jax.vmap(lambda n1,i1:
                 jax.vmap(lambda n2,i2: (( nbr_pot[n1] * nbr_pot[n2]
                                         * ((nbr_pot_cond[n1])
                                         & (nbr_pot_cond[n2])
                                         & (n2>n1)
                                         & (i1 != N) & (i2 != N))),
                          jnp.array([ctr_ind,
                                     n1,
                                     n2], dtype=jnp.int32)))(lcl_inds,nbrs))(lcl_inds,nbrs)

  return inds, vals

def find_body_4_inter(body_3_item, neigh_inds,neigh_bo, cutoff2):
  '''
  Finds all potential 4 body interactions involving a 3-body item
  '''
  N = neigh_inds.shape[0]
  ind2,n21,n22 = body_3_item
  bo1 = neigh_bo[ind2][n21]
  bo2 = neigh_bo[ind2][n22]
  ind1 = neigh_inds[ind2][n21]
  ind3 = neigh_inds[ind2][n22]
  bo_mult = bo1 * bo2
  # 3-body inter. list (i,j,k) exists if k>j
  sub_inds = jnp.arange(neigh_inds.shape[1], dtype=jnp.int32)
  sub_pots = neigh_bo[ind3]
  vals1,inds1 = jax.vmap(lambda x, n31:
                  ((x * bo_mult
                   * ((x > cutoff2)
                   & (bo1 > cutoff2)
                   & (bo2 > cutoff2)
                   & (ind1 != neigh_inds[ind2][n22]) # left != center2
                   & (neigh_inds[ind3][n31] != ind2) # right != center1
                   & (ind3 > ind2)
                   & (neigh_inds[ind2][n21] != neigh_inds[ind3][n31])
                   & (neigh_inds[ind3][n31] != N) # dummy item in filter2
                   & (ind2 != -1))), #dummy item in filter3
                   jnp.array([ind2,
                              n21,
                              n22,
                              n31], dtype=jnp.int32)))(sub_pots,sub_inds)

  ind3,ind1 = ind1,ind3
  n22,n21 = n21,n22
  sub_pots = neigh_bo[ind3]
  vals2,inds2 = jax.vmap(lambda x, n31:
                  ((x * bo_mult
                   * ((x > cutoff2)
                   & (bo1 > cutoff2)
                   & (bo2 > cutoff2)
                   & (ind1 != neigh_inds[ind2][n22]) # left != center2
                   & (neigh_inds[ind3][n31] != ind2) # right != center1
                   & (ind3 > ind2)
                   & (neigh_inds[ind2][n21] != neigh_inds[ind3][n31])
                   & (neigh_inds[ind3][n31] != N) # dummy item in filter2
                   & (ind2 != -1))), #dummy item in filter3
                   jnp.array([ind2,
                              n21,
                              n22,
                              n31], dtype=jnp.int32)))(sub_pots,sub_inds)

  # left : neigh_inds[ind2][n21]    or neigh_inds[center1][body_4_inds[:,1]]
  # center1: ind2                   or body_4_inds[:,0]
  # center2: neigh_inds[ind2][n22]  or neigh_inds[center1][body_4_inds[:,2]]
  # right: neigh_inds[center2][n31] or neigh_inds[center2][body_4_inds[:,3]]
  return jnp.vstack((inds1,inds2)), jnp.vstack((vals1,vals2))

def body_3_candidate_fn(nbr_inds,
                        neigh_bo,
                        species,
                        cutoff2,
                        param_mask):
  '''
  Creates full candidate index and value arrays for 3 body interactions,
  to be used by a filter.
  '''
  N = len(nbr_inds)
  atom_inds = jnp.arange(N, dtype=jnp.int32)
  find_all_3_body_inter = jax.vmap(find_3_body_inter,in_axes=(None,
                                                              0,0,0,
                                                              None,None))
  local_inds = jnp.arange(nbr_inds.shape[1], dtype=jnp.int32)
  inds,body_3_vals = find_all_3_body_inter(N,atom_inds,
                                           neigh_bo,
                                           nbr_inds,
                                           local_inds,
                                           cutoff2)
  inds = inds.reshape(-1,3)
  center = inds[:, 0]
  neigh1_lcl = inds[:,1]
  neigh2_lcl = inds[:,2]
  neigh1_glb = nbr_inds[center,neigh1_lcl]
  neigh2_glb = nbr_inds[center,neigh2_lcl]

  cent_types = species[center]
  neigh1_types = species[neigh1_glb]
  neigh2_types = species[neigh2_glb]
  my_param_mask = param_mask[neigh1_types,cent_types,neigh2_types].flatten()
  #print(body_3_vals.flatten())
  return inds, body_3_vals.flatten() * my_param_mask

def body_34_candidate_fn(body_3_inds,
                        nbr_inds,
                        neigh_bo,
                        species,
                        cutoff2,
                        param_mask):
  inds = body_3_inds.reshape(-1,3)
  center = inds[:, 0]
  neigh1_lcl = inds[:,1]
  neigh2_lcl = inds[:,2]
  neigh1_glb = nbr_inds[center,neigh1_lcl]
  neigh2_glb = nbr_inds[center,neigh2_lcl]

  cent_types = species[center]
  neigh1_types = species[neigh1_glb]
  neigh2_types = species[neigh2_glb]
  my_param_mask = param_mask[neigh1_types,cent_types,neigh2_types].flatten()
  pot_mult = (neigh_bo[center,neigh1_lcl] * nbr_inds[center,neigh2_lcl]) > cutoff2

  return body_3_inds, my_param_mask & pot_mult.flatten()

def body_4_candidate_fn(body_3_inds,
                        nbr_inds,
                        neigh_bo,
                        species,
                        cutoff2,
                        param_mask):
  '''
  Creates full candidate index and value arrays for 4 body interactions,
  to be used by a filter.
  '''
  find_all_4_body_inter = jax.vmap(find_body_4_inter,in_axes=(0,
                                                              None,None,None))
  inds, body_4_vals = find_all_4_body_inter(body_3_inds,
                                            nbr_inds,
                                            neigh_bo,
                                            cutoff2)
  inds = inds.reshape(-1,4)
  center1_glb = inds[:,0]
  left_lcl = inds[:,1] # local to center1
  left_glb = nbr_inds[center1_glb,left_lcl]
  center2_lcl = inds[:,2] # local to center1
  center2_glb = nbr_inds[center1_glb,center2_lcl]
  roght_lcl = inds[:,3] # local to center2
  right_glb = nbr_inds[center2_glb,roght_lcl]
  my_param_mask = param_mask[species[left_glb],species[center1_glb],
                     species[center2_glb],species[right_glb]].flatten()
  return inds,body_4_vals.flatten() * my_param_mask

def hbond_candidate_fn(donor_inds,
                       close_nbr_inds,
                       hb_short_inds,
                       far_nbr_inds,
                       hb_long_inds,
                       species,
                       param_mask):
  '''
  Creates full candidate index and value arrays for hydrogen bonds,
  to be used by a filter.
  '''
  # donor is h (1)
  # acceptors are other types (2)
  # part1 searches for long bond
  # part2 searches for short bond
  lcl_inds = jnp.arange(len(donor_inds), dtype=jnp.int32)
  inds = jax.vmap(lambda i:
            jax.vmap(lambda s_i:
               jax.vmap(lambda l_i:
                  jnp.array((donor_inds[i],
                             s_i,
                             l_i)))(hb_long_inds[i]))(hb_short_inds[i]))(lcl_inds)
  inds = inds.reshape(-1,3).astype(jnp.int32)

  glb_center = inds[:,0]
  lcl_close_nbr = inds[:,1]
  glb_close_nbr = close_nbr_inds[glb_center,lcl_close_nbr]
  lcl_far_nbr = inds[:,2]
  glb_far_nbr = far_nbr_inds[glb_center,lcl_far_nbr]

  cent_types = species[glb_center]
  close_nbr_types = species[glb_close_nbr]
  far_nbr_types = species[glb_far_nbr]
  # param order: local, center, far
  my_param_mask = param_mask[close_nbr_types,cent_types,far_nbr_types]
  full_mask = (my_param_mask
               * (lcl_close_nbr != -1)
               * (lcl_far_nbr != -1)
               * (glb_close_nbr != glb_far_nbr)).flatten()

  return inds.reshape(-1,3), full_mask

def reaxff_inter_list(displacement: DisplacementFn,
                    box: Box,
                    species: Array,
                    species_AN: Array,
                    force_field,
                    total_charge: float = 0.0,
                    tol: float = 1e-6,
                    max_solver_iter: int = 500,
                    backprop_solve: bool = False,
                    tors_2013: bool = False,
                    solver_model: str = "EEM",
                    short_inters_capacity_multiplier: int = 1.2,
                    long_inters_capacity_multiplier: int = 1.2
                    ) -> Tuple[ReaxFFNeighborListFns,
                                                Callable]:
  '''
  Contains all the neccesary logic to run a reaxff simulation and
  allocate, reallocate, update and energy_fn functions
  Args:
    species: An ndarray of shape `[n, ]` for the atom types.
    species_AN:  An ndarray of shape `[n, ]` for the atomic numbers
      of the atoms.
    total_charge: Total charge of the system (float)
    tol: Tolarence for the charge solver
    max_solver_iter: Maximum number of solver iterations
    backprop_solve: Control variable to decide whether to do a solve to
      calculate the gradients of the charges wrt positions. By definition,
      the gradients should be 0 but if the solver tolerance is high,
      the gradients might be non-ignorable.
    tors_2013: Control variable to decide whether to use more stable
      version of the torsion interactions
    solver_model: Control variable for the solver model
      ("EEM" or "ACKS")
    short_inters_capacity_multiplier: capacity multiplier for all short range
      interactions
    long_inters_capacity_multiplier: capacity multiplier for all long range
      interactions
  Returns:
    System energy in kcal/mol
  '''

  neighbor_fn1 = partition.neighbor_list(displacement,
                                  box=box,
                                  r_cutoff=5.0,
                                  dr_threshold=0.5,
                                  capacity_multiplier=long_inters_capacity_multiplier,
                                  format=partition.Dense)

  neighbor_fn2 = partition.neighbor_list(displacement,
                                  box=box,
                                  r_cutoff=10.0,
                                  dr_threshold=0.5,
                                  capacity_multiplier=short_inters_capacity_multiplier,
                                  format=partition.Dense)
  cutoff2 = force_field.cutoff2
  cutoff = force_field.cutoff
  FF_types_hb = force_field.nphb
  types_hb = FF_types_hb[species]
  hb_donor_mask = jnp.array(types_hb == 1)
  hb_acceptor_mask = jnp.array(types_hb == 2)
  hb_donor_inds = jnp.argwhere(hb_donor_mask).flatten()

  hbond_flag = (jnp.sum(hb_donor_mask) > 0) and (jnp.sum(hb_acceptor_mask) > 0)

  filter2_fn = filtration(lambda inds,vals: (inds,vals),
                          lambda x: x > 0.0,
                          is_dense=True)

  filter3_fn = filtration(body_3_candidate_fn, lambda x: x > 0.00001)

  filter34_fn = filtration(body_34_candidate_fn, lambda x: x)

  filter4_fn = filtration(body_4_candidate_fn, lambda x: x > cutoff2)

  filter_hb_close_fn = filtration(lambda inds, vals, acceptor_mask:
                                  (inds, vals * acceptor_mask[inds]),
                                  lambda x: x > 0.01,
                                  is_dense=True)

  filter_hb_far_fn = filtration(lambda inds, vals, acceptor_mask:
                                  (inds, vals * acceptor_mask[inds]),
                                  lambda x: (x < 7.5) & (x > 0.0),
                                  is_dense=True)

  filter_hb_fn = filtration(hbond_candidate_fn,
                                  lambda x: x > 0.0,
                                  is_dense=False)


  metric = space.metric(displacement)
  map_metric = space.map_neighbor(metric)
  map_disp = space.map_neighbor(displacement)

  new_body3_mask = force_field.body34_params_mask | force_field.body3_params_mask

  def update(R, reax_nbrs):

    [close_nbrs,
    far_nbrs,
    filter2,
    filter3,
    filter34,
    filter4,
    filter_hb_close,
    filter_hb_far,
    filter_hb,
    overflow_flag] = list(reax_nbrs)
    N = len(R)
    close_nbrs = close_nbrs.update(R)
    far_nbrs = far_nbrs.update(R)
    close_nbr_dist = map_metric(R, R[close_nbrs.idx])
    far_nbr_dist = map_metric(R, R[far_nbrs.idx])

    bo = calculate_bo(close_nbrs.idx,
                    close_nbr_dist,
                    species,
                    species_AN,
                    force_field)
    bo = bo.astype(close_nbr_dist.dtype)

    filter2 = filter2.update(candidate_args=(close_nbrs.idx,bo))
    atom_inds = jnp.arange(N, dtype=jnp.int32).reshape(-1,1)

    filtered_close_idx = close_nbrs.idx[atom_inds,filter2.idx]
    filtered_close_idx = jnp.where(filter2.idx != -1,
                                   filtered_close_idx,
                                   N)
    neigh_bo2 = bo[atom_inds,filter2.idx]
    neigh_bo2 = jnp.where(filter2.idx != -1,
                                   neigh_bo2,
                                   0.0)

    filter3 = filter3.update(candidate_args=(filtered_close_idx,
                                             neigh_bo2,
                                             #sub_inds,
                                             species,
                                             cutoff2,
                                             new_body3_mask))

    filter34 = filter34.update(candidate_args=(filter3.idx,
                                             filtered_close_idx,
                                             neigh_bo2,
                                             #sub_inds,
                                             species,
                                             cutoff2,
                                             force_field.body34_params_mask))

    filter4 = filter4.update(candidate_args=(filter34.idx,
                                             filtered_close_idx,
                                             neigh_bo2,
                                             #sub_inds,
                                             species,
                                             cutoff2,
                                             force_field.body4_params_mask))
    if hbond_flag:
      dnr_nbr_inds = filtered_close_idx[hb_donor_inds]
      dnr_nbr_pots = neigh_bo2[hb_donor_inds]
      dnr_long_nbr_inds = far_nbrs.idx[hb_donor_inds]
      dnr_long_nbr_dists = far_nbr_dist[hb_donor_inds]
      filter_hb_close = filter_hb_close.update(candidate_args=(dnr_nbr_inds,
                                                               dnr_nbr_pots,
                                                               hb_acceptor_mask))
      filter_hb_far = filter_hb_far.update(candidate_args=(dnr_long_nbr_inds,
                                                           dnr_long_nbr_dists,
                                                           hb_acceptor_mask))
      filter_hb = filter_hb.update(candidate_args=(hb_donor_inds,
                                                  filtered_close_idx,
                                                  filter_hb_close.idx,
                                                  far_nbrs.idx,
                                                  filter_hb_far.idx,
                                                  species,
                                                  force_field.hb_params_mask))
      filter_hb_did_buffer_overflow = filter_hb.did_buffer_overflow
    else:
      filter_hb_did_buffer_overflow = False
    overflow_flag = (overflow_flag | close_nbrs.did_buffer_overflow
                     | far_nbrs.did_buffer_overflow
                     | filter2.did_buffer_overflow
                     | filter3.did_buffer_overflow
                     | filter34.did_buffer_overflow
                     | filter4.did_buffer_overflow
                     | filter_hb_did_buffer_overflow)

    return ReaxFFNeighborLists(close_nbrs,
                                far_nbrs,
                                filter2,
                                filter3,
                                filter34,
                                filter4,
                                filter_hb_close,
                                filter_hb_far,
                                filter_hb,
                                overflow_flag)

  def allocate(R):
    N = len(R)
    close_nbrs = neighbor_fn1.allocate(R)
    far_nbrs = neighbor_fn2.allocate(R)
    close_nbr_dist = map_metric(R, R[close_nbrs.idx])
    far_nbr_dist = map_metric(R, R[far_nbrs.idx])

    bo = calculate_bo(close_nbrs.idx,
                    close_nbr_dist,
                    species,
                    species_AN,
                    force_field)

    filter2 = filter2_fn.allocate(candidate_args=(close_nbrs.idx,bo),
                                  capacity_multiplier=short_inters_capacity_multiplier)
    atom_inds = jnp.arange(N, dtype=jnp.int32).reshape(-1,1)

    filtered_close_idx = close_nbrs.idx[atom_inds,filter2.idx]
    filtered_close_idx = jnp.where(filter2.idx != -1,
                                   filtered_close_idx,
                                   N)
    neigh_bo2 = bo[atom_inds,filter2.idx]
    neigh_bo2 = jnp.where(filter2.idx != -1,
                                   neigh_bo2,
                                   0.0)

    #TODO: What if a 4 body interaction exists without having corresponding
    # 3 body interactions? (Not likely but possible)
    filter3 = filter3_fn.allocate(candidate_args=(filtered_close_idx,
                                             neigh_bo2,
                                             #sub_inds,
                                             species,
                                             force_field.cutoff2,
                                             new_body3_mask),
                                  capacity_multiplier=short_inters_capacity_multiplier)

    filter34 = filter34_fn.allocate(candidate_args=(filter3.idx,
                                             filtered_close_idx,
                                             neigh_bo2,
                                             species,
                                             force_field.cutoff2,
                                             force_field.body34_params_mask),
                                  capacity_multiplier=short_inters_capacity_multiplier)

    filter4 = filter4_fn.allocate(candidate_args=(filter34.idx,
                                             filtered_close_idx,
                                             neigh_bo2,
                                             #sub_inds,
                                             species,
                                             force_field.cutoff2,
                                             force_field.body4_params_mask),
                                  capacity_multiplier=short_inters_capacity_multiplier)
    if hbond_flag:
      dnr_nbr_inds = filtered_close_idx[hb_donor_inds]
      dnr_nbr_pots = neigh_bo2[hb_donor_inds]
      dnr_long_nbr_inds = far_nbrs.idx[hb_donor_inds]
      dnr_long_nbr_dists = far_nbr_dist[hb_donor_inds]
      filter_hb_close = filter_hb_close_fn.allocate(
                                            candidate_args=(dnr_nbr_inds,
                                                            dnr_nbr_pots,
                                                            hb_acceptor_mask),
                                            capacity_multiplier=short_inters_capacity_multiplier)
      filter_hb_far = filter_hb_far_fn.allocate(
                                            candidate_args=(dnr_long_nbr_inds,
                                                            dnr_long_nbr_dists,
                                                            hb_acceptor_mask),
                                            capacity_multiplier=short_inters_capacity_multiplier)
      filter_hb = filter_hb_fn.allocate(candidate_args=(hb_donor_inds,
                                               filtered_close_idx,
                                               filter_hb_close.idx,
                                               far_nbrs.idx,
                                               filter_hb_far.idx,
                                               species,
                                               force_field.hb_params_mask),
                                        capacity_multiplier=short_inters_capacity_multiplier)
    else:
      filter_hb_close = None
      filter_hb_far = None
      filter_hb = None


    return ReaxFFNeighborLists(close_nbrs,
                                far_nbrs,
                                filter2,
                                filter3,
                                filter34,
                                filter4,
                                filter_hb_close,
                                filter_hb_far,
                                filter_hb,
                                jnp.bool_(False))

  def energy_fn(R, nbr_lists, init_charges=None):
    (close_nbr_dist,
    far_nbr_dist,
    body_3_angles,
    body_4_angles,
    hb_ang_dist) = calculate_all_angles_and_distances(R,
                                                      nbr_lists,
                                                      map_metric,
                                                      map_disp)

    energy, charges = calculate_reaxff_energy(species,
                                            species_AN,
                                            nbr_lists,
                                            close_nbr_dist,
                                            far_nbr_dist,
                                            body_3_angles,
                                            body_4_angles,
                                            hb_ang_dist,
                                            force_field,
                                            init_charges,
                                            total_charge,
                                            tol,
                                            max_solver_iter,
                                            backprop_solve,
                                            tors_2013,
                                            solver_model=solver_model,
                                            )


    return energy

  return ReaxFFNeighborListFns(allocate, update), energy_fn






