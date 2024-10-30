# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for a2c crystallizer."""
import itertools
import os
from typing import Sequence, Any, List, Tuple, Optional, Union
import numpy as onp
import pymatgen as mg

class Box(onp.ndarray):
  """A 3x3 matrix to hold the lattice vectors."""
  def __new__(cls, input_array):
    obj = onp.asarray(input_array).view(cls)
    assert obj.shape == (3, 3)
    return obj


class BoxRowMatrix(Box):
  """Row-ordered box matrix (as in pyamtgen)."""
  @property
  def T(self):  # pylint: disable=invalid-name
    return BoxColumnMatrix(self.transpose())


class BoxColumnMatrix(Box):
  """Column-ordered box matrix (as in jax-md)."""
  @property
  def T(self):  # pylint: disable=invalid-name
    return BoxRowMatrix(self.transpose())


def get_subcells_to_crystallize(
    structure: mg.core.Structure,
    d_frac: float = 0.05,
    nmin: int = 1,
    nmax: int = 48,
    restrict_to_compositions: Optional[Sequence[str]] = None,
    max_coef: Optional[int] = None,
    elements: Optional[Sequence[str]] = None,
) -> List[Tuple[Sequence[int], onp.ndarray, onp.ndarray]]:
  """Get subcell structures to relax out of a large structure (e.g. amorphous).

  Args:
    structure: pymatgen Structure object to extract subcells from
    d_frac: fractional grid spacing to use for subcell slices
    nmin: minimum number of atoms allowed in a subcell
    nmax: maximum number of atoms allowed in a subcell
    restrict_to_compositions: limit search to subcells with these compositions
    max_coef: maximum coefficient in the formula. If supplied, we restrict
      formulas to stoich. of max_coef e.g. for 2, A, B, AB2, A2B.
    elements: list of elements to use in the subcell search. Required if
      max_coef is provided.

  Returns:
    List of tuples where the elements are indices of atoms corrresponding to
      the subcell, the lower and upper fractional coords forming the subcell.
  """
  position = structure.frac_coords
  species = onp.array([i.symbol for i in structure.species])

  # If max_coef is given in config, we will restrict formulas to
  # stoich. of max_coef e.g. for 2, A, B, AB2, A2B.
  if max_coef:
    stoichs = list(itertools.product(range(max_coef+1), repeat=len(elements)))
    stoichs.pop(0)
    comps = []
    for stoich in stoichs:
      comp = dict(zip(elements, stoich))
      comps.append(mg.Composition.from_dict(comp).reduced_formula)
    restrict_to_compositions = set(comps)

  # If a composition list is provided, ensure they are reduced formulas
  if restrict_to_compositions:
    restrict_to_compositions = [mg.Composition(i).reduced_formula
                                for i in restrict_to_compositions]
  else:
    restrict_to_compositions = None

  # Create orthorombic slices from the unit cube
  bins = int(1 / d_frac)
  llim = onp.array(
      list(itertools.product(*(3 * [onp.linspace(0, 1 - d_frac, bins)])))
  )
  hlim = onp.array(
      list(itertools.product(*(3 * [onp.linspace(d_frac, 1, bins)])))
  )
  candidates = []
  for l, h in itertools.product(llim, hlim):
    if onp.sum(h > l) == 3:
      mask = onp.logical_and(
          onp.all(h >= position % 1, axis=1), onp.all(l <= position % 1, axis=1)
      )
      ids = onp.argwhere(mask).flatten()  # indices of atoms in subcell
      if nmin <= len(ids) <= nmax:
        if restrict_to_compositions:
          if (
              mg.Composition(''.join(species[ids])).reduced_formula
              not in restrict_to_compositions
          ):
            continue
        candidates.append((ids, l, h))
  return candidates


def subcells_to_structures(
    candidates: List[Tuple[Sequence[int], onp.ndarray, onp.ndarray]],
    position: onp.ndarray,
    box: Union[BoxColumnMatrix, BoxRowMatrix],
    species: Sequence[str],
) -> List[mg.core.Structure]:
  """Create pymatgen Structure objects from subcell slices.

  Args:
    candidates: list of tuples (ids, ldot, hdot) where ids are indices of atoms
      in the subcell, ldot and hdot are the lower and upper fractional coords
      forming the subcell, as returned by get_structures_to_crystallize
    position: fractional coordinates of the atoms in the original structure
    box: lattice vectors of the original structure as a row or columnar matrix
    species: specie corresponding to each position in the original structure
  Returns:
    List of pymatgen Structure objects corresponding to the subcells in
    candidates
  """
  structures = []
  if isinstance(box, BoxRowMatrix):
    # If we are given a row-ordered box, we need to convert it to column-ordered
    # to be consistent with the rest of the code.
    box = box.T
  for subcell in candidates:
    ids, ldot, hdot = subcell
    pos = position[ids] % 1
    new_pos = (pos - ldot) / (hdot - ldot)
    new_box = box * (hdot - ldot)
    structures.append(mg.core.Structure(
        new_box.T,  # back to row format expected by pymatgen
        onp.array(species)[ids].tolist(),
        coords=new_pos,
    ))
  return structures


def get_candidate_subset(candidates: Sequence[Any],
                         n_workers: int = 1,
                         worker_id: int = 0):
  """Conveinence method to split candidates into subsets for parallel processing.

  Args:
    candidates: list of candidates as produced by get_structures_to_crystallize
    n_workers: number of workers to split the candidates into
    worker_id: id of the current worker (starting from 0)
  Returns:
    List of candidates corresponding to the current worker
  """
  return onp.array_split(onp.array(candidates, dtype=object),
                         n_workers)[worker_id]


def valid_subcell(structure: mg.core.Structure,
                  initial_energy: float,
                  final_energy: float,
                  e_tol: float = 0.001,
                  fe_lower_limit: float = -5.0,
                  fe_upper_limit: float = 0.0,
                  fusion_distance: float = 1.5,
                  distance_tolerance: float = 0.0001):
  """Validate the relaxed subcell."""
  # Unphysically negative formation energies indicate a problem
  if final_energy < fe_lower_limit:
    return False

  # If energy increases, the optimization may have been problematic
  if not (final_energy >= initial_energy + e_tol):
    return False

  # Skip structure if energy isn't low enough (e.g. negative wrt. elements),
  if not (final_energy <= fe_upper_limit + e_tol):
    return False

  # Skip structure if atoms are too close
  r = structure.distance_matrix
  r = onp.where(r < distance_tolerance, 10, r).min()
  if r < fusion_distance:
    logging.info('Bad structure! Fusion found.')
    return False
  return True

