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
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence, Any, List, Tuple, Optional, Union
import numpy as onp
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition


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
  structure: Structure,
  d_frac: float = 0.05,
  nmin: int = 1,
  nmax: int = 48,
  restrict_to_compositions: Sequence[str] | None = None,
  max_coef: int | None = None,
  elements: Sequence[str] | None = None,
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
    stoichs = list(itertools.product(range(max_coef + 1), repeat=len(elements)))
    stoichs.pop(0)
    comps = []
    for stoich in stoichs:
      comp = dict(zip(elements, stoich))
      comps.append(Composition.from_dict(comp).reduced_formula)
    restrict_to_compositions = set(comps)

  # If a composition list is provided, ensure they are reduced formulas
  if restrict_to_compositions:
    restrict_to_compositions = [
      Composition(i).reduced_formula for i in restrict_to_compositions
    ]
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
            Composition(''.join(species[ids])).reduced_formula
            not in restrict_to_compositions
          ):
            continue
        candidates.append((ids, l, h))
  return candidates


def _valid_axis_pairs(bins: int) -> List[Tuple[int, int]]:
  """Return (l_index, h_index) pairs with h_1d[j] > l_1d[i] for each axis.

  Args:
    bins: number of bins on each axis

  Returns:
    List of valid (l_index, h_index) pairs
  """
  return [(i, j) for i in range(bins) for j in range(i, bins)]


def _get_subcells_from_axis_triple_range(
  args: Tuple[
    int,
    int,
    onp.ndarray,
    onp.ndarray,
    onp.ndarray,
    onp.ndarray,
    Sequence[Tuple[int, int]],
    int,
    int,
    frozenset[str] | None,
  ],
) -> List[Tuple[Sequence[int], onp.ndarray, onp.ndarray]]:
  """Scan axis-triple indices [start, end) and return matching subcells.

  Args:
    args: tuple of arguments to the function
      start: start index of the axis-triple range
      end: end index of the axis-triple range
      position: fractional coordinates of the atoms in the original structure
      species: species of the atoms in the original structure
      l_1d: lower 1D coordinates
      h_1d: upper 1D coordinates
      axis_pairs: list of valid (l_index, h_index) pairs
      nmin: minimum number of atoms allowed in a subcell

  Returns:
    List of tuples where the elements are indices of atoms corrresponding to
      the subcell, the lower and upper fractional coords forming the subcell.
  """
  # Unpack the arguments
  (
    start,
    end,
    position,
    species,
    l_1d,
    h_1d,
    axis_pairs,
    nmin,
    nmax,
    restrict_to_compositions,
  ) = args
  n_axis_pairs = len(axis_pairs)
  n_sq = n_axis_pairs * n_axis_pairs
  frac = position % 1
  candidates: List[Tuple[Sequence[int], onp.ndarray, onp.ndarray]] = []
  for idx in range(start, end):
    i0 = idx // n_sq
    rem = idx % n_sq
    i1 = rem // n_axis_pairs
    i2 = rem % n_axis_pairs
    (l0, h0), (l1, h1), (l2, h2) = (
      axis_pairs[i0],
      axis_pairs[i1],
      axis_pairs[i2],
    )
    l = onp.array([l_1d[l0], l_1d[l1], l_1d[l2]])
    h = onp.array([h_1d[h0], h_1d[h1], h_1d[h2]])
    mask = onp.logical_and(
      onp.all(h >= frac, axis=1), onp.all(l <= frac, axis=1)
    )
    ids = onp.argwhere(mask).flatten()
    if nmin <= len(ids) <= nmax:
      if restrict_to_compositions is not None:
        if (
          Composition(''.join(species[ids])).reduced_formula
          not in restrict_to_compositions
        ):
          continue
      candidates.append((ids, l, h))
  return candidates


def get_subcells_to_crystallize_parallel(
  structure: Structure,
  d_frac: float = 0.05,
  nmin: int = 1,
  nmax: int = 48,
  restrict_to_compositions: Sequence[str] | None = None,
  max_coef: int | None = None,
  elements: Sequence[str] | None = None,
  n_workers: int | None = None,
) -> List[Tuple[Sequence[int], onp.ndarray, onp.ndarray]]:
  """Get subcell structures to relax out of a large structure (e.g. amorphous)
  via parallelized search over valid subcell slices.

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
    n_workers: number of processes; defaults to ``os.cpu_count()``. Use 1 to
      run sequentially without spawning a process pool.

  Returns:
    List of tuples where the elements are indices of atoms corrresponding to
      the subcell, the lower and upper fractional coords forming the subcell.
  """
  position = structure.frac_coords
  species = onp.array([i.symbol for i in structure.species])

  if max_coef:
    stoichs = list(itertools.product(range(max_coef + 1), repeat=len(elements)))
    stoichs.pop(0)
    comps = []
    for stoich in stoichs:
      comp = dict(zip(elements, stoich))
      comps.append(Composition.from_dict(comp).reduced_formula)
    restrict_to_compositions = set(comps)

  if restrict_to_compositions:
    restrict_to_compositions = frozenset(
      Composition(i).reduced_formula for i in restrict_to_compositions
    )
  else:
    restrict_to_compositions = None

  bins = int(1 / d_frac)
  l_1d = onp.linspace(0, 1 - d_frac, bins)
  h_1d = onp.linspace(d_frac, 1, bins)
  axis_pairs = _valid_axis_pairs(bins)
  n_axis_pairs = len(axis_pairs)
  total = n_axis_pairs**3

  if n_workers is None:
    n_workers = os.cpu_count() or 1
  n_workers = max(1, n_workers)

  worker_args = (
    position,
    species,
    l_1d,
    h_1d,
    axis_pairs,
    nmin,
    nmax,
    restrict_to_compositions,
  )

  if n_workers == 1:
    return _get_subcells_from_axis_triple_range((0, total, *worker_args))

  chunk_size = math.ceil(total / n_workers)
  ranges = [
    (i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)
  ]
  tasks = [(start, end, *worker_args) for start, end in ranges]

  candidates: List[Tuple[Sequence[int], onp.ndarray, onp.ndarray]] = []
  with ProcessPoolExecutor(max_workers=n_workers) as executor:
    for chunk in executor.map(_get_subcells_from_axis_triple_range, tasks):
      candidates.extend(chunk)
  return candidates


def subcells_to_structures(
  candidates: List[Tuple[Sequence[int], onp.ndarray, onp.ndarray]],
  position: onp.ndarray,
  box: Union[BoxColumnMatrix, BoxRowMatrix],
  species: Sequence[str],
) -> List[Structure]:
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
    structures.append(
      Structure(
        new_box.T,  # back to row format expected by pymatgen
        onp.array(species)[ids].tolist(),
        coords=new_pos,
      )
    )
  return structures


def get_candidate_subset(
  candidates: Sequence[Any], n_workers: int = 1, worker_id: int = 0
):
  """Conveinence method to split candidates into subsets for parallel processing.

  Args:
    candidates: list of candidates as produced by get_structures_to_crystallize
    n_workers: number of workers to split the candidates into
    worker_id: id of the current worker (starting from 0)
  Returns:
    List of candidates corresponding to the current worker
  """
  return onp.array_split(onp.array(candidates, dtype=object), n_workers)[
    worker_id
  ]


def valid_subcell(
  structure: Structure,
  initial_energy: float,
  final_energy: float,
  e_tol: float = 0.001,
  fe_lower_limit: float = -5.0,
  fe_upper_limit: float = 0.0,
  fusion_distance: float = 1.5,
  distance_tolerance: float = 0.0001,
):
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
