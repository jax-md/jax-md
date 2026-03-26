# Copyright 2024 Google LLC
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

"""ASE Calculator wrapper for UMA model in JAX-MD.

Provides an ASE-compatible calculator that uses the JAX UMA model
for energy and force predictions. This enables using UMA with ASE's
optimizers (BFGS, FIRE, etc.) and dynamics engines.

Example:
    >>> from ase.build import bulk
    >>> from ase.optimize import BFGS
    >>> from jax_md._nn.uma.ase_calculator import UMACalculator
    >>>
    >>> atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    >>> calc = UMACalculator(checkpoint_path='uma_checkpoint.pt')
    >>> atoms.calc = calc
    >>> opt = BFGS(atoms)
    >>> opt.run(fmax=0.01)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp

from jax_md._nn.uma.model import UMABackbone, UMAConfig, default_config
from jax_md._nn.uma.model_moe import UMAMoEBackbone, load_pretrained
from jax_md._nn.uma.heads import MLPEnergyHead, LinearEnergyHead
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices
from jax_md._nn.uma.pretrained import PRETRAINED_MODELS


class UMACalculator:
  """ASE-compatible calculator using the JAX UMA model.

  This calculator wraps the UMA backbone + energy head to provide
  energy and forces for ASE Atoms objects. It handles:
  - Periodic and non-periodic boundary conditions
  - Neighbor list construction from ASE
  - Charge and spin from atoms.info
  - Dataset selection

  Attributes:
      implemented_properties: List of properties this calculator can compute.
  """

  implemented_properties = ['energy', 'forces']

  def __init__(
    self,
    config: Optional[UMAConfig] = None,
    checkpoint_path: Optional[str] = None,
    params: Optional[dict] = None,
    head_type: str = 'mlp',
    task_name: str = 'omat',
    cutoff: Optional[float] = None,
  ):
    """Initialize the UMA ASE calculator.

    Args:
        config: UMA model configuration. If None, uses defaults or
            infers from checkpoint.
        checkpoint_path: Path to PyTorch checkpoint. If provided, loads
            pretrained weights.
        params: Pre-loaded JAX parameters. If provided, used directly
            (checkpoint_path is ignored).
        head_type: Energy head type ('mlp' or 'linear').
        task_name: Task/dataset name for dataset embedding
            ('omat', 'oc20', 'omol', etc.).
        cutoff: Distance cutoff. If None, uses config.cutoff.
    """
    self.task_name = task_name
    self.head_type = head_type
    self._is_moe = False

    # Load pretrained MoE model if checkpoint is a known pretrained name or .pt file
    if checkpoint_path is not None and params is None:
      is_pretrained = (
        checkpoint_path in PRETRAINED_MODELS
        or checkpoint_path.endswith('.pt')
      )
      if is_pretrained:
        moe_config, moe_params, pretrained_head_params = load_pretrained(checkpoint_path)
        self.config = moe_config
        self.backbone = UMAMoEBackbone(config=moe_config)
        self.params = moe_params
        self._pretrained_head_params = pretrained_head_params
        self._is_moe = True
      else:
        from jax_md._nn.uma.weight_conversion import (
          config_from_pytorch_checkpoint, load_pytorch_checkpoint,
        )
        config = config_from_pytorch_checkpoint(checkpoint_path)
        self.config = config
        self.backbone = UMABackbone(config=config)
        self.params = load_pytorch_checkpoint(checkpoint_path)
    else:
      if config is None:
        config = default_config()
      self.config = config
      self.backbone = UMABackbone(config=config)
      self.params = params  # May be None (init on first call)

    self.cutoff = cutoff or self.config.cutoff

    if head_type == 'mlp':
      self.head = MLPEnergyHead(
        sphere_channels=self.config.sphere_channels,
        hidden_channels=self.config.hidden_channels,
      )
    else:
      self.head = LinearEnergyHead(
        sphere_channels=self.config.sphere_channels,
      )

    if self.config.dataset_list:
      self._dataset_idx = dataset_names_to_indices(
        [task_name], self.config.dataset_list
      )
    else:
      self._dataset_idx = jnp.zeros(1, dtype=jnp.int32)

    # Cache
    self._results = {}
    self._atoms_cache = None

  @property
  def results(self):
    return self._results

  def calculate(self, atoms, properties=None, system_changes=None):
    """Compute energy and forces for an ASE Atoms object.

    Args:
        atoms: ASE Atoms object.
        properties: List of properties to compute (default: all).
        system_changes: List of changes since last calculation.
    """
    if properties is None:
      properties = self.implemented_properties

    # Extract data from ASE atoms
    positions = jnp.array(atoms.get_positions(), dtype=jnp.float32)
    atomic_numbers = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int32)
    num_atoms = len(atoms)

    # Build edges within cutoff
    edge_index, edge_distance_vec = self._build_edges(atoms)

    # System-level properties — use int for rand_emb, float for pos_emb
    if self.config.chg_spin_emb_type == 'rand_emb':
      charge = jnp.array([atoms.info.get('charge', 0)], dtype=jnp.int32)
      spin = jnp.array([atoms.info.get('spin', 0)], dtype=jnp.int32)
    else:
      charge = jnp.array([atoms.info.get('charge', 0.0)], dtype=jnp.float32)
      spin = jnp.array([atoms.info.get('spin', 0.0)], dtype=jnp.float32)
    batch = jnp.zeros(num_atoms, dtype=jnp.int32)
    dataset_idx = self._dataset_idx

    # Initialize params if needed
    if self.params is None:
      key = jax.random.PRNGKey(0)
      self.params = self._init_params(
        key, positions, atomic_numbers, batch,
        edge_index, edge_distance_vec, charge, spin, dataset_idx,
      )
    elif 'backbone' not in self.params:
      # Pretrained params — wrap with head
      if hasattr(self, '_pretrained_head_params') and self._pretrained_head_params:
        head_params = self._pretrained_head_params
      else:
        key = jax.random.PRNGKey(0)
        emb = self.backbone.apply(
          self.params, positions, atomic_numbers, batch,
          edge_index, edge_distance_vec, charge, spin, dataset_idx,
        )
        head_params = self.head.init(key, emb['node_embedding'], batch, 1)
      self.params = {'backbone': self.params, 'head': head_params}

    # Compute energy and forces
    energy, forces = self._compute_energy_and_forces(
      self.params, positions, atomic_numbers, batch,
      edge_index, edge_distance_vec, charge, spin, dataset_idx,
    )

    self._results = {
      'energy': float(energy),
      'forces': np.array(forces),
    }

  def get_potential_energy(self, atoms=None, force_consistent=False):
    if atoms is not None:
      self.calculate(atoms)
    return self._results.get('energy', None)

  def get_forces(self, atoms=None):
    if atoms is not None:
      self.calculate(atoms)
    return self._results.get('forces', None)

  def _build_edges(self, atoms):
    """Build edge list from ASE atoms within cutoff."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    # Use ASE's neighbor list for correct PBC handling
    try:
      from ase.neighborlist import neighbor_list
      # ASE convention: center_idx (i), neighbor_idx (j), offsets
      center_idx, neighbor_idx, offsets = neighbor_list(
        'ijS', atoms, cutoff=self.cutoff, self_interaction=False
      )
      # FairChem convention: edge_index = [source=neighbor, target=center]
      # edge_vec = pos[source_shifted] - pos[target]
      edge_distance_vec = (
        positions[neighbor_idx] + offsets @ cell - positions[center_idx]
      )
      idx_src = neighbor_idx  # source
      idx_tgt = center_idx    # target
    except ImportError:
      num_atoms = len(positions)
      idx_src, idx_tgt = [], []
      vecs = []
      for i in range(num_atoms):
        for j in range(num_atoms):
          if i != j:
            vec = positions[j] - positions[i]  # source=j, target=i
            dist = np.linalg.norm(vec)
            if dist < self.cutoff:
              idx_src.append(j)
              idx_tgt.append(i)
              vecs.append(vec)
      idx_src = np.array(idx_src, dtype=np.int32)
      idx_tgt = np.array(idx_tgt, dtype=np.int32)
      edge_distance_vec = np.array(vecs, dtype=np.float32)

    edge_index = jnp.array(np.stack([idx_src, idx_tgt]), dtype=jnp.int32)
    edge_distance_vec = jnp.array(edge_distance_vec, dtype=jnp.float32)
    return edge_index, edge_distance_vec

  def _init_params(self, key, positions, atomic_numbers, batch,
                   edge_index, edge_distance_vec, charge, spin, dataset_idx):
    """Initialize model parameters."""
    key1, key2 = jax.random.split(key)
    backbone_params = self.backbone.init(
      key1, positions, atomic_numbers, batch,
      edge_index, edge_distance_vec, charge, spin, dataset_idx,
    )
    emb = self.backbone.apply(
      backbone_params, positions, atomic_numbers, batch,
      edge_index, edge_distance_vec, charge, spin, dataset_idx,
    )
    head_params = self.head.init(
      key2, emb['node_embedding'], batch, 1,
    )
    return {'backbone': backbone_params, 'head': head_params}

  def _compute_energy_and_forces(
    self, params, positions, atomic_numbers, batch,
    edge_index, edge_distance_vec, charge, spin, dataset_idx,
  ):
    """Compute energy and forces via autodiff."""
    backbone_params = params['backbone']
    head_params = params['head']

    def energy_fn(pos):
      # Recompute edge vectors from positions (needed for gradient)
      edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
      emb = self.backbone.apply(
        backbone_params, pos, atomic_numbers, batch,
        edge_index, edge_vec, charge, spin, dataset_idx,
      )
      result = self.head.apply(
        head_params, emb['node_embedding'], batch, 1,
      )
      return result['energy'].sum()

    energy, grad = jax.value_and_grad(energy_fn)(positions)
    forces = -grad

    return energy, forces
