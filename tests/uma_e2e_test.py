"""
End-to-end parity test: FairChem PyTorch vs JAX UMA on 20 diverse structures.

Runs both the original PyTorch eSCNMDMoeBackbone and the JAX UMAMoEBackbone
on the same structures and compares backbone embeddings and energies.
No pre-generated .npz files needed -- PT reference is computed on-the-fly."""

import sys

import numpy as np
from absl.testing import absltest
from ase import Atoms
from ase.build import bulk, molecule

import jax.numpy as jnp

from jax_md import test_util


def make_structures():
  """Build diverse structures for comparison testing."""
  structures = {}
  rng = np.random.default_rng(2024)

  for el, a in [
    ('Al', 4.05),
    ('Cu', 3.615),
    ('Ag', 4.085),
    ('Au', 4.078),
    ('Ni', 3.524),
    ('Pt', 3.924),
    ('Pd', 3.890),
    ('Rh', 3.803),
    ('Ir', 3.839),
    ('Pb', 4.951),
  ]:
    atoms = bulk(el, 'fcc', a=a, cubic=True)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{el}_fcc'] = (atoms, 'omat')

  for el, a in [
    ('Fe', 2.87),
    ('W', 3.165),
    ('Mo', 3.147),
    ('Cr', 2.91),
    ('V', 3.03),
    ('Nb', 3.30),
    ('Ta', 3.30),
  ]:
    atoms = bulk(el, 'bcc', a=a, cubic=True)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{el}_bcc'] = (atoms, 'omat')

  for el, a in [('Si', 5.43), ('Ge', 5.66), ('C', 3.567)]:
    atoms = bulk(el, 'diamond', a=a)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{el}_diamond'] = (atoms, 'omat')

  def rocksalt(el1, el2, a):
    return Atoms(
      f'{el1}{el2}' * 4,
      scaled_positions=[
        (0, 0, 0),
        (0.5, 0.5, 0.5),
        (0.5, 0, 0),
        (0, 0.5, 0.5),
        (0, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0, 0.5),
        (0.5, 0.5, 0),
      ],
      cell=[a, a, a],
      pbc=True,
    )

  for el1, el2, a in [
    ('Na', 'Cl', 5.64),
    ('K', 'Cl', 6.29),
    ('Li', 'F', 4.03),
    ('Mg', 'O', 4.21),
    ('Ca', 'O', 4.81),
    ('Ba', 'O', 5.52),
    ('Na', 'F', 4.62),
    ('K', 'Br', 6.60),
  ]:
    atoms = rocksalt(el1, el2, a)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{el1}{el2}_rs'] = (atoms, 'omat')

  def perovskite(A, B, a):
    return Atoms(
      f'{A}{B}OOO',
      scaled_positions=[
        (0, 0, 0),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),
      ],
      cell=[a, a, a],
      pbc=True,
    )

  for A, B, a in [('Sr', 'Ti', 3.905), ('Ba', 'Ti', 4.01), ('Ca', 'Ti', 3.84)]:
    atoms = perovskite(A, B, a)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{A}{B}O3'] = (atoms, 'omat')

  for el, a in [('Cu', 3.615), ('Fe', 2.87), ('Al', 4.05)]:
    cryst = 'fcc' if el in ('Cu', 'Al') else 'bcc'
    atoms = bulk(el, cryst, a=a, cubic=True).repeat((2, 2, 2))
    atoms.positions += rng.normal(scale=0.015, size=atoms.positions.shape)
    structures[f'{el}_2x2x2'] = (atoms, 'omat')

  for mol in [
    'H2O',
    'NH3',
    'CH4',
    'CO2',
    'H2',
    'N2',
    'O2',
    'C2H6',
    'C2H4',
    'C2H2',
    'CH3OH',
    'H2CO',
    'HCN',
    'HF',
    'H2S',
    'PH3',
    'SiH4',
    'NF3',
    'CF4',
    'SF6',
  ]:
    try:
      atoms = molecule(mol)
      atoms.center(vacuum=10.0)
      atoms.pbc = False
      atoms.info['charge'] = 0
      atoms.info['spin'] = 1
      atoms.positions += rng.normal(scale=0.03, size=atoms.positions.shape)
      structures[f'{mol}_mol'] = (atoms, 'omol')
    except Exception:
      pass

  for el, a0 in [('Cu', 3.615), ('Al', 4.05), ('Pt', 3.924)]:
    for strain in [-0.03, 0.03]:
      a = a0 * (1 + strain)
      tag = 'comp' if strain < 0 else 'tens'
      atoms = bulk(el, 'fcc', a=a, cubic=True)
      structures[f'{el}_{tag}'] = (atoms, 'omat')

  o2 = molecule('O2')
  o2.center(vacuum=10.0)
  o2.pbc = False
  o2.info['charge'] = 0
  o2.info['spin'] = 3
  structures['O2_triplet'] = (o2.copy(), 'omol')
  o2.info['spin'] = 1
  structures['O2_singlet'] = (o2.copy(), 'omol')

  oh = Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.97]])
  oh.center(vacuum=10.0)
  oh.pbc = False
  oh.info['charge'] = -1
  oh.info['spin'] = 1
  structures['OH_anion'] = (oh, 'omol')

  return structures


MODEL_NAME = 'uma-s-1p2'

E2E_STRUCTURES = [
  'Cu_fcc',
  'Au_fcc',
  'Fe_bcc',
  'W_bcc',
  'Si_diamond',
  'C_diamond',
  'NaCl_rs',
  'MgO_rs',
  'SrTiO3',
  'BaTiO3',
  'Cu_2x2x2',
  'H2O_mol',
  'CH4_mol',
  'CO2_mol',
  'C2H6_mol',
  'SF6_mol',
  'Cu_tens',
  'O2_triplet',
  'OH_anion',
  'NH3_mol',
]


def _build_edges(atoms, cutoff):
  """Build edge_index and edge_vec from ASE atoms."""
  positions = atoms.get_positions().astype(np.float32)
  n = len(atoms)

  if any(atoms.pbc):
    from ase.neighborlist import neighbor_list as ase_nl

    center_idx, neighbor_idx, offsets = ase_nl(
      'ijS', atoms, cutoff=cutoff, self_interaction=False
    )
    cell = np.array(atoms.get_cell())
    idx_src, idx_tgt = neighbor_idx, center_idx
    cell_offsets = offsets.astype(np.float32)
    edge_vec = (
      positions[idx_src]
      + (cell_offsets @ cell).astype(np.float32)
      - positions[idx_tgt]
    ).astype(np.float32)
  else:
    idx_src, idx_tgt = [], []
    for i in range(n):
      for j in range(n):
        if i != j and np.linalg.norm(positions[i] - positions[j]) < cutoff:
          idx_src.append(j)
          idx_tgt.append(i)
    idx_src = np.array(idx_src, dtype=np.int64)
    idx_tgt = np.array(idx_tgt, dtype=np.int64)
    cell_offsets = None
    edge_vec = (positions[idx_src] - positions[idx_tgt]).astype(np.float32)

  edge_index = np.stack([idx_src, idx_tgt])
  return edge_index, edge_vec, cell_offsets


def _atoms_to_pt_data(atoms, task, cutoff, ds_list):
  """Convert ASE atoms to PT model input dict."""
  import torch

  positions = atoms.get_positions().astype(np.float32)
  Z = atoms.get_atomic_numbers().astype(np.int64)
  n = len(atoms)
  edge_index, _, cell_offsets = _build_edges(atoms, cutoff)
  n_e = edge_index.shape[1]
  if cell_offsets is None:
    cell_offsets = np.zeros((n_e, 3), dtype=np.float32)

  class DB(dict):
    def __getattr__(self, k):
      try:
        return self[k]
      except KeyError:
        raise AttributeError(k)

    def __setattr__(self, k, v):
      self[k] = v

    def get(self, k, *a, **kw):
      return super().get(k, kw.get('default', a[0] if a else None))

  return DB(
    pos=torch.tensor(positions),
    atomic_numbers=torch.tensor(Z),
    atomic_numbers_full=torch.tensor(Z),
    batch=torch.zeros(n, dtype=torch.long),
    batch_full=torch.zeros(n, dtype=torch.long),
    edge_index=torch.tensor(edge_index.astype(np.int64)),
    charge=torch.tensor([int(atoms.info.get('charge', 0))], dtype=torch.long),
    spin=torch.tensor([int(atoms.info.get('spin', 0))], dtype=torch.long),
    dataset=[task],
    natoms=torch.tensor([n]),
    cell=torch.tensor(
      np.array(atoms.get_cell()), dtype=torch.float32
    ).unsqueeze(0)
    if any(atoms.pbc)
    else torch.eye(3).unsqueeze(0),
    cell_offsets=torch.tensor(cell_offsets),
    nedges=torch.tensor([n_e]),
  )


def _silu_np(x):
  return x / (1 + np.exp(-np.clip(x, -88, 88)))


class EndToEndParityTest(test_util.JAXMDTestCase):
  """Compare FairChem PyTorch and JAX UMA on 20 diverse structures."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    try:
      import torch
    except ImportError:
      raise absltest.SkipTest('torch not available')
    try:
      from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
    except ImportError:
      raise absltest.SkipTest('fairchem not installed')

    from jax_md._nn.uma.model_moe import UMAMoEBackbone, load_pretrained
    from jax_md._nn.uma.pretrained import load_checkpoint_raw
    from jax_md._nn.uma.heads import MLPEnergyHead

    try:
      config, jax_params, jax_head_params = load_pretrained(MODEL_NAME)
    except Exception as e:
      raise absltest.SkipTest(f'Could not load {MODEL_NAME}: {e}')

    cls.config = config
    cls.jax_model = UMAMoEBackbone(config=config)
    cls.jax_params = jax_params
    cls.jax_head = MLPEnergyHead(
      sphere_channels=config.sphere_channels,
      hidden_channels=config.hidden_channels,
    )
    cls.jax_head_params = jax_head_params

    from jax_md._nn.uma.pretrained import download_pretrained

    pt_path = download_pretrained(MODEL_NAME)
    ckpt = load_checkpoint_raw(pt_path)
    mc = ckpt.model_config['backbone']
    cls.ema_sd = {
      k.replace('module.', '', 1): v for k, v in ckpt.ema_state_dict.items()
    }
    get = (
      mc.get if isinstance(mc, dict) else (lambda k, d=None: getattr(mc, k, d))
    )
    ds_list = list(get('dataset_list', ['oc20', 'omol', 'omat', 'odac', 'omc']))
    cls.ds_list = ds_list
    cls.cutoff = float(get('cutoff', 6.0))

    for k in list(sys.modules.keys()):
      if k.startswith('fairchem') and not hasattr(sys.modules[k], '__file__'):
        del sys.modules[k]
    from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

    cls.pt_model = eSCNMDMoeBackbone(
      num_experts=get('num_experts', 64),
      sphere_channels=get('sphere_channels', 128),
      max_num_elements=get('max_num_elements', 100),
      lmax=get('lmax', 2),
      mmax=get('mmax', 2),
      num_layers=get('num_layers', 4),
      hidden_channels=get('hidden_channels', 128),
      cutoff=cls.cutoff,
      edge_channels=get('edge_channels', 128),
      num_distance_basis=get('num_distance_basis', 32),
      norm_type=get('norm_type', 'rms_norm_sh'),
      act_type=get('act_type', 'gate'),
      ff_type=get('ff_type', 'spectral'),
      chg_spin_emb_type=get('chg_spin_emb_type', 'rand_emb'),
      dataset_list=ds_list,
      use_composition_embedding=get('use_composition_embedding', True),
      model_version=get('model_version', 1.1),
      moe_dropout=0.0,
      otf_graph=False,
    )
    cls.pt_model.eval()
    bb_sd = {
      k.replace('backbone.', ''): v
      for k, v in cls.ema_sd.items()
      if k.startswith('backbone.')
    }
    cls.pt_model.load_state_dict(bb_sd, strict=False)

    cls.head_weights = cls._extract_head_weights(cls.ema_sd)

    all_structures = make_structures()
    cls.structures = {
      k: v for k, v in all_structures.items() if k in E2E_STRUCTURES
    }

  @staticmethod
  def _extract_head_weights(ema_sd):
    prefix = 'output_heads.energyandforcehead.head.energy_block.'
    weights = {}
    for idx in [0, 2, 4]:
      wkey = f'{prefix}{idx}.weight'
      bkey = f'{prefix}{idx}.bias'
      if wkey in ema_sd:
        weights[f'w{idx}'] = ema_sd[wkey].numpy()
        weights[f'b{idx}'] = ema_sd[bkey].numpy()
    if not weights:
      for k, v in ema_sd.items():
        if 'energy_block' in k and 'weight' in k:
          weights[k] = v.numpy()
        elif 'energy_block' in k and 'bias' in k:
          weights[k] = v.numpy()
    return weights

  def _run_comparison(self, name):
    import torch

    atoms, task = self.structures[name]
    n = len(atoms)

    data = _atoms_to_pt_data(atoms, task, self.cutoff, self.ds_list)
    with torch.no_grad():
      pt_out = self.pt_model(data)
    pt_emb = pt_out['node_embedding'].numpy()

    hw = self.head_weights
    scalar = pt_emb[:, 0, :]
    x = _silu_np(scalar @ hw['w0'].T + hw['b0'])
    x = _silu_np(x @ hw['w2'].T + hw['b2'])
    x = x @ hw['w4'].T + hw['b4']
    pt_energy = x.squeeze(-1).sum()

    edge_index, edge_vec, _ = _build_edges(atoms, self.cutoff)
    ei = jnp.array(edge_index.astype(np.int32))
    batch = jnp.zeros(n, dtype=jnp.int32)
    from jax_md._nn.uma.nn.embedding import dataset_names_to_indices

    ds_idx = dataset_names_to_indices([task], self.config.dataset_list)

    jax_out = self.jax_model.apply(
      self.jax_params,
      jnp.array(atoms.get_positions().astype(np.float32)),
      jnp.array(atoms.get_atomic_numbers()),
      batch,
      ei,
      jnp.array(edge_vec),
      jnp.array([int(atoms.info.get('charge', 0))], dtype=jnp.int32),
      jnp.array([int(atoms.info.get('spin', 0))], dtype=jnp.int32),
      ds_idx,
    )
    jax_emb = np.array(jax_out['node_embedding'])

    e_result = self.jax_head.apply(
      self.jax_head_params,
      jax_out['node_embedding'],
      batch,
      1,
    )
    jax_energy = float(e_result['energy'][0])

    self.assertTrue(np.all(np.isfinite(jax_emb)), f'{name}: NaN in JAX')
    self.assertEqual(pt_emb.shape, jax_emb.shape, f'{name}: shape mismatch')

    emb_max = np.max(np.abs(pt_emb - jax_emb))
    energy_diff = abs(pt_energy - jax_energy)

    self.assertLess(
      emb_max,
      0.05,
      f'{name}: embedding max diff {emb_max:.4f} exceeds threshold',
    )
    self.assertLess(
      energy_diff,
      0.05,
      f'{name}: energy diff {energy_diff:.4f} exceeds threshold',
    )


def _make_test_method(structure_name):
  def test_method(self):
    if structure_name not in self.structures:
      self.skipTest(f'{structure_name} not in uma_structures')
    self._run_comparison(structure_name)

  test_method.__doc__ = f'PT vs JAX parity for {structure_name}'
  return test_method


for _name in E2E_STRUCTURES:
  setattr(
    EndToEndParityTest,
    f'test_{_name}',
    _make_test_method(_name),
  )


if __name__ == '__main__':
  absltest.main()
