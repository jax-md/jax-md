"""Tests for OPLSAA forcefield."""

from absl.testing import absltest
import os

import jax
from jax import jit, grad
import jax.numpy as jnp

from jax_md.mm_forcefields import oplsaa
from jax_md.mm_forcefields.nonbonded.electrostatics import (
  PMECoulomb,
  CutoffCoulomb,
)
from jax_md.mm_forcefields.base import NonbondedOptions
from jax_md import quantity
from jax_md import test_util as jtu

jax.config.parse_flags_with_absl()


class OPLSAAEnergyTest(jtu.JAXMDTestCase):
  """Tests for OPLSAA energy computation."""

  def setUp(self):
    """Set up test fixtures - load system and initialize energy function."""
    data_dir = 'notebooks/data/torsion-data'
    pdb_file = os.path.join(data_dir, 'scan_1.pdb')
    prm_file = os.path.join(data_dir, 'scan_1.prm')
    rtf_file = os.path.join(data_dir, 'scan_1.rtf')
    self.positions, self.topology, self.parameters = oplsaa.load_charmm_system(
      pdb_file, prm_file, rtf_file
    )

    coords_range = jnp.max(self.positions, axis=0) - jnp.min(
      self.positions, axis=0
    )
    box_size = coords_range + 20.0
    self.box = jnp.array([box_size[0], box_size[1], box_size[2]])
    nb_options = NonbondedOptions(
      r_cut=12.0,
      dr_threshold=0.5,
      scale_14_lj=0.5,
      scale_14_coul=0.5,
      use_soft_lj=False,
      use_shift_lj=False,
    )
    coulomb = PMECoulomb(grid_size=32, alpha=0.3, r_cut=12.0)
    self.energy_fn, self.neighbor_fn, self.displacement_fn = oplsaa.energy(
      self.topology, self.parameters, self.box, coulomb, nb_options
    )

    self.nlist = self.neighbor_fn.allocate(self.positions)

  def test_charmm_system_parsing(self):
    """Test that CHARMM files can be parsed successfully."""
    self.assertIsNotNone(self.positions)
    self.assertIsNotNone(self.topology)
    self.assertIsNotNone(self.parameters)

    self.assertEqual(self.positions.shape[1], 3)
    self.assertEqual(self.topology.n_atoms, self.positions.shape[0])
    self.assertGreater(self.topology.n_atoms, 0)

  def test_energy_computation(self):
    """Test that energy can be computed and print the terms."""
    E = self.energy_fn(self.positions, self.nlist)

    self.assertIsInstance(E, dict)
    expected_keys = [
      'bond',
      'angle',
      'torsion',
      'improper',
      'lj',
      'coulomb',
      'total',
    ]
    for key in expected_keys:
      self.assertIn(key, E)
      self.assertIsInstance(E[key], jnp.ndarray)
      self.assertEqual(E[key].shape, ())

    # Extract energies
    E_bond = float(E['bond'])
    E_angle = float(E['angle'])
    E_torsion = float(E['torsion'])
    E_improper = float(E['improper'])
    E_lj = float(E['lj'])
    E_coul = float(E['coulomb'])
    E_total = float(E['total'])

    E_jax = {
      'Bond': E_bond,
      'Angle': E_angle,
      'Torsion': E_torsion,
      'Improper': E_improper,
      'vdwl': E_lj,
      'Coulomb': E_coul,
      'Total': E_total,
    }

    E_lammps = {
      'Bond': 1.4015458,
      'Angle': 0.21927227,
      'Torsion': 0.0,
      'Improper': 3.1140185e-05,
      'vdwl': 11.911979,
      'Coulomb': -1.0519856,
      'Total': 12.480843,
    }

    # NOTE: Higher tolerance for OPLSAA could be related
    # to the libpargen (https://zarbi.chem.yale.edu/ligpargen/)
    for key in E_jax:
      self.assertAllClose(E_jax[key], E_lammps[key], rtol=5e-2, atol=5e-2)
      diff = E_jax[key] - E_lammps[key]
      print(
        f'{key}: JAX-MD = {E_jax[key]:.6f}, LAMMPS = {E_lammps[key]:.6f}, diff = {diff:.6e}'
      )

  def test_force_computation(self):
    """Test that forces can be computed."""

    def total_energy_fn(pos, nlist):
      E = self.energy_fn(pos, nlist)
      return E['total']

    force_fn = quantity.force(total_energy_fn)
    forces = force_fn(self.positions, self.nlist)

    self.assertEqual(forces.shape, self.positions.shape)
    self.assertTrue(jnp.all(jnp.isfinite(forces)))

  def test_jit_compilation(self):
    """Test that energy function can be JIT compiled."""
    energy_fn_jit = jit(self.energy_fn)

    E = energy_fn_jit(self.positions, self.nlist)

    self.assertIsInstance(E, dict)
    self.assertIn('total', E)
    self.assertTrue(jnp.isfinite(E['total']))

  def test_gradient_computation(self):
    """Test that gradients can be computed."""

    def total_energy_fn(pos, nlist):
      E = self.energy_fn(pos, nlist)
      return E['total']

    grad_fn = grad(total_energy_fn, argnums=0)
    gradients = grad_fn(self.positions, self.nlist)

    self.assertEqual(gradients.shape, self.positions.shape)
    self.assertTrue(jnp.all(jnp.isfinite(gradients)))

  def test_stress_computation(self):
    """Test stress computation via autodiff w.r.t. the box."""

    def energy_wrt_box(box):
      # Create a temporary wrapper around energy_fn that replaces self.box
      # but keeps positions and neighbor list fixed
      # Assumes energy_fn is differentiable w.r.t. box
      # Note: energy_fn expects only (positions, nlist)
      energy = oplsaa.energy(
        self.topology,
        self.parameters,
        box,  # pass box as variable
        CutoffCoulomb(alpha=0.3, r_cut=12.0),
        NonbondedOptions(
          r_cut=12.0,
          dr_threshold=0.5,
          scale_14_lj=0.5,
          scale_14_coul=0.5,
          use_soft_lj=False,
          use_shift_lj=False,
        ),
      )[0](self.positions, self.nlist)
      return energy['total']

    # Compute stress = - dE/d(box) using autodiff
    stress = -jax.grad(energy_wrt_box)(self.box)

    # Check that values are finite
    self.assertTrue(jnp.all(jnp.isfinite(stress)))
    # Check shape (per-axis stress)
    self.assertEqual(stress.shape, (3,))


if __name__ == '__main__':
  absltest.main()
