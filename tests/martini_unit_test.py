"""
Unit tests for LINCS (LINear Constraint Solver) implementation and
center-of-mass motion removal.
Tests cover topology building, helper functions, and the full constraint solver.

Run with:
  python martini_unit_test.py
or with JAX 64-bit mode enabled for numerical precision:
  JAX_ENABLE_X64=1 python martini_unit_test.py
"""

import os

os.environ['JAX_ENABLE_X64'] = (
  '1'  # Enable 64-bit floats for numerical precision
)

import numpy as np
from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_md import space

from jax import jit, random
from jax_md import dataclasses
from jax_md.simulate import NVTLangevinState

from jax_md.mm_forcefields.martini.cm_motion_remover import (
  remove_cm_motion,
  make_cm_remover,
)

from jax_md.mm_forcefields.martini.lincs import (
  LincsTopology,
  _compute_B,
  _build_A,
  _solve,
  _apply_sol,
  lincs_positions,
  apply_lincs,
)

from jax_md.test_util import JAXMDTestCase

# ---------------------------------------------------------------------------
# Shared tolerances
# ---------------------------------------------------------------------------
TIGHT = 1e-6  # constraint-length constraint satisfaction after LINCS
LOOSE = 1e-4  # acceptable for low-nrec or large-angle deviations

_DEFAULT_RNG = random.PRNGKey(0)


def free_displacement():
  """Displacement function for unbounded (free) space."""
  disp_fn, _ = space.free()
  return disp_fn


def linear_3atom_topo():
  """
  Three atoms in a line: 0-1-2.
  Two constraints: constraint 0-1 (length 1.0) and constraint 1-2 (length 1.5).
  Equal masses of 1.0.
  """
  constraint_pairs = np.array([[0, 1], [1, 2]], dtype=np.int32)
  constraint_lengths = np.array([1.0, 1.5])
  masses = np.array([1.0, 1.0, 1.0])
  topo = LincsTopology.build_lincs_topology(
    constraint_pairs, masses, constraint_lengths
  )
  return topo, constraint_pairs, constraint_lengths, masses


def single_constraint_topo(constraint_length=1.0, m1=1.0, m2=1.0):
  """Simplest possible topology: one constraint between atoms 0 and 1."""
  constraint_pairs = np.array([[0, 1]], dtype=np.int32)
  constraint_lengths = np.array([constraint_length])
  masses = np.array([m1, m2])
  topo = LincsTopology.build_lincs_topology(
    constraint_pairs, masses, constraint_lengths
  )
  return topo


def water_topo():
  """
  Rigid water-like molecule: O at index 0, H1 at 1, H2 at 2.
  Two O-H constraints of length 0.096 nm.  Masses: O=16, H=1.
  """
  constraint_pairs = np.array([[0, 1], [0, 2]], dtype=np.int32)
  constraint_lengths = np.array([0.096, 0.096])
  masses = np.array([16.0, 1.0, 1.0])
  topo = LincsTopology.build_lincs_topology(
    constraint_pairs, masses, constraint_lengths
  )
  return topo, constraint_pairs, constraint_lengths, masses


class TestBuildTopology(JAXMDTestCase):
  def test_single_constraint_shapes(self):
    topo = single_constraint_topo()
    self.assertEqual(topo.atom1.shape, (1,))
    self.assertEqual(topo.atom2.shape, (1,))
    self.assertEqual(topo.constraint_lengths.shape, (1,))
    self.assertEqual(topo.Sdiag.shape, (1,))
    self.assertEqual(topo.inv_masses.shape, (2,))
    self.assertEqual(topo.ncc.shape, (1,))
    self.assertEqual(topo.conn.shape[0], 1)

  def test_Sdiag_formula(self):
    """Sdiag[i] = 1/sqrt(1/m1 + 1/m2)."""
    m1, m2, d = 2.0, 4.0, 1.0
    topo = single_constraint_topo(constraint_length=d, m1=m1, m2=m2)
    expected = 1.0 / np.sqrt(1.0 / m1 + 1.0 / m2)
    np.testing.assert_allclose(np.array(topo.Sdiag), [expected], rtol=1e-12)

  def test_inv_masses(self):
    masses = np.array([2.0, 4.0, 1.0])
    constraint_pairs = np.array([[0, 1], [1, 2]])
    constraint_lengths = np.array([1.0, 1.5])
    topo = LincsTopology.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )
    np.testing.assert_allclose(
      np.array(topo.inv_masses), 1.0 / masses, rtol=1e-12
    )

  def test_no_coupling_independent_constraints(self):
    """Two constraints that share no atom should have zero coupling."""
    constraint_pairs = np.array([[0, 1], [2, 3]], dtype=np.int32)
    constraint_lengths = np.array([1.0, 1.0])
    masses = np.ones(4)
    topo = LincsTopology.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )
    self.assertTrue(
      np.all(topo.ncc == 0),
      'Independent constraints must have zero neighbours',
    )

  def test_coupling_linear_chain(self):
    """In a 3-atom chain each constraint has exactly one neighbour."""
    topo, *_ = linear_3atom_topo()
    np.testing.assert_array_equal(topo.ncc, [1, 1])

  def test_coupling_water(self):
    """Water: each O-H constraint is coupled to the other O-H constraint."""
    topo, *_ = water_topo()
    np.testing.assert_array_equal(topo.ncc, [1, 1])

  def test_virtual_site_raises(self):
    """Atoms with zero mass in constraints should raise ValueError."""
    constraint_pairs = np.array([[0, 1]], dtype=np.int32)
    constraint_lengths = np.array([1.0])
    masses = np.array([1.0, 0.0])  # atom 1 is virtual
    with self.assertRaisesRegex(ValueError, 'zero mass'):
      LincsTopology.build_lincs_topology(
        constraint_pairs, masses, constraint_lengths
      )

  def test_constraint_lengths_stored_correctly(self):
    constraint_lengths = np.array([0.096, 0.152])
    constraint_pairs = np.array([[0, 1], [1, 2]])
    masses = np.ones(3)
    topo = LincsTopology.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )
    np.testing.assert_allclose(
      np.array(topo.constraint_lengths), constraint_lengths
    )

  def test_sign_linear_chain(self):
    """
    In a linear chain 0-1-2 (constraints [0,1] and [1,2]):
    atom 1 is atom2 of constraint 0 and atom1 of constraint 1 → different slots → sign = +1.
    """
    topo, *_ = linear_3atom_topo()
    sign = topo.coef_sign[0, 0]  # sign for constraint-0 looking at constraint-1
    self.assertEqual(
      float(sign), +1.0, f'Expected +1 for linear chain, got {sign}'
    )

  def test_sign_shared_atom1(self):
    """
    Two constraints sharing atom1: [0,1] and [0,2].
    Shared atom (0) is atom1 in both → same slot → sign = -1.
    """
    topo, *_ = water_topo()
    sign = topo.coef_sign[0, 0]
    self.assertEqual(
      float(sign),
      -1.0,
      f'Expected -1 when shared atom is atom1 of both, got {sign}',
    )


class TestComputeB(JAXMDTestCase):
  def test_unit_length(self):
    """constraint vectors must be unit length."""
    topo = single_constraint_topo()
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    B = _compute_B(positions, topo, free_displacement())
    norms = jnp.linalg.norm(B, axis=-1)
    np.testing.assert_allclose(np.array(norms), [1.0], atol=1e-12)

  def test_direction(self):
    """B[i] = (r_a1 - r_a2) / |r_a1 - r_a2|."""
    topo = single_constraint_topo()
    r1 = jnp.array([1.0, 0.0, 0.0])
    r2 = jnp.array([0.0, 0.0, 0.0])
    positions = jnp.stack([r1, r2])
    B = _compute_B(positions, topo, free_displacement())
    np.testing.assert_allclose(np.array(B[0]), [1.0, 0.0, 0.0], atol=1e-12)

  def test_shape(self):
    topo, *_ = linear_3atom_topo()
    positions = jnp.zeros((3, 3))
    positions = positions.at[1].set(jnp.array([1.0, 0.0, 0.0]))
    positions = positions.at[2].set(jnp.array([2.5, 0.0, 0.0]))
    B = _compute_B(positions, topo, free_displacement())
    self.assertEqual(B.shape, (2, 3))


class TestBuildA(JAXMDTestCase):
  def test_shape(self):
    topo, *_ = linear_3atom_topo()
    positions = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
      ]
    )
    B = _compute_B(positions, topo, free_displacement())
    A = _build_A(B, topo)
    self.assertEqual(A.shape, (2, topo.cmax))

  def test_zero_for_independent_constraints(self):
    """Independent constraints (no shared atom) must produce A = 0."""
    constraint_pairs = np.array([[0, 1], [2, 3]])
    masses = np.ones(4)
    constraint_lengths = np.array([1.0, 1.0])
    topo = LincsTopology.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )
    positions = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
      ]
    )
    B = _compute_B(positions, topo, free_displacement())
    A = _build_A(B, topo)
    np.testing.assert_allclose(np.array(A), np.zeros_like(A), atol=1e-12)

  def test_collinear_constraints_magnitude(self):
    """
    For collinear constraints 0-1-2 with equal masses m and constraint lengths d:
      Sdiag = sqrt(m/2), A[0,0] = sign * invmass * Sdiag^2 * (B0·B1)
      B0·B1 = +1 (both point +x), sign = +1 (different slots)
      so A[0,0] = (1/m) * (m/2) * 1 = 0.5.
    """
    m = 2.0
    constraint_pairs = np.array([[0, 1], [1, 2]])
    masses = np.array([m, m, m])
    constraint_lengths = np.array([1.0, 1.0])
    topo = LincsTopology.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )
    positions = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
      ]
    )
    B = _compute_B(positions, topo, free_displacement())
    A = _build_A(B, topo)
    # For collinear constraints: A[0,0] = invmass[1] * Sdiag[0] * Sdiag[1] * 1.0 * sign
    # invmass[1] = 1/m, Sdiag = 1/sqrt(2/m) = sqrt(m/2)
    expected = (1.0 / m) * (m / 2.0) * 1.0  # = 0.5
    np.testing.assert_allclose(float(A[0, 0]), expected, rtol=1e-10)

  def test_perpendicular_constraints_zero_coupling(self):
    """Perpendicular constraints sharing a central atom have zero A coupling."""
    constraint_pairs = np.array([[0, 1], [0, 2]])
    masses = np.array([1.0, 1.0, 1.0])
    constraint_lengths = np.array([1.0, 1.0])
    topo = LincsTopology.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )
    positions = jnp.array(
      [
        [0.0, 0.0, 0.0],  # central atom
        [1.0, 0.0, 0.0],  # along x
        [0.0, 1.0, 0.0],  # along y
      ]
    )
    B = _compute_B(positions, topo, free_displacement())
    A = _build_A(B, topo)
    np.testing.assert_allclose(np.array(A), np.zeros_like(A), atol=1e-12)


class TestSolve(JAXMDTestCase):
  def test_nrec_zero_returns_rhs(self):
    """With nrec=0 the solver should return rhs unchanged."""
    rhs = jnp.array([1.0, 2.0])
    constraint_pairs = np.array([[0, 1], [2, 3]])
    masses = np.ones(4)
    constraint_lengths = np.array([1.0, 1.0])
    topo2 = LincsTopology.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )
    sol = _solve(rhs, jnp.zeros((2, topo2.cmax)), topo2, nrec=0)
    np.testing.assert_allclose(np.array(sol), np.array(rhs), atol=1e-12)

  def test_converges_for_zero_A(self):
    """When A = 0, (I - A)^{-1} rhs = rhs for all nrec."""
    constraint_pairs = np.array([[0, 1], [2, 3]])
    masses = np.ones(4)
    constraint_lengths = np.array([1.0, 1.0])
    topo = LincsTopology.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )
    rhs = jnp.array([3.0, -1.5])
    A = jnp.zeros((2, topo.cmax))
    for nrec in [1, 4, 8]:
      sol = _solve(rhs, A, topo, nrec=nrec)
      np.testing.assert_allclose(np.array(sol), np.array(rhs), atol=1e-12)


class TestApplySol(JAXMDTestCase):
  def test_momentum_conservation(self):
    """
    Total momentum change = 0 when masses are equal (corrections cancel).
    sum_i m_i * delta_i = 0 for a single constraint.
    """
    topo = single_constraint_topo(m1=2.0, m2=3.0)
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sol = jnp.array([1.0])
    B = jnp.array([[1.0, 0.0, 0.0]])
    pos_new = _apply_sol(positions, sol, B, topo)
    delta = pos_new - positions  # (2, 3)
    masses = np.array(topo.masses)
    momentum_change = (masses[:, None] * np.array(delta)).sum(axis=0)
    np.testing.assert_allclose(momentum_change, np.zeros(3), atol=1e-12)

  def test_correction_direction(self):
    """atom1 moves in -B direction, atom2 in +B direction."""
    topo = single_constraint_topo(m1=1.0, m2=1.0)
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    sol = jnp.array([1.0])
    B = jnp.array([[1.0, 0.0, 0.0]])
    pos_new = _apply_sol(positions, sol, B, topo)
    # atom1 (index 0) should shift in -x, atom2 (index 1) in +x
    self.assertLess(float(pos_new[0, 0]), float(positions[0, 0]))
    self.assertGreater(float(pos_new[1, 0]), float(positions[1, 0]))


class TestLincsPositions(JAXMDTestCase):
  def _check_constraint_lengths(self, pos_con, topo, atol=TIGHT):
    """Assert all constrained constraints have the correct length."""
    for i in range(len(topo.atom1)):
      a1, a2 = topo.atom1[i], topo.atom2[i]
      r1, r2 = np.array(pos_con[a1]), np.array(pos_con[a2])
      length = np.linalg.norm(r1 - r2)
      target = float(topo.constraint_lengths[i])
      np.testing.assert_allclose(
        length,
        target,
        atol=atol,
        err_msg=f'constraint {i} ({a1}-{a2}): got {length:.8f}, want {target:.8f}',
      )

  def test_single_constraint_already_satisfied(self):
    """If the unconstrained position already satisfies the constraint, LINCS should be a no-op."""
    topo = single_constraint_topo(constraint_length=1.0)
    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions_unc = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pos_con, _, _ = lincs_positions(
      positions_old, positions_unc, topo, free_displacement(), nrec=4
    )
    self._check_constraint_lengths(pos_con, topo)

  def test_single_constraint_stretched(self):
    """Correct a stretched constraint (length 2.0 → target 1.0)."""
    topo = single_constraint_topo(constraint_length=1.0)
    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions_unc = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pos_con, _, _ = lincs_positions(
      positions_old, positions_unc, topo, free_displacement(), nrec=4
    )
    self._check_constraint_lengths(pos_con, topo)

  def test_single_constraint_compressed(self):
    """Correct a compressed constraint (length 0.5 → target 1.0)."""
    topo = single_constraint_topo(constraint_length=1.0)
    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions_unc = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    pos_con, _, _ = lincs_positions(
      positions_old, positions_unc, topo, free_displacement(), nrec=4
    )
    self._check_constraint_lengths(pos_con, topo)

  def test_single_constraint_unequal_masses(self):
    """Constraint with unequal masses (heavy atom + light atom)."""
    topo = single_constraint_topo(constraint_length=1.5, m1=12.0, m2=1.0)
    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    positions_unc = jnp.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]])
    pos_con, _, _ = lincs_positions(
      positions_old, positions_unc, topo, free_displacement(), nrec=4
    )
    self._check_constraint_lengths(pos_con, topo)

  def test_linear_chain(self):
    """Both constraints in a 3-atom chain must be satisfied simultaneously."""
    topo, *_ = linear_3atom_topo()
    positions_old = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
      ]
    )
    positions_unc = jnp.array(
      [
        [0.01, 0.0, 0.0],
        [1.015, 0.0, 0.0],
        [2.485, 0.0, 0.0],
      ]
    )
    pos_con, _, _ = lincs_positions(
      positions_old, positions_unc, topo, free_displacement(), nrec=8
    )
    self._check_constraint_lengths(pos_con, topo)

  def test_water_both_constraints(self):
    """Water-like molecule: both O-H constraints must be constrained."""
    topo, *_ = water_topo()
    positions_old = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [0.096, 0.0, 0.0],
        [0.0, 0.096, 0.0],
      ]
    )
    positions_unc = jnp.array(
      [
        [0.001, 0.001, 0.0],
        [0.11, 0.005, 0.0],
        [-0.005, 0.10, 0.0],
      ]
    )
    pos_con, _, _ = lincs_positions(
      positions_old, positions_unc, topo, free_displacement(), nrec=6
    )
    self._check_constraint_lengths(pos_con, topo, atol=TIGHT)

  def test_3d_perturbation(self):
    """constraint correction works for out-of-plane displacements."""
    topo = single_constraint_topo(constraint_length=1.0)
    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions_unc = jnp.array([[0.1, 0.2, 0.3], [1.3, 0.4, 0.5]])
    pos_con, _, _ = lincs_positions(
      positions_old, positions_unc, topo, free_displacement(), nrec=4
    )
    self._check_constraint_lengths(pos_con, topo)

  def test_nrec_monotone_improvement(self):
    """Higher nrec should give equal or better constraint satisfaction."""
    topo = single_constraint_topo(constraint_length=1.0)
    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions_unc = jnp.array(
      [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
    )  # large stretch
    errors = []
    for nrec in [1, 2, 4, 8]:
      pos_con, _, _ = lincs_positions(
        positions_old, positions_unc, topo, free_displacement(), nrec=nrec
      )
      r1, r2 = np.array(pos_con[0]), np.array(pos_con[1])
      errors.append(abs(np.linalg.norm(r1 - r2) - 1.0))
    for i in range(len(errors) - 1):
      self.assertLessEqual(
        errors[i + 1],
        errors[i] + 1e-14,
        msg=(
          f'Error increased from nrec={i + 1} ({errors[i]:.2e})'
          f' to nrec={i + 2} ({errors[i + 1]:.2e})'
        ),
      )

  def test_returns_three_values(self):
    """lincs_positions must return (positions, B, mlambda)."""
    topo = single_constraint_topo()
    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = lincs_positions(
      positions_old, positions_old, topo, free_displacement()
    )
    self.assertLen(result, 3)

  def test_B_unit_vectors(self):
    """B returned by lincs_positions must be unit vectors."""
    topo, *_ = linear_3atom_topo()
    positions_old = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
      ]
    )
    _, B, _ = lincs_positions(
      positions_old, positions_old, topo, free_displacement()
    )
    norms = np.linalg.norm(np.array(B), axis=-1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-12)

  def test_jit_compatible(self):
    """lincs_positions must be JIT-compilable."""
    topo = single_constraint_topo()
    disp_fn = free_displacement()

    @jax.jit
    def f(old, unc):
      return lincs_positions(old, unc, topo, disp_fn, nrec=4)

    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions_unc = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    pos_con, _, _ = f(positions_old, positions_unc)
    r1, r2 = np.array(pos_con[0]), np.array(pos_con[1])
    np.testing.assert_allclose(np.linalg.norm(r1 - r2), 1.0, atol=TIGHT)


class TestApplyLincs(JAXMDTestCase):
  def _constraint_length(self, pos, a1, a2):
    return float(jnp.linalg.norm(pos[a1] - pos[a2]))

  def test_free_space_single_constraint(self):
    """apply_lincs with no box (free space) constrains a single constraint."""
    topo = single_constraint_topo(constraint_length=1.0)
    disp_fn, shift_fn = space.free()
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
    R_con, dV = apply_lincs(R_old, R_unc, topo, disp_fn, dt=0.002)
    length = self._constraint_length(R_con, 0, 1)
    np.testing.assert_allclose(length, 1.0, atol=TIGHT)

  def test_velocity_correction_shape(self):
    """dV must have same shape as positions."""
    topo = single_constraint_topo()
    disp_fn, shift_fn = space.free()
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    R_con, dV = apply_lincs(R_old, R_unc, topo, disp_fn, dt=0.002)
    self.assertEqual(dV.shape, R_old.shape)

  def test_velocity_correction_nonzero_when_constraint_violated(self):
    """If the constraint was violated, velocity correction must be non-zero."""
    topo = single_constraint_topo()
    disp_fn, shift_fn = space.free()
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    _, dV = apply_lincs(R_old, R_unc, topo, disp_fn, dt=0.002)
    self.assertTrue(
      jnp.any(dV != 0.0),
      'Velocity correction must be non-zero for a violated constraint',
    )

  def test_velocity_correction_zero_when_satisfied(self):
    """If the constraint is already at target length, dV should be essentially zero."""
    topo = single_constraint_topo(constraint_length=1.0)
    disp_fn, shift_fn = space.free()
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _, dV = apply_lincs(R_old, R_unc, topo, disp_fn, dt=0.002)
    np.testing.assert_allclose(np.array(dV), np.zeros_like(dV), atol=1e-8)

  def test_dt_scales_velocity_correction(self):
    """dV should scale inversely with dt (mlambda / dt)."""
    topo = single_constraint_topo()
    disp_fn, shift_fn = space.free()
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    _, dV1 = apply_lincs(R_old, R_unc, topo, disp_fn, dt=0.001)
    _, dV2 = apply_lincs(R_old, R_unc, topo, disp_fn, dt=0.002)
    np.testing.assert_allclose(
      np.array(dV1),
      np.array(dV2) * 2.0,
      rtol=1e-8,
      err_msg='Velocity correction should scale as 1/dt',
    )

  def test_periodic_box_single_constraint(self):
    """apply_lincs with a periodic box and real-space coordinates."""
    topo = single_constraint_topo(constraint_length=1.0)
    box = jnp.eye(3) * 10.0
    disp_fn, shift_fn = space.periodic_general(
      box, fractional_coordinates=False
    )
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]])
    R_con, dV = apply_lincs(
      R_old,
      R_unc,
      topo,
      disp_fn,
      dt=0.002,
      box=box,
      fractional_coordinates=False,
    )
    length = self._constraint_length(R_con, 0, 1)
    np.testing.assert_allclose(length, 1.0, atol=TIGHT)

  def test_jit_compatible(self):
    """apply_lincs must be JIT-compilable."""
    topo = single_constraint_topo()
    disp_fn, shift_fn = space.free()

    @jax.jit
    def f(old, unc):
      return apply_lincs(old, unc, topo, disp_fn, dt=0.002)

    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    R_con, dV = f(R_old, R_unc)
    length = self._constraint_length(R_con, 0, 1)
    np.testing.assert_allclose(length, 1.0, atol=TIGHT)


class TestFractionalVsRealCoordinates(JAXMDTestCase):
  def _run_both_modes(self, R_old_real, R_unc_real, topo, box, dt=0.002):
    """Run apply_lincs in real and fractional coordinate modes, return results."""
    disp_real, shift_real = space.periodic_general(
      box, fractional_coordinates=False
    )
    R_con_real, dV_real = apply_lincs(
      R_old_real,
      R_unc_real,
      topo,
      disp_real,
      dt=dt,
      box=box,
      fractional_coordinates=False,
    )

    # Convert to fractional coordinates
    inv_box = space.inverse(box)
    R_old_frac = space.transform(inv_box, R_old_real)
    R_unc_frac = space.transform(inv_box, R_unc_real)

    disp_frac, shift_frac = space.periodic_general(
      box, fractional_coordinates=True
    )
    R_con_frac, dV_frac = apply_lincs(
      R_old_frac,
      R_unc_frac,
      topo,
      disp_frac,
      dt=dt,
      box=box,
      fractional_coordinates=True,
    )
    # Convert constrained positions back to real space for comparison
    R_con_frac_real = space.transform(box, R_con_frac)
    dV_frac_real = space.transform(box, dV_frac)

    return R_con_real, dV_real, R_con_frac_real, dV_frac_real

  def test_single_constraint_positions_agree(self):
    """Constrained positions from fractional and real modes must match."""
    topo = single_constraint_topo(constraint_length=1.0)
    box = jnp.eye(3) * 10.0
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]])

    R_con_real, _, R_con_frac_real, _ = self._run_both_modes(
      R_old, R_unc, topo, box
    )
    np.testing.assert_allclose(
      np.array(R_con_real),
      np.array(R_con_frac_real),
      atol=1e-6,
      err_msg='Constrained positions differ between real and fractional modes',
    )

  def test_single_constraint_velocity_correction_agrees(self):
    """Velocity corrections from fractional and real modes must match."""
    topo = single_constraint_topo(constraint_length=1.0)
    box = jnp.eye(3) * 10.0
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]])

    _, dV_real, _, dV_frac_real = self._run_both_modes(R_old, R_unc, topo, box)
    np.testing.assert_allclose(
      np.array(dV_real),
      np.array(dV_frac_real),
      atol=1e-6,
      err_msg='Velocity corrections differ between real and fractional modes',
    )

  def test_constraint_satisfied_in_both_modes(self):
    """Both modes must satisfy the constraint length to within TIGHT tolerance."""
    topo = single_constraint_topo(constraint_length=1.0)
    box = jnp.eye(3) * 10.0
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.7, 0.0, 0.0]])

    R_con_real, _, R_con_frac_real, _ = self._run_both_modes(
      R_old, R_unc, topo, box
    )
    for label, R_con in [('real', R_con_real), ('fractional', R_con_frac_real)]:
      length = float(jnp.linalg.norm(R_con[0] - R_con[1]))
      np.testing.assert_allclose(
        length,
        1.0,
        atol=TIGHT,
        err_msg=f'Constraint not satisfied in {label} mode: got {length:.8f}',
      )

  def test_water_positions_agree(self):
    """Multi-constraint water topology: both modes give identical constrained positions."""
    topo, *_ = water_topo()
    box = jnp.eye(3) * 10.0
    R_old = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [0.096, 0.0, 0.0],
        [0.0, 0.096, 0.0],
      ]
    )
    R_unc = jnp.array(
      [
        [0.001, 0.001, 0.0],
        [0.11, 0.005, 0.0],
        [-0.005, 0.10, 0.0],
      ]
    )
    R_con_real, dV_real, R_con_frac_real, dV_frac_real = self._run_both_modes(
      R_old, R_unc, topo, box
    )
    np.testing.assert_allclose(
      np.array(R_con_real),
      np.array(R_con_frac_real),
      atol=1e-6,
      err_msg='Water constrained positions differ between real and fractional modes',
    )
    np.testing.assert_allclose(
      np.array(dV_real),
      np.array(dV_frac_real),
      atol=1e-6,
      err_msg='Water velocity corrections differ between real and fractional modes',
    )

  def test_non_cubic_box_positions_agree(self):
    """Agrees under a non-cubic (orthorhombic) box."""
    topo = single_constraint_topo(constraint_length=1.0)
    box = jnp.diag(jnp.array([8.0, 9.0, 11.0]))
    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.1, 0.05, 0.0], [1.6, 0.1, 0.0]])

    R_con_real, dV_real, R_con_frac_real, dV_frac_real = self._run_both_modes(
      R_old, R_unc, topo, box
    )
    np.testing.assert_allclose(
      np.array(R_con_real),
      np.array(R_con_frac_real),
      atol=1e-6,
      err_msg='Non-cubic box: positions differ between real and fractional modes',
    )
    np.testing.assert_allclose(
      np.array(dV_real),
      np.array(dV_frac_real),
      atol=1e-6,
      err_msg='Non-cubic box: velocity corrections differ between real and fractional modes',
    )

  def test_jit_compatible_fractional(self):
    """apply_lincs in fractional mode must be JIT-compilable."""
    topo = single_constraint_topo(constraint_length=1.0)
    box = jnp.eye(3) * 10.0
    disp_frac, shift_frac = space.periodic_general(
      box, fractional_coordinates=True
    )
    inv_box = space.inverse(box)

    @jax.jit
    def f(R_old_real, R_unc_real):
      R_old_frac = space.transform(inv_box, R_old_real)
      R_unc_frac = space.transform(inv_box, R_unc_real)
      return apply_lincs(
        R_old_frac,
        R_unc_frac,
        topo,
        disp_frac,
        dt=0.02,
        box=box,
        fractional_coordinates=True,
      )

    R_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    R_unc = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    R_con_frac, _ = f(R_old, R_unc)
    R_con_real = space.transform(box, R_con_frac)
    length = float(jnp.linalg.norm(R_con_real[0] - R_con_real[1]))
    np.testing.assert_allclose(length, 1.0, atol=TIGHT)


class TestNumericalRegression(JAXMDTestCase):
  def test_mlambda_sign_convention(self):
    """
    For a stretched constraint, mlambda should be positive:
    the multiplier brings atoms closer together.
    """
    topo = single_constraint_topo(constraint_length=1.0)
    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions_unc = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    _, _, mlambda = lincs_positions(
      positions_old, positions_unc, topo, free_displacement()
    )
    self.assertGreater(
      float(mlambda[0]),
      0,
      'mlambda should be positive for a stretched constraint',
    )

  def test_symmetric_masses_equal_displacement(self):
    """With equal masses, both atoms should move by the same amount (opposite directions)."""
    topo = single_constraint_topo(constraint_length=1.0, m1=1.0, m2=1.0)
    positions_old = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions_unc = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    pos_con, _, _ = lincs_positions(
      positions_old, positions_unc, topo, free_displacement()
    )
    d0 = np.array(pos_con[0]) - np.array(positions_unc[0])
    d1 = np.array(pos_con[1]) - np.array(positions_unc[1])
    np.testing.assert_allclose(np.abs(d0), np.abs(d1), atol=1e-10)

  def test_large_system_no_nan(self):
    """A longer chain of 10 atoms should produce finite constrained positions."""
    n = 10
    constraint_pairs = np.array(
      [[i, i + 1] for i in range(n - 1)], dtype=np.int32
    )
    constraint_lengths = np.ones(n - 1)
    masses = np.ones(n)
    topo = LincsTopology.build_lincs_topology(
      constraint_pairs, masses, constraint_lengths
    )

    positions_old = jnp.array([[float(i), 0.0, 0.0] for i in range(n)])
    rng = jax.random.PRNGKey(42)
    noise = jax.random.normal(rng, (n, 3)) * 0.05
    positions_unc = positions_old + noise

    pos_con, _, _ = lincs_positions(
      positions_old, positions_unc, topo, free_displacement(), nrec=4
    )
    self.assertFalse(
      jnp.any(jnp.isnan(pos_con)), 'Constrained positions contain NaN'
    )
    self.assertFalse(
      jnp.any(jnp.isinf(pos_con)), 'Constrained positions contain Inf'
    )


def _make_state(momentum, mass, position=None, force=None, rng=None):
  """Return an NVTLangevinState with JAX arrays.

  position and force default to zero arrays matching momentum's shape.
  rng defaults to a fixed PRNGKey.
  """
  momentum = jnp.array(momentum, dtype=float)
  mass = jnp.array(mass, dtype=float)

  if position is None:
    position = jnp.zeros_like(momentum)
  else:
    position = jnp.array(position, dtype=float)

  if force is None:
    force = jnp.zeros_like(momentum)
  else:
    force = jnp.array(force, dtype=float)

  if rng is None:
    rng = _DEFAULT_RNG

  return NVTLangevinState(
    position=position,
    momentum=momentum,
    force=force,
    mass=mass,
    rng=rng,
  )


def _com_momentum(state):
  """Compute total (summed) momentum of a state."""
  return jnp.sum(state.momentum, axis=0)


class TestRemoveCmMotion(JAXMDTestCase):
  # --- Basic correctness ---------------------------------------------------

  def test_zero_com_unchanged(self):
    """State with zero net momentum should be unchanged."""
    momentum = [[1.0, -1.0], [-1.0, 1.0]]
    mass = [1.0, 1.0]
    state = _make_state(momentum, mass)

    new_state = remove_cm_motion(state)

    self.assertAllClose(new_state.momentum, state.momentum, atol=1e-6)

  def test_net_momentum_is_zero_after_removal(self):
    """Total momentum must be (near) zero after COM removal."""
    momentum = [[3.0, 1.0], [2.0, -0.5], [-1.0, 0.3]]
    mass = [1.0, 2.0, 0.5]
    state = _make_state(momentum, mass)

    new_state = remove_cm_motion(state)
    p_total = _com_momentum(new_state)

    expected_momentum = jnp.array(
      [
        [13 / 7, 27 / 35],
        [-2 / 7, -33.5 / 35],
        [-11 / 7, 6.5 / 35],
      ]
    )
    self.assertAllClose(new_state.momentum, expected_momentum, atol=1e-6)
    self.assertAllClose(p_total, jnp.zeros(2), atol=1e-6)

  def test_single_particle(self):
    """Single particle: all momentum is COM, result must be zero."""
    momentum = [[4.0, -3.0]]
    mass = [2.0]
    state = _make_state(momentum, mass)

    new_state = remove_cm_motion(state)

    self.assertAllClose(new_state.momentum, jnp.zeros((1, 2)), atol=1e-6)

  def test_equal_masses(self):
    """Equal masses: simple average velocity is subtracted from each."""
    momentum = [[3.0, 0.0], [1.0, 0.0]]
    mass = [1.0, 1.0]
    state = _make_state(momentum, mass)

    new_state = remove_cm_motion(state)

    # v_com = (3+1)/(1+1) = 2; correction = m*v_com = 2 each
    expected = jnp.array([[1.0, 0.0], [-1.0, 0.0]])
    self.assertAllClose(new_state.momentum, expected, atol=1e-6)

  def test_3d_particles(self):
    """Works in 3-D."""
    momentum = [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [2.0, 0.0, 0.0]]
    mass = [1.0, 1.0, 1.0]
    state = _make_state(momentum, mass)

    new_state = remove_cm_motion(state)
    p_total = _com_momentum(new_state)

    self.assertAllClose(p_total, jnp.zeros(3), atol=1e-6)

  # --- Mass shape handling -------------------------------------------------

  def test_mass_1d_input(self):
    """1-D mass array should be handled correctly."""
    momentum = [[2.0, 0.0], [0.0, 2.0]]
    mass = [1.0, 1.0]  # 1-D
    state = _make_state(momentum, mass)

    new_state = remove_cm_motion(state)
    p_total = _com_momentum(new_state)

    self.assertAllClose(p_total, jnp.zeros(2), atol=1e-6)

  def test_mass_column_vector_input(self):
    """Column-vector mass [N, 1] should also work."""
    momentum = [[2.0, 0.0], [0.0, 2.0]]
    mass = [[1.0], [1.0]]  # 2-D column
    state = _make_state(momentum, mass)

    new_state = remove_cm_motion(state)
    p_total = _com_momentum(new_state)

    self.assertAllClose(p_total, jnp.zeros(2), atol=1e-6)

  def test_unequal_masses(self):
    """Heavier particles contribute more to COM correction."""
    mass = [1.0, 4.0]
    momentum = [[5.0, 0.0], [0.0, 0.0]]  # p_com = [5, 0], v_com = [1, 0]
    state = _make_state(momentum, mass)

    new_state = remove_cm_motion(state)
    p_total = _com_momentum(new_state)

    self.assertAllClose(p_total, jnp.zeros(2), atol=1e-6)

  # --- Zero-mass particles -------------------------------------------------

  def test_zero_mass_particle_ignored(self):
    """Particles with zero mass must not contribute to COM correction."""
    mass = [0.0, 2.0]
    momentum = [[0.0, 0.0], [4.0, 0.0]]
    state = _make_state(momentum, mass)

    new_state = remove_cm_motion(state)
    p_total = _com_momentum(new_state)

    self.assertAllClose(p_total, jnp.zeros(2), atol=1e-6)

  # --- Return-type contract ------------------------------------------------

  def test_returns_same_type(self):
    """remove_cm_motion must return the same dataclass type."""
    state = _make_state([[1.0, 0.0]], [1.0])
    new_state = remove_cm_motion(state)
    self.assertIs(type(new_state), type(state))

  def test_momentum_field_updated(self):
    """The returned state must have a different momentum field when COM ≠ 0."""
    state = _make_state([[5.0, 3.0], [0.0, 0.0]], [1.0, 1.0])
    new_state = remove_cm_motion(state)
    self.assertFalse(jnp.allclose(state.momentum, new_state.momentum))

  def test_mass_field_unchanged(self):
    """Mass must not be modified by COM removal."""
    mass = [1.0, 2.0]
    state = _make_state([[1.0, 0.0], [0.0, 1.0]], mass)
    new_state = remove_cm_motion(state)
    self.assertAllClose(new_state.mass, state.mass)

  # --- Idempotency ---------------------------------------------------------

  def test_idempotent(self):
    """Applying COM removal twice should equal applying it once."""
    state = _make_state([[3.0, -1.0], [-2.0, 4.0]], [1.0, 2.0])
    once = remove_cm_motion(state)
    twice = remove_cm_motion(once)
    self.assertAllClose(twice.momentum, once.momentum, atol=1e-6)

  # --- JIT compatibility ---------------------------------------------------

  def test_jit_compatible(self):
    """remove_cm_motion must survive jit compilation."""
    jitted = jit(remove_cm_motion)
    state = _make_state([[2.0, 1.0], [-1.0, 0.5]], [1.0, 1.0])
    result = jitted(state)
    p_total = _com_momentum(result)
    self.assertAllClose(p_total, jnp.zeros(2), atol=1e-6)


class TestMakeCmRemover(JAXMDTestCase):
  # --- Wrapper structure ---------------------------------------------------

  def test_returns_callable(self):
    """make_cm_remover must return a callable."""
    identity = lambda state, **kw: state
    wrapped = make_cm_remover(identity, freq=1)
    self.assertTrue(callable(wrapped))

  def test_apply_fn_is_called(self):
    """The inner apply_fn must be invoked exactly once per call."""
    calls = []

    def counting_apply(state, **kw):
      calls.append(1)
      return state

    wrapped = make_cm_remover(counting_apply, freq=1)
    state = _make_state([[1.0, 0.0]], [1.0])
    wrapped(0, state)
    self.assertLen(calls, 1)

  # --- COM removal at freq=1 -----------------------------------------------

  def test_freq1_removes_com_every_step(self):
    """With freq=1 every step should yield zero total momentum."""
    identity = lambda state, **kw: state
    wrapped = make_cm_remover(identity, freq=1)

    state = _make_state([[3.0, 1.0], [-1.0, 2.0]], [1.0, 1.0])

    for i in range(5):
      state = wrapped(i, state)
      p_total = _com_momentum(state)
      self.assertAllClose(
        p_total,
        jnp.zeros(2),
        atol=1e-6,
        err_msg=f'COM not removed at step {i}',
      )

  # --- COM removal at freq>1 -----------------------------------------------

  def test_freq_gt1_removes_com_on_multiples(self):
    """With freq=3, COM removal should happen only on steps 0, 3, 6 …"""

    # apply_fn adds a constant COM drift each step
    def drifting_apply(state, **kw):
      drift = jnp.array([[1.0, 0.0], [1.0, 0.0]])
      return dataclasses.replace(state, momentum=state.momentum + drift)

    freq = 3
    wrapped = make_cm_remover(drifting_apply, freq=freq)
    state = _make_state([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])

    for i in range(9):
      state = wrapped(i, state)
      p_total = _com_momentum(state)
      if i % freq == 0:
        self.assertAllClose(
          p_total,
          jnp.zeros(2),
          atol=1e-6,
          err_msg=f'COM should be zero at step {i}',
        )
      else:
        # drift has accumulated; total momentum must be non-zero
        self.assertGreater(
          float(jnp.linalg.norm(p_total)),
          0.5,
          msg=f'Expected drift at step {i}, got p_total={p_total}',
        )

  # --- kwargs forwarding ---------------------------------------------------

  def test_kwargs_forwarded_to_apply_fn(self):
    """Extra keyword arguments must reach the inner apply_fn."""
    received_kwargs = {}

    def recording_apply(state, **kw):
      received_kwargs.update(kw)
      return state

    wrapped = make_cm_remover(recording_apply, freq=1)
    state = _make_state([[0.0, 0.0]], [1.0])
    wrapped(0, state, temperature=300.0, dt=0.001)

    self.assertIn('temperature', received_kwargs)
    self.assertIn('dt', received_kwargs)

  # --- JIT compatibility ---------------------------------------------------

  def test_wrapped_fn_is_jit_compiled(self):
    """make_cm_remover must return a jit-compiled function (has __wrapped__ or runs)."""
    identity = lambda state, **kw: state
    wrapped = make_cm_remover(identity, freq=1)
    state = _make_state([[2.0, -1.0], [-2.0, 1.0]], [1.0, 1.0])
    # Just confirm it runs without error under jit
    result = wrapped(0, state)
    self.assertIsNotNone(result)

  # --- Step-index type handling --------------------------------------------

  def test_integer_step_index(self):
    """Step index supplied as a plain Python int must work."""
    identity = lambda state, **kw: state
    wrapped = make_cm_remover(identity, freq=2)
    state = _make_state([[1.0, 0.0], [-1.0, 0.0]], [1.0, 1.0])
    # Should not raise
    result = wrapped(4, state)
    self.assertIsNotNone(result)

  # --- Momentum conservation after apply_fn --------------------------------

  def test_state_momentum_zero_after_removal(self):
    """After a step where removal fires, net momentum must be zero."""

    def apply_adds_drift(state, **kw):
      extra = jnp.array([[5.0, -2.0], [5.0, -2.0]])
      return dataclasses.replace(state, momentum=state.momentum + extra)

    wrapped = make_cm_remover(apply_adds_drift, freq=1)
    state = _make_state([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])
    new_state = wrapped(0, state)
    p_total = _com_momentum(new_state)

    self.assertAllClose(p_total, jnp.zeros(2), atol=1e-6)


if __name__ == '__main__':
  absltest.main()
