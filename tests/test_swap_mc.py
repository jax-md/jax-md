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

"""Tests for the species-swap Monte Carlo simulator (swap_mc).

Verifies:
  - Stoichiometry is preserved after each MC step.
  - MCState has the correct fields (position, species, key).
  - Energy is finite after each step.
  - The step function compiles and runs under jax.jit without error.
"""

import sys
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax import random

# Allow running from repo root without an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_md import energy, simulate, space, smap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ATOMS = 16          # total number of atoms
N_SPECIES_0 = 8       # number of species-0 atoms
N_SPECIES_1 = 8       # number of species-1 atoms
N_SPECIES = 2         # number of distinct species (integer, for dynamic lookup)
BOX_SIZE = 4.0        # cubic box length (Angstrom-like units)
DIM = 3               # spatial dimension
KT = 1.0              # temperature (float, not Array -- selects simpler apply_fn)
SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def key():
    return random.PRNGKey(SEED)


@pytest.fixture(scope="module")
def species_array():
    """Binary species: first N_SPECIES_0 atoms are type 0, rest type 1."""
    return jnp.array([0] * N_SPECIES_0 + [1] * N_SPECIES_1, dtype=jnp.int32)


@pytest.fixture(scope="module")
def initial_positions(key):
    """Positions uniformly distributed in a cubic periodic box."""
    return random.uniform(key, shape=(N_ATOMS, DIM)) * BOX_SIZE


@pytest.fixture(scope="module")
def displacement_fn():
    """Periodic displacement function for a cubic box."""
    displacement, _ = space.periodic(BOX_SIZE)
    return displacement


@pytest.fixture(scope="module")
def lj_energy_fn(displacement_fn):
    """
    LJ energy that accepts ``species`` as a keyword argument.

    ``smap.pair`` with ``species=N_SPECIES`` (an integer) produces a function
    whose signature is::

        energy_fn(R, species=species_array) -> scalar

    which matches exactly what ``swap_mc`` calls internally.
    """
    # Per-pair sigma/epsilon matrices indexed by (species_i, species_j).
    sigma = jnp.array([[1.0, 1.2], [1.2, 1.4]])
    epsilon = jnp.array([[1.0, 0.5], [0.5, 1.0]])

    return smap.pair(
        energy.lennard_jones,
        space.canonicalize_displacement_or_metric(displacement_fn),
        species=N_SPECIES,
        sigma=sigma,
        epsilon=epsilon,
        ignore_unused_parameters=True,
    )


@pytest.fixture(scope="module")
def swap_mc_fns(lj_energy_fn):
    """Return (init_fn, apply_fn) for the swap_mc simulator at fixed kT."""
    return simulate.swap_mc(lj_energy_fn, kT=KT)


@pytest.fixture(scope="module")
def initial_state(swap_mc_fns, key, initial_positions, species_array):
    """Initialised MCState."""
    init_fn, _ = swap_mc_fns
    return init_fn(key, initial_positions, species_array)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def species_counts(species):
    """Return (count_0, count_1) for a binary species array."""
    return int(jnp.sum(species == 0)), int(jnp.sum(species == 1))


# ---------------------------------------------------------------------------
# Tests — MCState structure
# ---------------------------------------------------------------------------


class TestMCStateStructure:
    """MCState must expose position, species, and key with correct shapes."""

    def test_has_position_field(self, initial_state):
        assert hasattr(initial_state, "position")

    def test_has_species_field(self, initial_state):
        assert hasattr(initial_state, "species")

    def test_has_key_field(self, initial_state):
        assert hasattr(initial_state, "key")

    def test_position_shape(self, initial_state):
        assert initial_state.position.shape == (N_ATOMS, DIM)

    def test_species_shape(self, initial_state):
        assert initial_state.species.shape == (N_ATOMS,)

    def test_initial_stoichiometry(self, initial_state):
        c0, c1 = species_counts(initial_state.species)
        assert c0 == N_SPECIES_0
        assert c1 == N_SPECIES_1


# ---------------------------------------------------------------------------
# Tests — stoichiometry preservation
# ---------------------------------------------------------------------------


class TestStoichiometryPreservation:
    """
    swap_mc may accept or reject a proposed species swap, but in either case
    the total count of each species must remain constant.
    """

    def test_single_step(self, swap_mc_fns, initial_state):
        _, apply_fn = swap_mc_fns
        state = apply_fn(initial_state)
        c0, c1 = species_counts(state.species)
        assert c0 == N_SPECIES_0, f"Type-0 count changed: {c0}"
        assert c1 == N_SPECIES_1, f"Type-1 count changed: {c1}"

    def test_fifty_steps(self, swap_mc_fns, initial_state):
        _, apply_fn = swap_mc_fns
        state = initial_state
        for step in range(50):
            state = apply_fn(state)
            c0, c1 = species_counts(state.species)
            assert c0 == N_SPECIES_0, f"Step {step}: type-0 count = {c0}"
            assert c1 == N_SPECIES_1, f"Step {step}: type-1 count = {c1}"

    def test_positions_unchanged(self, swap_mc_fns, initial_state):
        """swap_mc changes species labels only, not atomic positions."""
        _, apply_fn = swap_mc_fns
        state = apply_fn(initial_state)
        np.testing.assert_array_equal(
            np.array(state.position),
            np.array(initial_state.position),
        )


# ---------------------------------------------------------------------------
# Tests — energy finiteness
# ---------------------------------------------------------------------------


class TestEnergyFiniteness:
    """Energy must remain finite after successive MC steps."""

    def test_energy_finite_after_ten_steps(
        self, swap_mc_fns, initial_state, lj_energy_fn
    ):
        _, apply_fn = swap_mc_fns
        state = initial_state
        for i in range(10):
            state = apply_fn(state)
            e = lj_energy_fn(state.position, species=state.species)
            assert jnp.isfinite(e), f"Step {i}: energy = {e}"


# ---------------------------------------------------------------------------
# Tests — JIT compatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """apply_fn must be compilable with jax.jit."""

    def test_jit_compiles_and_runs(self, swap_mc_fns, initial_state):
        _, apply_fn = swap_mc_fns
        jit_apply = jax.jit(apply_fn)
        state = jit_apply(initial_state)
        assert state.position.shape == (N_ATOMS, DIM)
        assert state.species.shape == (N_ATOMS,)

    def test_jit_stoichiometry(self, swap_mc_fns, initial_state):
        _, apply_fn = swap_mc_fns
        jit_apply = jax.jit(apply_fn)
        state = initial_state
        for _ in range(5):
            state = jit_apply(state)
        c0, c1 = species_counts(state.species)
        assert c0 == N_SPECIES_0
        assert c1 == N_SPECIES_1

    def test_jit_energy_finite(self, swap_mc_fns, initial_state, lj_energy_fn):
        _, apply_fn = swap_mc_fns
        jit_apply = jax.jit(apply_fn)
        state = jit_apply(initial_state)
        e = lj_energy_fn(state.position, species=state.species)
        assert jnp.isfinite(e), f"JIT step energy: {e}"
