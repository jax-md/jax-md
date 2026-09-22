# Project Integration Roadmap — `alloy-release`

This file tracks outstanding integration tasks for the three custom feature
branches consolidated in `alloy-release`:

- `feat/eam_alloy` — EAM alloy potential loader
- `feat/swap_mc_simulator` — Species-swap Monte Carlo simulator
- `feat/jit_pair_correlation` — JIT-compatible multi-species pair correlation

---

## Status Overview

| Feature | Merged | Tests | Docs | Export (`__init__.py`) | Notebook |
|---------|--------|-------|------|-----------------------|----------|
| EAM Alloy (`eam_alloy`) | ✅ | ⚠️ (energy_test only) | ✅ | ✅ | ❌ |
| Swap MC (`swap_mc`, `MCState`) | ✅ | ✅ | ✅ | ⚠️ (check) | ❌ |
| JIT Pair Correlation | ✅ | ⚠️ (quantity_test only) | ✅ | ⚠️ (check) | ❌ |

---

## Open Tasks

### 1. Testing

- [ ] **EAM Alloy**: Write a dedicated `tests/test_eam_alloy.py` that loads
  `tests/data/Ni_Al.eam.alloy` via `load_lammps_eam_parameters_alloy` and
  asserts energy is finite and forces are non-zero for a small Ni-Al system.
- [x] **Swap MC**: `tests/test_swap_mc.py` added in this PR (stoichiometry,
  MCState shape, energy, JIT).
- [ ] **Pair Correlation**: Extend `tests/quantity_test.py` with a
  multi-species test using dynamic species (integer `species` argument to
  `pair_correlation`).

### 2. `__init__.py` Exports

- [ ] Confirm `swap_mc` and `MCState` are exported from `jax_md/__init__.py`
  (or `jax_md/simulate.py` re-exports). Add if missing.
- [ ] Confirm `pair_correlation` and `pair_correlation_neighbor_list` are
  exported. Add if missing.
- [ ] Confirm `eam_alloy`, `eam_alloy_neighbor_list`, and
  `load_lammps_eam_parameters_alloy` are exported from `jax_md.energy`.

### 3. Documentation

- [x] `docs/custom_alloy_extensions.md` — EAM alloy, Swap MC, pair
  correlation API guide (added in this PR).
- [ ] Update `docs/index.rst` (or `docs/conf.py`) to include the new doc page.
- [ ] Add Sphinx autodoc entries for `eam_alloy`, `swap_mc`, and
  `pair_correlation`.

### 4. Notebooks / Examples

- [ ] Create `notebooks/alloy_swap_mc_demo.ipynb` demonstrating a Ni-Al
  binary alloy simulation with EAM + swap MC + pair correlation analysis.
- [ ] Add a quick-start example to `README.md` showcasing `eam_alloy`.

### 5. CI / Upstream Compatibility

- [ ] Verify the three features pass `tests/energy_test.py`,
  `tests/simulate_test.py`, and `tests/quantity_test.py` without regressions
  on the upstream test suite.
- [ ] Add the new test file `tests/test_swap_mc.py` to the CI matrix in
  `.github/workflows/build.yml`.
- [ ] Check JAX version compatibility (the branch targets `jax>=0.5.0`).

### 6. Performance

- [ ] Benchmark `eam_alloy` vs. the single-species `eam` on a 1000-atom
  periodic system; confirm no more than 10% overhead from the multi-species
  masking logic.
- [ ] Profile `pair_correlation_neighbor_list` with dynamic species on a
  large system to ensure no Python-level retracing occurs.

### 7. Upstream PR Preparation

- [ ] Rebase `alloy-release` onto upstream `main` once the PR is reviewed.
- [ ] Add `CHANGELOG` or release notes entry.
- [ ] Consider splitting into three separate upstream PRs for easier review.

---

## Notes

- The `swap_mc` API differs from `hybrid_swap_mc`: it performs *only* species
  swaps (no MD moves). The energy function must accept `(R, species=species)`.
- For `kT` as a `float` (not a JAX array), `apply_fn` has the signature
  `apply_fn(state, **kwargs)`. For `kT` as a JAX `Array`,
  `apply_fn(state, kT, **kwargs)`.
- `eam_alloy` uses `load_lammps_eam_parameters_alloy` to read `.eam.alloy`
  (setfl format) files. The returned `(charge_fns, embedding_fns, pairwise_fns, cutoff)`
  tuple is passed directly to `eam_alloy(displacement, ...)`.
