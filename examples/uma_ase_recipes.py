# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Advanced ASE Recipes with Pretrained UMA
#
# Advanced atomistic simulation recipes using a pretrained UMA model
# with ASE. For basic getting-started workflows (relaxation, EOS,
# molecule optimization, trajectory I/O), see `uma_pretrained_ase.py`.
#
# **Prerequisites:**
# ```bash
# pip install ase jax jaxlib flax torch
# ```

# %%
import numpy as np

# %% [markdown]
# ## Recipe 1: Surface Slab + Adsorbate

# %%
def recipe_surface_relaxation():
  """Create a surface slab and relax with fixed bottom layers."""
  from ase.build import fcc111, add_adsorbate
  from ase.constraints import FixAtoms
  from ase.optimize import BFGS
  from jax_md._nn.uma.ase_calculator import UMACalculator

  slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)

  z_positions = slab.positions[:, 2]
  bottom_layer = z_positions < z_positions.min() + 1.0
  slab.set_constraint(FixAtoms(mask=bottom_layer))

  add_adsorbate(slab, 'C', height=1.8, position='ontop')

  slab.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='oc20')

  print(f"Slab: {slab.get_chemical_formula()}, {len(slab)} atoms")
  print(f"Fixed atoms: {sum(bottom_layer)}")
  print(f"Initial energy: {slab.get_potential_energy():.4f} eV")

  opt = BFGS(slab, logfile=None)
  opt.run(fmax=0.1, steps=20)

  print(f"Final energy:   {slab.get_potential_energy():.4f} eV")
  print(f"Steps: {opt.nsteps}")

try:
  recipe_surface_relaxation()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 2: Molecular Dynamics with ASE

# %%
def recipe_ase_md():
  """Run NVT molecular dynamics using ASE's Langevin thermostat."""
  from ase.build import bulk
  from ase.md.langevin import Langevin
  from ase import units
  from jax_md._nn.uma.ase_calculator import UMACalculator

  atoms = bulk('Cu', 'fcc', a=3.615, cubic=True).repeat((2, 2, 2))
  atoms.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omat')

  dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.01 / units.fs,
    logfile=None,
  )

  print(f"System: {atoms.get_chemical_formula()}, {len(atoms)} atoms")
  energies = []

  for step in range(10):
    dyn.run(steps=1)
    E = atoms.get_potential_energy()
    T = atoms.get_kinetic_energy() / (1.5 * units.kB * len(atoms))
    energies.append(E)
    if step % 5 == 0:
      print(f"  Step {step}: E={E:.4f} eV, T={T:.0f} K")

  print(f"Done: {len(energies)} steps")

try:
  recipe_ase_md()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 3: Elastic Constants

# %%
def recipe_elastic():
  """Compute elastic constants via strain-stress relation."""
  from ase.build import bulk
  from jax_md._nn.uma.ase_calculator import UMACalculator

  calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omat')

  atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
  atoms.calc = calc
  E0 = atoms.get_potential_energy()

  delta = 0.005
  strains = [-2*delta, -delta, 0, delta, 2*delta]

  print("Strain-energy curve (volumetric):")
  for eps in strains:
    strained = atoms.copy()
    cell = strained.get_cell()
    strained.set_cell(cell * (1 + eps), scale_atoms=True)
    strained.calc = calc
    E = strained.get_potential_energy()
    print(f"  eps={eps:+.4f}: E={E:.6f} eV (dE={E-E0:+.6f})")

try:
  recipe_elastic()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 4: Batch Screening
#
# Evaluate multiple structures efficiently by reusing the calculator.

# %%
def recipe_batch_screening():
  """Screen multiple structures for their energies."""
  from ase.build import bulk
  from jax_md._nn.uma.ase_calculator import UMACalculator

  calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omat')

  metals = {
    'Cu': 3.615,
    'Ag': 4.085,
    'Au': 4.078,
    'Al': 4.050,
    'Ni': 3.524,
    'Pt': 3.924,
  }

  print(f"{'Metal':>5} {'a (A)':>8} {'E/atom (eV)':>12}")
  print("-" * 28)

  results = {}
  for element, a in metals.items():
    atoms = bulk(element, 'fcc', a=a)
    atoms.calc = calc
    E = atoms.get_potential_energy()
    e_per_atom = E / len(atoms)
    results[element] = e_per_atom
    print(f"{element:>5} {a:>8.3f} {e_per_atom:>12.6f}")

  sorted_metals = sorted(results.items(), key=lambda x: x[1])
  print(f"\nMost stable: {sorted_metals[0][0]} ({sorted_metals[0][1]:.4f} eV/atom)")

try:
  recipe_batch_screening()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 5: Charged/Spin Systems (omol)

# %%
def recipe_charged_system():
  """Handle charged molecules and spin states."""
  from ase import Atoms
  from jax_md._nn.uma.ase_calculator import UMACalculator

  # O2 molecule: neutral triplet
  o2 = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.21]])
  o2.center(vacuum=5.0)
  o2.info['charge'] = 0
  o2.info['spin'] = 3  # triplet: spin multiplicity = 2S+1 = 3

  o2.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omol')
  E_neutral = o2.get_potential_energy()
  print(f"O2 (neutral, triplet): E = {E_neutral:.4f} eV")

  # O2- anion: doublet
  o2_anion = o2.copy()
  o2_anion.info['charge'] = -1
  o2_anion.info['spin'] = 2  # doublet
  o2_anion.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omol')
  E_anion = o2_anion.get_potential_energy()
  print(f"O2- (anion, doublet):  E = {E_anion:.4f} eV")
  print(f"Electron affinity:     {E_neutral - E_anion:.4f} eV")

try:
  recipe_charged_system()
except ImportError as e:
  print(f"Skipping: {e}")
