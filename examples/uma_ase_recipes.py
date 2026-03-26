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
# # ASE Recipes with UMA-JAX
#
# A collection of common atomistic simulation recipes using the UMA
# calculator with ASE. Each recipe is self-contained.
#
# **Prerequisites:**
# ```bash
# pip install ase jax jaxlib flax torch
# ```

# %%
import numpy as np

# %% [markdown]
# ## Recipe 1: Single-Point Calculation

# %%
def recipe_single_point():
  """Compute energy and forces for a given structure."""
  from ase.build import bulk
  from jax_md._nn.uma.ase_calculator import UMACalculator
  from jax_md._nn.uma.model import UMAConfig

  atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)

  cfg = UMAConfig(
    sphere_channels=32, lmax=2, mmax=2, num_layers=1,
    hidden_channels=32, cutoff=5.0, edge_channels=32,
    num_distance_basis=64, use_dataset_embedding=False,
  )
  atoms.calc = UMACalculator(config=cfg, task_name='omat')

  energy = atoms.get_potential_energy()
  forces = atoms.get_forces()
  max_force = np.max(np.abs(forces))

  print(f"System: {atoms.get_chemical_formula()}")
  print(f"Energy: {energy:.6f} eV")
  print(f"Max force: {max_force:.6f} eV/A")
  print(f"Forces shape: {forces.shape}")
  return atoms

try:
  atoms = recipe_single_point()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 2: Geometry Optimization (BFGS)

# %%
def recipe_bfgs_relaxation():
  """Relax atomic positions with BFGS."""
  from ase.build import molecule
  from ase.optimize import BFGS
  from jax_md._nn.uma.ase_calculator import UMACalculator
  from jax_md._nn.uma.model import UMAConfig

  # Water molecule in a box
  atoms = molecule('H2O')
  atoms.center(vacuum=5.0)
  atoms.pbc = False

  # Perturb positions
  rng = np.random.default_rng(42)
  atoms.positions += rng.normal(scale=0.1, size=atoms.positions.shape)

  cfg = UMAConfig(
    sphere_channels=32, lmax=2, mmax=2, num_layers=1,
    hidden_channels=32, cutoff=6.0, edge_channels=32,
    num_distance_basis=64, use_dataset_embedding=False,
  )
  atoms.calc = UMACalculator(config=cfg, task_name='omol')

  print(f"Initial energy: {atoms.get_potential_energy():.6f} eV")

  opt = BFGS(atoms, logfile=None)
  opt.run(fmax=0.05, steps=30)

  print(f"Final energy:   {atoms.get_potential_energy():.6f} eV")
  print(f"Steps taken:    {opt.nsteps}")
  print(f"Max force:      {np.max(np.abs(atoms.get_forces())):.4f} eV/A")

try:
  recipe_bfgs_relaxation()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 3: Equation of State (EOS) Curve

# %%
def recipe_eos():
  """Compute energy vs volume curve for equation of state fitting."""
  from ase.build import bulk
  from jax_md._nn.uma.ase_calculator import UMACalculator
  from jax_md._nn.uma.model import UMAConfig

  cfg = UMAConfig(
    sphere_channels=32, lmax=2, mmax=2, num_layers=1,
    hidden_channels=32, cutoff=5.0, edge_channels=32,
    num_distance_basis=64, use_dataset_embedding=False,
  )

  # Scan lattice constants around equilibrium
  a0 = 3.615  # Cu FCC equilibrium
  strains = np.linspace(-0.05, 0.05, 11)
  volumes = []
  energies = []

  for strain in strains:
    a = a0 * (1 + strain)
    atoms = bulk('Cu', 'fcc', a=a)
    atoms.calc = UMACalculator(config=cfg, task_name='omat')

    E = atoms.get_potential_energy()
    V = atoms.get_volume()

    volumes.append(V)
    energies.append(E)
    print(f"  a={a:.3f} A, V={V:.2f} A^3, E={E:.6f} eV")

  # Fit Birch-Murnaghan EOS
  try:
    from ase.eos import EquationOfState
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0, e0, B = eos.fit()
    print(f"\nEOS fit: V0={v0:.2f} A^3, E0={e0:.6f} eV, B={B/1.602e-19*1e30/1e9:.1f} GPa")
  except Exception as e:
    print(f"\nEOS fit skipped: {e}")

try:
  recipe_eos()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 4: Surface Slab + Adsorbate

# %%
def recipe_surface_relaxation():
  """Create a surface slab and relax with fixed bottom layers."""
  from ase.build import fcc111, add_adsorbate
  from ase.constraints import FixAtoms
  from ase.optimize import BFGS
  from jax_md._nn.uma.ase_calculator import UMACalculator
  from jax_md._nn.uma.model import UMAConfig

  # Build 3-layer Cu(111) slab with vacuum
  slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)

  # Fix bottom layer
  z_positions = slab.positions[:, 2]
  bottom_layer = z_positions < z_positions.min() + 1.0
  slab.set_constraint(FixAtoms(mask=bottom_layer))

  # Add CO adsorbate on top
  add_adsorbate(slab, 'C', height=1.8, position='ontop')

  cfg = UMAConfig(
    sphere_channels=32, lmax=2, mmax=2, num_layers=1,
    hidden_channels=32, cutoff=5.0, edge_channels=32,
    num_distance_basis=64, use_dataset_embedding=False,
  )
  slab.calc = UMACalculator(config=cfg, task_name='oc20')

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
# ## Recipe 5: Molecular Dynamics with ASE

# %%
def recipe_ase_md():
  """Run NVT molecular dynamics using ASE's Langevin thermostat."""
  from ase.build import bulk
  from ase.md.langevin import Langevin
  from ase import units
  from jax_md._nn.uma.ase_calculator import UMACalculator
  from jax_md._nn.uma.model import UMAConfig

  atoms = bulk('Cu', 'fcc', a=3.615, cubic=True).repeat((2, 2, 2))

  cfg = UMAConfig(
    sphere_channels=32, lmax=2, mmax=2, num_layers=1,
    hidden_channels=32, cutoff=5.0, edge_channels=32,
    num_distance_basis=64, use_dataset_embedding=False,
  )
  atoms.calc = UMACalculator(config=cfg, task_name='omat')

  # Langevin NVT at 300K
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
# ## Recipe 6: Elastic Constants

# %%
def recipe_elastic():
  """Compute elastic constants via strain-stress relation."""
  from ase.build import bulk
  from jax_md._nn.uma.ase_calculator import UMACalculator
  from jax_md._nn.uma.model import UMAConfig

  cfg = UMAConfig(
    sphere_channels=32, lmax=2, mmax=2, num_layers=1,
    hidden_channels=32, cutoff=5.0, edge_channels=32,
    num_distance_basis=64, use_dataset_embedding=False,
  )

  # Reference structure
  atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
  atoms.calc = UMACalculator(config=cfg, task_name='omat')
  E0 = atoms.get_potential_energy()
  V0 = atoms.get_volume()

  # Apply small strains and compute energy
  delta = 0.005
  strains = [-2*delta, -delta, 0, delta, 2*delta]

  print("Strain-energy curve (volumetric):")
  for eps in strains:
    strained = atoms.copy()
    cell = strained.get_cell()
    strained.set_cell(cell * (1 + eps), scale_atoms=True)
    strained.calc = UMACalculator(config=cfg, task_name='omat')
    E = strained.get_potential_energy()
    print(f"  eps={eps:+.4f}: E={E:.6f} eV (dE={E-E0:+.6f})")

try:
  recipe_elastic()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 7: Batch Screening
#
# Evaluate multiple structures efficiently by reusing the calculator.

# %%
def recipe_batch_screening():
  """Screen multiple structures for their energies."""
  from ase.build import bulk
  from jax_md._nn.uma.ase_calculator import UMACalculator
  from jax_md._nn.uma.model import UMAConfig

  cfg = UMAConfig(
    sphere_channels=32, lmax=2, mmax=2, num_layers=1,
    hidden_channels=32, cutoff=5.0, edge_channels=32,
    num_distance_basis=64, use_dataset_embedding=False,
  )
  calc = UMACalculator(config=cfg, task_name='omat')

  # Screen FCC metals at experimental lattice constants
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

  # Rank by energy
  sorted_metals = sorted(results.items(), key=lambda x: x[1])
  print(f"\nMost stable: {sorted_metals[0][0]} ({sorted_metals[0][1]:.4f} eV/atom)")

try:
  recipe_batch_screening()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 8: Charged/Spin Systems (omol)

# %%
def recipe_charged_system():
  """Handle charged molecules and spin states."""
  from ase import Atoms
  from jax_md._nn.uma.ase_calculator import UMACalculator
  from jax_md._nn.uma.model import UMAConfig

  cfg = UMAConfig(
    sphere_channels=32, lmax=2, mmax=2, num_layers=1,
    hidden_channels=32, cutoff=6.0, edge_channels=32,
    num_distance_basis=64,
    dataset_list=['oc20', 'omol', 'omat', 'odac', 'omc'],
    use_dataset_embedding=True,
  )

  # O2 molecule: neutral triplet
  o2 = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.21]])
  o2.center(vacuum=5.0)
  o2.info['charge'] = 0
  o2.info['spin'] = 3  # triplet: spin multiplicity = 2S+1 = 3

  o2.calc = UMACalculator(config=cfg, task_name='omol')
  E_neutral = o2.get_potential_energy()
  print(f"O2 (neutral, triplet): E = {E_neutral:.4f} eV")

  # O2- anion: doublet
  o2_anion = o2.copy()
  o2_anion.info['charge'] = -1
  o2_anion.info['spin'] = 2  # doublet
  o2_anion.calc = UMACalculator(config=cfg, task_name='omol')
  E_anion = o2_anion.get_potential_energy()
  print(f"O2- (anion, doublet):  E = {E_anion:.4f} eV")
  print(f"Electron affinity:     {E_neutral - E_anion:.4f} eV")

try:
  recipe_charged_system()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Recipe 9: Trajectory I/O

# %%
def recipe_trajectory():
  """Save and load optimization trajectories."""
  from ase.build import bulk
  from ase.optimize import BFGS
  from ase.io import read, write
  from ase.io.trajectory import Trajectory
  from jax_md._nn.uma.ase_calculator import UMACalculator
  from jax_md._nn.uma.model import UMAConfig

  cfg = UMAConfig(
    sphere_channels=32, lmax=2, mmax=2, num_layers=1,
    hidden_channels=32, cutoff=5.0, edge_channels=32,
    num_distance_basis=64, use_dataset_embedding=False,
  )

  atoms = bulk('Si', 'diamond', a=5.43)
  rng = np.random.default_rng(42)
  atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)

  atoms.calc = UMACalculator(config=cfg, task_name='omat')

  # Save trajectory
  traj_file = '/tmp/uma_relax.traj'
  with Trajectory(traj_file, 'w', atoms) as traj:
    opt = BFGS(atoms, logfile=None)
    opt.attach(traj.write, interval=1)
    opt.run(fmax=0.1, steps=10)

  # Read trajectory
  frames = read(traj_file, index=':')
  print(f"Trajectory: {len(frames)} frames saved to {traj_file}")
  for i, frame in enumerate(frames):
    print(f"  Frame {i}: E={frame.get_potential_energy():.4f} eV")

  # Export to XYZ for visualization
  xyz_file = '/tmp/uma_relax.xyz'
  write(xyz_file, frames)
  print(f"XYZ written to {xyz_file}")

try:
  recipe_trajectory()
except ImportError as e:
  print(f"Skipping: {e}")

# %% [markdown]
# ## Using Pretrained Models
#
# All recipes above use random weights for demonstration. To use
# pretrained UMA weights from FairChem:
#
# ```python
# calc = UMACalculator(
#     checkpoint_path='path/to/uma_sm_conserve.pt',
#     task_name='omat',  # or 'omol', 'oc20', etc.
# )
# atoms.calc = calc
# ```
