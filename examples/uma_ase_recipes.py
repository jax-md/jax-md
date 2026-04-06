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
# # ASE Recipes with Pretrained UMA
#
# A comprehensive cookbook of atomistic simulation recipes using a
# pretrained UMA model with ASE. All recipes use the `UMACalculator`
# with `checkpoint_path='uma-s-1p1'` (downloaded from HuggingFace).
#
# **Prerequisites:**
# ```bash
# pip install ase jax jaxlib flax torch
# ```

# %%
import numpy as np

try:
  from ase import Atoms
  from ase.build import bulk, molecule, fcc111, add_adsorbate
  from ase.optimize import BFGS
  from ase.constraints import FixAtoms
  from ase.md.langevin import Langevin
  from ase import units
  HAS_ASE = True
except ImportError:
  HAS_ASE = False
  print("ASE not installed. Run: pip install ase")

if HAS_ASE:
  from jax_md._nn.uma.ase_calculator import UMACalculator

# %% [markdown]
# ## Setup: create calculator with pretrained weights

# %%
if HAS_ASE:
  calc = UMACalculator(
    checkpoint_path='uma-s-1p1',
    task_name='omat',
  )
  print(f"Calculator ready: {calc.config.num_layers} layers, "
        f"{calc.config.sphere_channels} channels")

# %% [markdown]
# ## Recipe 1: Bulk crystal relaxation

# %%
if HAS_ASE:
  print("\n=== Bulk Cu relaxation ===")
  cu = bulk('Cu', 'fcc', a=3.7, cubic=True)
  rng = np.random.default_rng(42)
  cu.positions += rng.normal(scale=0.05, size=cu.positions.shape)
  cu.calc = calc

  opt = BFGS(cu, logfile=None)
  opt.run(fmax=0.1, steps=20)
  print(f"Energy: {cu.get_potential_energy():.4f} eV")
  print(f"Max force: {np.max(np.abs(cu.get_forces())):.4f} eV/A")
  print(f"Steps: {opt.nsteps}")

# %% [markdown]
# ## Recipe 2: Molecular geometry optimization

# %%
if HAS_ASE:
  print("\n=== H2O geometry optimization ===")
  h2o = molecule('H2O')
  h2o.center(vacuum=5.0)
  h2o.positions += rng.normal(scale=0.1, size=h2o.positions.shape)

  mol_calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omol')
  h2o.calc = mol_calc

  opt = BFGS(h2o, logfile=None)
  opt.run(fmax=0.1, steps=20)

  d_OH1 = h2o.get_distance(0, 1)
  d_OH2 = h2o.get_distance(0, 2)
  angle = h2o.get_angle(1, 0, 2)
  print(f"O-H distances: {d_OH1:.3f}, {d_OH2:.3f} A")
  print(f"H-O-H angle: {angle:.1f} deg")

# %% [markdown]
# ## Recipe 3: Surface slab + adsorbate relaxation

# %%
if HAS_ASE:
  print("\n=== Cu(111) + C adsorbate ===")
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

# %% [markdown]
# ## Recipe 4: Equation of state

# %%
if HAS_ASE:
  print("\n=== Cu EOS ===")
  strains = np.linspace(-0.03, 0.03, 7)
  a0 = 3.615

  volumes, energies = [], []
  for eps in strains:
    a = a0 * (1 + eps)
    atoms = bulk('Cu', 'fcc', a=a)
    atoms.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omat')
    E = atoms.get_potential_energy()
    V = atoms.get_volume()
    volumes.append(V)
    energies.append(E)
    print(f"  a={a:.3f} A, V={V:.2f} A^3, E={E:.4f} eV")

# %% [markdown]
# ## Recipe 5: Langevin MD

# %%
if HAS_ASE:
  print("\n=== Langevin MD (Cu 2x2x2, 300K) ===")
  cu_md = bulk('Cu', 'fcc', a=3.615, cubic=True).repeat((2, 2, 2))
  cu_md.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omat')

  dyn = Langevin(
    cu_md,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.01 / units.fs,
    logfile=None,
  )

  print(f"System: {cu_md.get_chemical_formula()}, {len(cu_md)} atoms")
  md_energies = []

  for step in range(10):
    dyn.run(steps=1)
    E = cu_md.get_potential_energy()
    T = cu_md.get_kinetic_energy() / (1.5 * units.kB * len(cu_md))
    md_energies.append(E)
    if step % 5 == 0:
      print(f"  Step {step}: E={E:.4f} eV, T={T:.0f} K")

  print(f"Done: {len(md_energies)} steps")

# %% [markdown]
# ## Recipe 6: Elastic constants

# %%
if HAS_ASE:
  print("\n=== Elastic constants (Cu) ===")
  elastic_calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omat')

  atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
  atoms.calc = elastic_calc
  E0 = atoms.get_potential_energy()

  delta = 0.005
  strain_values = [-2*delta, -delta, 0, delta, 2*delta]

  print("Strain-energy curve (volumetric):")
  for eps in strain_values:
    strained = atoms.copy()
    cell = strained.get_cell()
    strained.set_cell(cell * (1 + eps), scale_atoms=True)
    strained.calc = elastic_calc
    E = strained.get_potential_energy()
    print(f"  eps={eps:+.4f}: E={E:.6f} eV (dE={E-E0:+.6f})")

# %% [markdown]
# ## Recipe 7: Batch screening

# %%
if HAS_ASE:
  print("\n=== FCC metals screening ===")
  screen_calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omat')

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
    atoms.calc = screen_calc
    E = atoms.get_potential_energy()
    e_per_atom = E / len(atoms)
    results[element] = e_per_atom
    print(f"{element:>5} {a:>8.3f} {e_per_atom:>12.6f}")

  sorted_metals = sorted(results.items(), key=lambda x: x[1])
  print(f"\nMost stable: {sorted_metals[0][0]} ({sorted_metals[0][1]:.4f} eV/atom)")

# %% [markdown]
# ## Recipe 8: Charged and spin systems (omol)

# %%
if HAS_ASE:
  print("\n=== Charged system: OH- ===")
  oh = Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.97]])
  oh.center(vacuum=5.0)
  oh.info['charge'] = -1
  oh.info['spin'] = 1

  oh.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omol')
  E_oh = oh.get_potential_energy()
  F_oh = oh.get_forces()
  print(f"OH- energy: {E_oh:.4f} eV")
  print(f"Forces: {F_oh}")

  print("\n=== O2 electron affinity ===")
  o2 = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.21]])
  o2.center(vacuum=5.0)
  o2.info['charge'] = 0
  o2.info['spin'] = 3  # triplet

  o2.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omol')
  E_neutral = o2.get_potential_energy()
  print(f"O2 (neutral, triplet): E = {E_neutral:.4f} eV")

  o2_anion = o2.copy()
  o2_anion.info['charge'] = -1
  o2_anion.info['spin'] = 2  # doublet
  o2_anion.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omol')
  E_anion = o2_anion.get_potential_energy()
  print(f"O2- (anion, doublet):  E = {E_anion:.4f} eV")
  print(f"Electron affinity:     {E_neutral - E_anion:.4f} eV")

# %% [markdown]
# ## Recipe 9: Trajectory I/O

# %%
if HAS_ASE:
  from ase.io import write
  from ase.io.trajectory import Trajectory

  print("\n=== Optimization trajectory ===")
  si = bulk('Si', 'diamond', a=5.5)
  rng2 = np.random.default_rng(123)
  si.positions += rng2.normal(scale=0.05, size=si.positions.shape)
  si.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omat')

  traj_file = '/tmp/uma_si_relax.traj'
  with Trajectory(traj_file, 'w', si) as traj:
    opt = BFGS(si, logfile=None)
    opt.attach(traj.write, interval=1)
    opt.run(fmax=0.1, steps=10)

  frames = Trajectory(traj_file)
  print(f"Saved {len(frames)} frames to {traj_file}")

  xyz_file = '/tmp/uma_si_relax.xyz'
  write(xyz_file, list(frames))
  print(f"Exported to {xyz_file}")
