# %% [markdown]
# # ASE Integration with Pretrained UMA
#
# Use pretrained UMA models with ASE's optimization, dynamics, and
# analysis tools via the UMACalculator.

# %%
import numpy as np

try:
  from ase import Atoms
  from ase.build import bulk, molecule, fcc111, add_adsorbate
  from ase.optimize import BFGS, FIRE
  from ase.constraints import FixAtoms, ExpCellFilter
  from ase.md.langevin import Langevin
  from ase import units
  HAS_ASE = True
except ImportError:
  HAS_ASE = False
  print("ASE not installed. Run: pip install ase")

if HAS_ASE:
  from jax_md._nn.uma.ase_calculator import UMACalculator

# %% [markdown]
# ## Setup calculator with pretrained weights

# %%
if HAS_ASE:
  calc = UMACalculator(
    checkpoint_path='uma-s-1p1',  # downloads from HuggingFace
    task_name='omat',
  )
  print(f"Calculator ready: {calc.config.num_layers} layers, "
        f"{calc.config.sphere_channels} channels")

# %% [markdown]
# ## Recipe 1: Relax a bulk crystal

# %%
if HAS_ASE:
  print("\n=== Bulk Cu relaxation ===")
  cu = bulk('Cu', 'fcc', a=3.7, cubic=True)  # slightly wrong lattice constant
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

  # Use omol task for molecules
  mol_calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omol')
  h2o.calc = mol_calc

  opt = BFGS(h2o, logfile=None)
  opt.run(fmax=0.1, steps=20)

  # Measure bond lengths
  d_OH1 = h2o.get_distance(0, 1)
  d_OH2 = h2o.get_distance(0, 2)
  angle = h2o.get_angle(1, 0, 2)
  print(f"O-H distances: {d_OH1:.3f}, {d_OH2:.3f} A")
  print(f"H-O-H angle: {angle:.1f} deg")

# %% [markdown]
# ## Recipe 3: Surface + adsorbate

# %%
if HAS_ASE:
  print("\n=== Cu(111) surface ===")
  slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)

  # Fix bottom layer
  z = slab.positions[:, 2]
  slab.set_constraint(FixAtoms(mask=(z < z.min() + 1.0)))

  slab.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='oc20')
  E_slab = slab.get_potential_energy()
  print(f"Slab ({len(slab)} atoms): E = {E_slab:.4f} eV")
  print(f"Max force on free atoms: "
        f"{np.max(np.abs(slab.get_forces()[~(z < z.min() + 1.0)])):.4f} eV/A")

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
# ## Recipe 5: NVT molecular dynamics

# %%
if HAS_ASE:
  print("\n=== Langevin MD (Cu, 300K) ===")
  cu_md = bulk('Cu', 'fcc', a=3.615, cubic=True)
  cu_md.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omat')

  dyn = Langevin(cu_md, timestep=1.0 * units.fs, temperature_K=300,
                 friction=0.01 / units.fs, logfile=None)

  for step in range(5):
    dyn.run(steps=1)
    E = cu_md.get_potential_energy()
    T = cu_md.get_kinetic_energy() / (1.5 * units.kB * len(cu_md))
    print(f"  Step {step}: E={E:.4f} eV, T={T:.0f} K")

# %% [markdown]
# ## Recipe 6: Charged molecules (omol)

# %%
if HAS_ASE:
  print("\n=== Charged system: OH- ===")
  oh = Atoms('OH', positions=[[0, 0, 0], [0, 0, 0.97]])
  oh.center(vacuum=5.0)
  oh.info['charge'] = -1   # hydroxide anion
  oh.info['spin'] = 1       # singlet

  oh.calc = UMACalculator(checkpoint_path='uma-s-1p1', task_name='omol')
  E = oh.get_potential_energy()
  F = oh.get_forces()
  print(f"OH- energy: {E:.4f} eV")
  print(f"Forces: {F}")

# %% [markdown]
# ## Recipe 7: Trajectory output

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
