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
# # UMA Structure Relaxation with JAX-MD
#
# This example demonstrates structure relaxation using the UMA (Universal
# Model for Atoms) neural network potential with JAX-MD's FIRE minimizer.
#
# Two approaches are shown:
# 1. **JAX-MD native**: Using `minimize.fire_descent` directly (fully JIT-compiled)
# 2. **ASE integration**: Using ASE's optimizers with a JAX-backed calculator

# %% [markdown]
# ## 1. JAX-MD Native Relaxation (FIRE)
#
# This approach uses JAX-MD's built-in FIRE optimizer. The entire optimization
# loop is JIT-compilable for maximum performance.

# %%
import jax
import jax.numpy as jnp
from jax import jit, random
import numpy as np

from jax_md import space, energy, minimize

from jax_md._nn.uma.model import UMAConfig

# %%
# === System Setup ===
# Create a simple periodic silicon system with slightly perturbed positions

# Diamond cubic silicon lattice constant
a = 5.43  # Angstroms
box_size = a  # Cubic box

# Silicon basis positions (diamond cubic, 8 atoms)
basis = np.array([
  [0.00, 0.00, 0.00],
  [0.50, 0.50, 0.00],
  [0.50, 0.00, 0.50],
  [0.00, 0.50, 0.50],
  [0.25, 0.25, 0.25],
  [0.75, 0.75, 0.25],
  [0.75, 0.25, 0.75],
  [0.25, 0.75, 0.75],
]) * a

# Add random perturbation (to give the optimizer something to do)
key = random.PRNGKey(42)
perturbation = random.normal(key, basis.shape) * 0.1  # 0.1 Angstrom noise
positions = jnp.array(basis + perturbation, dtype=jnp.float32)

# Atomic numbers (all silicon)
atoms = jnp.array([14] * 8, dtype=jnp.int32)

print(f"System: {len(atoms)} Si atoms in {box_size:.2f} A box")
print(f"Initial max displacement from ideal: {np.max(np.abs(perturbation)):.3f} A")

# %%
# === Build UMA Energy Function ===
# Use a small config for demonstration (use default_config() for production)

displacement_fn, shift_fn = space.periodic(box_size)

cfg = UMAConfig(
  sphere_channels=32,  # Small for demo; use 128 for production
  lmax=2,
  mmax=2,
  num_layers=1,        # Small for demo; use 4+ for production
  hidden_channels=32,
  cutoff=5.0,
  edge_channels=32,
  num_distance_basis=64,
  use_dataset_embedding=False,
)

# Create the (neighbor_fn, init_fn, energy_fn) triple
neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
  displacement_fn, box_size, cfg=cfg, atoms=atoms,
  # For pretrained model, add: checkpoint_path='path/to/uma.pt'
)

# %%
# === Initialize ===
# Allocate neighbor list and initialize parameters

nbrs = neighbor_fn.allocate(positions)
params = init_fn(key, positions, nbrs)

# Create a closure that includes params and neighbor list
def uma_energy(R, **kwargs):
  """Energy function for FIRE optimizer (params/nbrs captured in closure)."""
  nbrs_updated = nbrs.update(R)
  return energy_fn(params, R, nbrs_updated)

initial_energy = uma_energy(positions)
print(f"Initial energy: {initial_energy:.6f}")

# %%
# === FIRE Relaxation ===

fire_init, fire_apply = minimize.fire_descent(
  uma_energy, shift_fn,
  dt_start=0.1,
  dt_max=0.4,
)
fire_apply = jit(fire_apply)

# Initialize FIRE state
fire_state = fire_init(positions)

# Run FIRE minimization
energies = []
max_forces = []
max_steps = 100

print(f"\n{'Step':>5} {'Energy':>14} {'Max Force':>12}")
print("-" * 35)

for step in range(max_steps):
  fire_state = fire_apply(fire_state)

  E = float(uma_energy(fire_state.position))
  F = -jax.grad(uma_energy)(fire_state.position)
  fmax = float(jnp.max(jnp.abs(F)))

  energies.append(E)
  max_forces.append(fmax)

  if step % 10 == 0 or fmax < 0.05:
    print(f"{step:5d} {E:14.6f} {fmax:12.6f}")

  if fmax < 0.01:
    print(f"\nConverged at step {step}! (fmax = {fmax:.6f})")
    break

final_positions = fire_state.position
displacement_from_initial = jnp.linalg.norm(final_positions - positions, axis=-1)
print(f"\nMax atomic displacement: {float(jnp.max(displacement_from_initial)):.4f} A")

# %%
# === Plot Energy Convergence ===
try:
  import matplotlib.pyplot as plt

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

  ax1.plot(energies, linewidth=2)
  ax1.set_xlabel('FIRE Step')
  ax1.set_ylabel('Energy')
  ax1.set_title('Energy Convergence')
  ax1.grid(True)

  ax2.semilogy(max_forces, linewidth=2)
  ax2.axhline(y=0.01, color='r', linestyle='--', label='fmax threshold')
  ax2.set_xlabel('FIRE Step')
  ax2.set_ylabel('Max |Force| Component')
  ax2.set_title('Force Convergence')
  ax2.legend()
  ax2.grid(True)

  plt.tight_layout()
  plt.savefig('uma_relaxation_convergence.png', dpi=150)
  plt.show()
  print("Plot saved to uma_relaxation_convergence.png")
except ImportError:
  print("matplotlib not available, skipping plots")

# %% [markdown]
# ## 2. ASE Integration
#
# For users who prefer ASE's ecosystem (BFGS, constraints, trajectory I/O),
# we provide `UMACalculator` — an ASE-compatible calculator backed by JAX.

# %%
try:
  from ase import Atoms
  from ase.build import bulk
  from ase.optimize import BFGS, FIRE as ASE_FIRE
  from ase.constraints import ExpCellFilter
  from jax_md._nn.uma.ase_calculator import UMACalculator

  # === Create ASE Atoms ===
  si = bulk('Si', 'diamond', a=5.43)
  si = si.repeat((2, 2, 2))  # 2x2x2 supercell = 64 atoms

  # Perturb positions
  rng = np.random.default_rng(42)
  si.positions += rng.normal(scale=0.05, size=si.positions.shape)

  print(f"ASE system: {len(si)} atoms, PBC={si.pbc}")

  # === Attach UMA Calculator ===
  calc = UMACalculator(
    config=cfg,
    task_name='omat',
    # For pretrained: checkpoint_path='path/to/uma.pt'
  )
  si.calc = calc

  # === Relaxation with BFGS ===
  print("\nRunning BFGS relaxation...")
  opt = BFGS(si, logfile='-')
  opt.run(fmax=0.05, steps=50)

  print(f"\nFinal energy: {si.get_potential_energy():.6f} eV")
  print(f"Final max force: {np.max(np.abs(si.get_forces())):.6f} eV/A")

  # === Cell Relaxation (Volume + Positions) ===
  # Wrap with ExpCellFilter to relax cell parameters too
  print("\n--- Cell + Position Relaxation ---")
  si2 = bulk('Si', 'diamond', a=5.50)  # Wrong lattice constant
  si2.calc = UMACalculator(config=cfg, task_name='omat')

  ecf = ExpCellFilter(si2)
  opt2 = BFGS(ecf, logfile='-')
  opt2.run(fmax=0.05, steps=50)

  print(f"\nRelaxed lattice constant: {si2.cell.lengths()[0]:.3f} A")

except ImportError as e:
  print(f"ASE not available ({e}), skipping ASE examples.")
  print("Install with: pip install ase")

# %% [markdown]
# ## 3. Loading Pretrained Checkpoints
#
# To use a pretrained UMA model from FairChem:
#
# ```python
# from jax_md._nn.uma.ase_calculator import UMACalculator
#
# # Automatically loads config + weights from checkpoint
# calc = UMACalculator(checkpoint_path='uma_sm_conserve.pt', task_name='omat')
#
# atoms.calc = calc
# energy = atoms.get_potential_energy()
# forces = atoms.get_forces()
# ```
#
# For JAX-MD native usage:
#
# ```python
# from jax_md import space, energy
#
# displacement_fn, shift_fn = space.periodic_general(cell)
# neighbor_fn, init_fn, energy_fn = energy.uma_neighbor_list(
#     displacement_fn, cell,
#     checkpoint_path='uma_sm_conserve.pt',
#     atoms=atomic_numbers,
# )
#
# nbrs = neighbor_fn.allocate(positions)
# params = init_fn(key, positions, nbrs)  # Uses pretrained weights
# E = energy_fn(params, positions, nbrs)
# forces = -jax.grad(energy_fn, argnums=1)(params, positions, nbrs)
# ```

# %% [markdown]
# ## Tips
#
# - **Performance**: JAX-MD native is faster (fully JIT-compiled loop).
#   ASE calls back into Python each step.
# - **Periodic boundaries**: Both approaches handle PBC correctly.
#   JAX-MD via `space.periodic()`, ASE via `atoms.pbc`.
# - **Pretrained models**: Pass `checkpoint_path` to load FairChem weights.
# - **GPU**: JAX automatically uses GPU if available. No code changes needed.
# - **Float64**: For tight energy conservation in MD, set
#   `jax.config.update('jax_enable_x64', True)` before any JAX calls.
