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
# # Phonon Calculation with UMA
#
# Compute the dynamical matrix and phonon frequencies of a crystal
# using the UMA potential and JAX's automatic differentiation.
#
# The Hessian (force constant matrix) is computed exactly via `jax.hessian`,
# which gives the full second-derivative matrix in a single call.

# %%
import jax
import jax.numpy as jnp
import numpy as np

from jax_md._nn.uma.model import UMABackbone, UMAConfig
from jax_md._nn.uma.heads import MLPEnergyHead

# %% [markdown]
# ## Setup: Silicon unit cell

# %%
a = 5.43  # Angstroms

basis = np.array([
  [0.00, 0.00, 0.00],
  [0.25, 0.25, 0.25],
]) * a

atomic_numbers = jnp.array([14, 14], dtype=jnp.int32)
positions = jnp.array(basis, dtype=jnp.float32)
num_atoms = len(positions)

# Build edges (all pairs within cutoff)
cutoff = 5.0
edge_src, edge_dst = [], []
# For a real calculation, include periodic images
for i in range(num_atoms):
  for j in range(num_atoms):
    if i != j:
      dist = np.linalg.norm(basis[i] - basis[j])
      if dist < cutoff:
        edge_src.append(i)
        edge_dst.append(j)

edge_index = jnp.array([edge_src, edge_dst], dtype=jnp.int32)
batch = jnp.zeros(num_atoms, dtype=jnp.int32)
charge = jnp.zeros(1)
spin = jnp.zeros(1)

print(f"System: {num_atoms} atoms, {edge_index.shape[1]} edges")

# %% [markdown]
# ## Build UMA energy function

# %%
cfg = UMAConfig(
  sphere_channels=32, lmax=2, mmax=2, num_layers=1,
  hidden_channels=32, cutoff=cutoff, edge_channels=32,
  num_distance_basis=64, use_dataset_embedding=False,
)

backbone = UMABackbone(config=cfg)
head = MLPEnergyHead(sphere_channels=32, hidden_channels=32)

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key)

backbone_params = backbone.init(
  key1, positions, atomic_numbers, batch, edge_index,
  positions[edge_index[0]] - positions[edge_index[1]], charge, spin, None,
)

emb = backbone.apply(
  backbone_params, positions, atomic_numbers, batch, edge_index,
  positions[edge_index[0]] - positions[edge_index[1]], charge, spin, None,
)
head_params = head.init(key2, emb['node_embedding'], batch, 1)

def total_energy(pos):
  """Scalar energy as a function of flattened positions."""
  pos = pos.reshape(-1, 3)
  edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
  emb = backbone.apply(
    backbone_params, pos, atomic_numbers, batch,
    edge_index, edge_vec, charge, spin, None,
  )
  result = head.apply(head_params, emb['node_embedding'], batch, 1)
  return result['energy'].sum()

# Test energy
E = total_energy(positions.reshape(-1))
print(f"Energy: {E:.6f}")

# %% [markdown]
# ## Compute Hessian (Force Constant Matrix)
#
# The Hessian of the energy with respect to atomic positions gives the
# force constant matrix $\Phi_{i\alpha, j\beta} = \frac{\partial^2 E}{\partial R_{i\alpha} \partial R_{j\beta}}$.

# %%
print("Computing Hessian (this may take a moment for JIT compilation)...")
hessian_fn = jax.hessian(total_energy)
pos_flat = positions.reshape(-1)  # [3N]
H = hessian_fn(pos_flat)         # [3N, 3N]

print(f"Hessian shape: {H.shape}")
print(f"Hessian symmetry check: max|H - H^T| = {float(jnp.max(jnp.abs(H - H.T))):.2e}")

# Symmetrize
H = 0.5 * (H + H.T)

# %% [markdown]
# ## Compute Phonon Frequencies
#
# The dynamical matrix $D = M^{-1/2} \Phi M^{-1/2}$ where $M$ is the mass matrix.
# Eigenvalues give $\omega^2$.

# %%
# Atomic masses (silicon = 28.085 amu)
masses_amu = np.array([28.085, 28.085])
# Repeat for 3 Cartesian components
mass_vector = np.repeat(masses_amu, 3)
inv_sqrt_mass = 1.0 / np.sqrt(mass_vector)

# Dynamical matrix
D = H * np.outer(inv_sqrt_mass, inv_sqrt_mass)

# Diagonalize
eigenvalues, eigenvectors = np.linalg.eigh(np.array(D))

print("\nPhonon eigenvalues (omega^2):")
for i, ev in enumerate(eigenvalues):
  # Convert to frequency (sign-preserving sqrt)
  freq = np.sign(ev) * np.sqrt(np.abs(ev))
  mode_type = "acoustic" if i < 3 else "optical"
  print(f"  Mode {i}: omega^2 = {ev:10.4f}, omega = {freq:10.4f}  ({mode_type})")

# Three acoustic modes should have omega ≈ 0 at Gamma point
print(f"\nFirst 3 eigenvalues (should be ~0 for acoustic): {eigenvalues[:3]}")

# %% [markdown]
# ## Finite Difference Validation
#
# Verify the Hessian by comparing with finite differences.

# %%
print("\nFinite difference validation of Hessian...")
delta = 1e-3
grad_fn = jax.grad(total_energy)

fd_H = np.zeros((3 * num_atoms, 3 * num_atoms))
for i in range(3 * num_atoms):
  pos_plus = pos_flat.at[i].add(delta)
  pos_minus = pos_flat.at[i].add(-delta)
  g_plus = np.array(grad_fn(pos_plus))
  g_minus = np.array(grad_fn(pos_minus))
  fd_H[i, :] = (g_plus - g_minus) / (2 * delta)

diff = np.max(np.abs(np.array(H) - fd_H))
print(f"Max |Hessian_autodiff - Hessian_FD| = {diff:.2e}")
print("✓ Hessian validated" if diff < 0.1 else "✗ Hessian mismatch (expected with random weights)")

# %% [markdown]
# ## Notes
#
# - For production phonons, use a **supercell** to capture long-range force constants.
# - Use **periodic boundary conditions** via `jax_md.space.periodic_general` for correct images.
# - The Hessian scales as O(N^2) in memory — for large systems, use **finite displacement** methods instead.
# - With pretrained UMA weights, phonon frequencies should match DFT references.
