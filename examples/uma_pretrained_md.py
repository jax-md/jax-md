# %% [markdown]
# # Molecular Dynamics with Pretrained UMA
#
# Run NVE and NVT simulations using a pretrained UMA MoE model
# with JAX-MD's simulation infrastructure. Forces are computed
# via `jax.grad(energy)` for exact energy conservation.

# %%
import jax
import jax.numpy as jnp
import numpy as np

from jax_md import space, simulate, quantity
from jax_md._nn.uma import load_pretrained, UMAMoEBackbone
from jax_md._nn.uma.nn.embedding import dataset_names_to_indices
from jax_md._nn.uma.heads import MLPEnergyHead

# %% [markdown]
# ## Setup: pretrained model + energy function

# %%
config, params, head_params = load_pretrained('uma-s-1p1')
model = UMAMoEBackbone(config=config)

head = MLPEnergyHead(
  sphere_channels=config.sphere_channels,
  hidden_channels=config.hidden_channels,
)

# FCC Cu unit cell
a = 3.615
positions = jnp.array([
  [0, 0, 0], [a/2, a/2, 0], [a/2, 0, a/2], [0, a/2, a/2],
], dtype=jnp.float32)
Z = jnp.array([29, 29, 29, 29], dtype=jnp.int32)
batch = jnp.zeros(4, dtype=jnp.int32)
ds_idx = dataset_names_to_indices(['omat'], config.dataset_list)
charge = jnp.array([0], dtype=jnp.int32)
spin = jnp.array([0], dtype=jnp.int32)

def build_edges(pos, cutoff):
  n = len(pos)
  s, d = [], []
  pos_np = np.asarray(pos)
  for i in range(n):
    for j in range(n):
      if i != j and np.linalg.norm(pos_np[i] - pos_np[j]) < cutoff:
        s.append(j)
        d.append(i)
  return jnp.array([s, d], dtype=jnp.int32)

edge_index = build_edges(positions, config.cutoff)

# Initialize head params
emb = model.apply(params, positions, Z, batch, edge_index,
                   positions[edge_index[0]] - positions[edge_index[1]],
                   charge, spin, ds_idx)
key = jax.random.PRNGKey(0)
head_params = head.init(key, emb['node_embedding'], batch, 1)

# %% [markdown]
# ## Energy function for MD

# %%
def energy_fn(pos):
  """Total energy as a function of positions (for autodiff forces)."""
  ev = pos[edge_index[0]] - pos[edge_index[1]]
  emb = model.apply(params, pos, Z, batch, edge_index, ev,
                     charge, spin, ds_idx)
  return head.apply(head_params, emb['node_embedding'], batch, 1)['energy'][0]

# Test
E0 = energy_fn(positions)
F0 = -jax.grad(energy_fn)(positions)
print(f"Initial energy: {E0:.6f}")
print(f"Initial forces:\n{F0}")

# %% [markdown]
# ## NVE Simulation (constant energy)

# %%
displacement_fn, shift_fn = space.free()
dt = 0.0005  # small timestep (reduced units)

def nve_energy(R, **kwargs):
  return energy_fn(R)

init_fn, apply_fn = simulate.nve(nve_energy, shift_fn, dt=dt)
apply_fn = jax.jit(apply_fn)

# Initialize with small random velocities
key, subkey = jax.random.split(key)
state = init_fn(subkey, positions, kT=0.01)

print("\n=== NVE Simulation ===")
print(f"{'Step':>5} {'KE':>12} {'PE':>12} {'Total':>12} {'Drift %':>10}")

E_total_0 = None
for step in range(50):
  state = apply_fn(state)

  if step % 10 == 0:
    KE = float(quantity.kinetic_energy(state.momentum, state.mass))
    PE = float(nve_energy(state.position))
    total = KE + PE
    if E_total_0 is None:
      E_total_0 = total
    drift = abs(total - E_total_0) / max(abs(E_total_0), 1e-10) * 100
    print(f"{step:5d} {KE:12.6f} {PE:12.6f} {total:12.6f} {drift:10.4f}")

# %% [markdown]
# ## NVT Simulation (constant temperature, Nose-Hoover)

# %%
kT = 0.01
tau = dt * 50

init_nvt, apply_nvt = simulate.nvt_nose_hoover(
  nve_energy, shift_fn, dt=dt, kT=kT, tau=tau,
)
apply_nvt = jax.jit(apply_nvt)

state_nvt = init_nvt(key, positions, kT=kT)

print("\n=== NVT Simulation (Nose-Hoover) ===")
print(f"Target temperature: kT = {kT}")

for step in range(50):
  state_nvt = apply_nvt(state_nvt, kT=kT)

  if step % 10 == 0:
    KE = float(quantity.kinetic_energy(state_nvt.momentum, state_nvt.mass))
    T_inst = 2 * KE / (3 * len(Z))
    print(f"  Step {step:3d}: T = {T_inst:.6f} (target {kT})")

# %% [markdown]
# ## Charged molecule MD (omol task)

# %%
print("\n=== Charged molecule: Li+ in water ===")

# LiF ion pair
lif_pos = jnp.array([
  [0.0, 0.0, 0.0],   # Li
  [2.0, 0.0, 0.0],   # F
], dtype=jnp.float32)
lif_Z = jnp.array([3, 9], dtype=jnp.int32)
lif_batch = jnp.zeros(2, dtype=jnp.int32)
lif_ds = dataset_names_to_indices(['omol'], config.dataset_list)
lif_charge = jnp.array([0], dtype=jnp.int32)  # neutral pair
lif_spin = jnp.array([1], dtype=jnp.int32)     # singlet

lif_ei = build_edges(lif_pos, config.cutoff)

emb_lif = model.apply(params, lif_pos, lif_Z, lif_batch, lif_ei,
                       lif_pos[lif_ei[0]] - lif_pos[lif_ei[1]],
                       lif_charge, lif_spin, lif_ds)
print(f"LiF embedding: {emb_lif['node_embedding'].shape}")
print(f"Li l=0: {emb_lif['node_embedding'][0, 0, :4]}")
print(f"F  l=0: {emb_lif['node_embedding'][1, 0, :4]}")
