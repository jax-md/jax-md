# Custom Alloy Extensions for `jax-md`

This document covers three custom extensions merged into `alloy-release` that
collectively enable JAX-accelerated binary alloy simulations:

1. [EAM Alloy Potential Loader](#1-eam-alloy-potential-loader)
2. [Species-Swap Monte Carlo Simulator](#2-species-swap-monte-carlo-simulator)
3. [JIT-Compatible Multi-Species Pair Correlation](#3-jit-compatible-multi-species-pair-correlation)

---

## 1. EAM Alloy Potential Loader

### What it does

The **Embedded Atom Method (EAM)** models metallic bonding by computing each
atom's energy as a function of the local electron density contributed by its
neighbours (the *embedding* energy) plus a pair-wise repulsion term. For
multi-component alloys, a separate embedding function, electron-density
(charge) function, and pair potential is needed for each element pair.

`jax_md.energy.eam_alloy` extends the single-species `eam` function to
support an arbitrary number of species. It takes pre-built JAX-callable
functions for each element and element pair and computes the full alloy
energy in a JIT-compatible way.

### Reading a LAMMPS `.eam.alloy` file

LAMMPS distributes EAM potentials in the *setfl* (`.eam.alloy`) format.
Use `load_lammps_eam_parameters_alloy` to parse such a file:

```python
from jax_md.energy import load_lammps_eam_parameters_alloy, eam_alloy

with open("tests/data/Ni_Al.eam.alloy") as f:
    charge_fns, embedding_fns, pairwise_fns, cutoff = \
        load_lammps_eam_parameters_alloy(f)
```

The function returns:

| Return value | Type | Description |
|---|---|---|
| `charge_fns` | `list[Callable[[Array], Array]]` | Electron density function per species |
| `embedding_fns` | `list[Callable[[Array], Array]]` | Embedding energy function per species |
| `pairwise_fns` | `list[Callable[[Array], Array]]` | Pairwise repulsion functions (lower-triangular, one per pair `(i,j)` with `j≤i`) |
| `cutoff` | `float` | Cutoff radius in Angstroms |

### Setting up the energy function

```python
import jax.numpy as jnp
from jax_md import space, energy

# 32-atom Ni-Al system: species 0 = Ni, species 1 = Al
N = 32
species = jnp.array([0] * 16 + [1] * 16)
box_size = 8.0  # Angstroms

displacement, _ = space.periodic(box_size)

with open("tests/data/Ni_Al.eam.alloy") as f:
    charge_fns, embedding_fns, pairwise_fns, cutoff = \
        energy.load_lammps_eam_parameters_alloy(f)

# Build the energy function once; species is fixed here.
energy_fn = energy.eam_alloy(
    displacement,
    charge_fns=charge_fns,
    embedding_fns=embedding_fns,
    pairwise_fns=pairwise_fns,
    cutoff=cutoff,
    species=species,  # fixed integer species array
)

# Evaluate on positions R of shape (N, 3).
E = energy_fn(R, species=species)
```

> **Note:** `eam_alloy` requires `species` both at *construction time* (to
> determine the number of species for loop unrolling) and at *call time* as a
> keyword argument.

### Neighbor-list variant

For large systems, use `eam_alloy_neighbor_list` to avoid O(N²) distance
evaluations:

```python
from jax_md.energy import eam_alloy_neighbor_list

neighbor_fn, energy_fn = eam_alloy_neighbor_list(
    displacement,
    box_size=box_size,
    charge_fns=charge_fns,
    embedding_fns=embedding_fns,
    pairwise_fns=pairwise_fns,
    cutoff=cutoff,
    species=species,
)

nbrs = neighbor_fn.allocate(R)
E = energy_fn(R, neighbor=nbrs, species=species)
```

---

## 2. Species-Swap Monte Carlo Simulator

### What it does

`jax_md.simulate.swap_mc` implements a basic **species-swap Monte Carlo**
simulation. At each step, the algorithm:

1. Randomly selects a pair of atoms `(i, j)`.
2. Proposes swapping their species labels.
3. Computes the energy before and after the swap.
4. Accepts the swap with the Metropolis probability
   `min(1, exp(-ΔE / kT))`.

Only species labels change; atomic positions remain fixed. This is useful for
equilibrating compositional disorder in alloys.

### API

#### `MCState`

```python
@dataclasses.dataclass
class MCState:
    position: Array   # shape (N, D)
    species: Array    # shape (N,), integer dtype
    key: Array        # JAX PRNGKey
```

#### `swap_mc(energy_fn, kT) -> (init_fn, apply_fn)`

```python
from jax_md.simulate import swap_mc, MCState

init_fn, apply_fn = swap_mc(energy_fn, kT=1.0)
```

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `energy_fn` | `Callable[[Array, ...], Array]` | Energy function with signature `fn(R, species=species, **kwargs)` |
| `kT` | `float` or JAX `Array` | Temperature × Boltzmann constant |

> **Important:** The signature of `apply_fn` depends on the type of `kT`:
>
> - `kT: float` → `apply_fn(state, **kwargs)`
> - `kT: jnp.ndarray` → `apply_fn(state, kT, **kwargs)` (dynamic temperature)

### Full example

```python
import jax
import jax.numpy as jnp
from jax import random
from jax_md import space, energy, smap, simulate

# ── System setup ──────────────────────────────────────────────────────────
N, DIM, BOX = 32, 3, 6.0
key = random.PRNGKey(0)
species = jnp.array([0] * 16 + [1] * 16, dtype=jnp.int32)

pos_key, sim_key = random.split(key)
R = random.uniform(pos_key, (N, DIM)) * BOX

displacement, _ = space.periodic(BOX)

# ── Energy function (LJ with per-species parameters) ──────────────────────
sigma   = jnp.array([[1.0, 1.2], [1.2, 1.4]])
epsilon = jnp.array([[1.0, 0.5], [0.5, 1.0]])

energy_fn = smap.pair(
    energy.lennard_jones,
    space.canonicalize_displacement_or_metric(displacement),
    species=2,          # integer → dynamic species at call time
    sigma=sigma,
    epsilon=epsilon,
    ignore_unused_parameters=True,
)

# ── Build and run swap MC ─────────────────────────────────────────────────
init_fn, apply_fn = simulate.swap_mc(energy_fn, kT=1.0)
state = init_fn(sim_key, R, species)

# JIT-compile for speed.
apply_fn_jit = jax.jit(apply_fn)

for step in range(1000):
    state = apply_fn_jit(state)

print("Final species:", state.species)
# Species counts are conserved.
assert jnp.sum(state.species == 0) == 16
assert jnp.sum(state.species == 1) == 16
```

### Combining with MD (hybrid swap MC)

For a full hybrid MD + swap MC algorithm see `simulate.hybrid_swap_mc` and
`simulate.SwapMCState`. The `swap_mc` function above implements the
swap-only variant.

---

## 3. JIT-Compatible Multi-Species Pair Correlation

### What it does

The **pair correlation function** g(r) measures the probability of finding a
particle at distance r from a reference particle, normalized by the ideal-gas
value. It is the primary structural diagnostic for liquids and amorphous
solids.

`jax_md.quantity.pair_correlation` (and its neighbor-list variant
`pair_correlation_neighbor_list`) support:

- **Single-species** systems: one g(r) for all pairs.
- **Static multi-species**: species array provided at *construction* time;
  one g(r) per species, computed as separate fixed-shape arrays.
- **Dynamic multi-species** *(JIT-compatible)*: pass an *integer* `species`
  at construction time; the species array is then supplied at *call time*,
  enabling JIT compilation even when species change (e.g., during swap MC).

### API

```python
from jax_md.quantity import pair_correlation, pair_correlation_neighbor_list

g_fn = pair_correlation(
    displacement_or_metric,   # displacement or metric function
    radii,                    # array of r values at which to evaluate g(r)
    sigma,                    # Gaussian smoothing width
    species=None,             # None | int array | integer
    eps=1e-7,
    compute_average=False,
)
```

| `species` value | `g_fn` signature | Output |
|---|---|---|
| `None` | `g_fn(R)` | Array `(N, nr)` — one g(r) per particle |
| `jnp.ndarray` (int array) | `g_fn(R)` | List of `(N, nr)` arrays, one per species |
| `int` (integer) | `g_fn(R, species)` | List of `(N, nr)` arrays, JIT-compatible |

### Multi-species example (static)

```python
import jax.numpy as jnp
from jax_md import space, quantity

displacement, _ = space.periodic(6.0)
radii = jnp.linspace(0.1, 3.0, 200)
species = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)

g_fn = quantity.pair_correlation(displacement, radii, sigma=0.1, species=species)
gofr = g_fn(R)   # list of two (6, 200) arrays: [g_Ni(r), g_Al(r)]
```

### Dynamic species example (JIT-compatible, for use with swap MC)

```python
import jax
import jax.numpy as jnp
from jax_md import space, quantity

displacement, _ = space.periodic(6.0)
radii = jnp.linspace(0.1, 3.0, 200)

# Pass the *number* of species as an integer to enable dynamic dispatch.
g_fn = quantity.pair_correlation(
    displacement,
    radii,
    sigma=0.1,
    species=2,          # ← integer, not an array
)
g_fn_jit = jax.jit(g_fn)

# At call time, supply the (potentially changing) species array.
gofr = g_fn_jit(R, species_array)
```

### Neighbor-list variant

For large systems (N > ~1000), use the neighbor-list variant to avoid O(N²)
cost:

```python
from jax_md.quantity import pair_correlation_neighbor_list

neighbor_fn, g_fn = pair_correlation_neighbor_list(
    displacement,
    box_size=6.0,
    radii=radii,
    sigma=0.1,
    species=2,
)

nbrs = neighbor_fn.allocate(R)
gofr = g_fn(R, nbrs, species=species_array)
```

### Integration with swap MC

Because `pair_correlation` with integer `species` accepts the species array
at call time, it can be evaluated at every swap MC step without recompilation:

```python
g_fn_jit = jax.jit(
    quantity.pair_correlation(displacement, radii, sigma=0.05, species=2)
)

# Inside the MC loop:
for step in range(1000):
    state = apply_fn_jit(state)
    if step % 100 == 0:
        gofr = g_fn_jit(state.position, state.species)
```

---

## See Also

- `jax_md.energy.eam` — single-species EAM
- `jax_md.simulate.hybrid_swap_mc` — combined MD + species-swap MC
- `jax_md.quantity.pair_correlation_neighbor_list` — neighbor-list g(r)
- `tests/test_swap_mc.py` — pytest tests for the swap MC simulator
- `tests/data/Ni_Al.eam.alloy` — Ni-Al EAM potential file (setfl format)
