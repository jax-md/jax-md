# mace_jaxmd_bridge.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax_md import space, partition


def _as_scalar(x):
    x = jnp.asarray(x)
    return jnp.reshape(x, ()) if x.shape == (1,) else x


def _make_cell_1x3x3(box_in, dtype):
    b = jnp.asarray(box_in)
    if b.shape == (3,):
        cell = jnp.diag(b)
    elif b.shape == (3, 3):
        cell = b
    else:
        raise ValueError(f"Unexpected box shape {b.shape} (expected (3,) or (3,3))")
    return cell[None, :, :].astype(dtype)


def make_mace_jaxmd_energy(
    *,
    jax_model,
    template_batch: dict,
    config: dict,
    box,
    z_atomic,                      # (N_real,) atomic numbers (Z) for REAL atoms only
    r_cutoff: float,
    dr_threshold: float,
    k_neighbors: int = 64,         # keep first k neighbor slots (no ranking!)
    capacity_multiplier: float = 2.0,
    include_head: bool = True,
):
    """
    MACE(JAX) potential wrapped for jax-md with neighbor lists.

    Key design choice for stable MD:
    - NO distance-based top-k ranking (which causes discontinuous edge sets and energy blow-up).
    - We take the FIRST k neighbor slots from the jax-md neighbor buffer (piecewise-constant
      between neighbor-list rebuilds, which is what the skin is for).

    Returns: neighbor_fn, shift_fn, energy_fn
      energy_fn(R, *, neighbors=None, neighbor_idx=None) -> scalar
    """

    displacement_fn, shift_fn = space.periodic(box)
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box,
        r_cutoff=r_cutoff,
        dr_threshold=dr_threshold,
        mask=True,
        capacity_multiplier=capacity_multiplier,
    )

    # Template sizes/dtypes expected by the converted model
    N_template = int(template_batch["positions"].shape[0])
    E_template = int(template_batch["edge_index"].shape[1])
    C_node = int(template_batch["node_attrs"].shape[1])

    # Hard sizing rule: E_template must cover N_template * k_neighbors
    if E_template < N_template * k_neighbors:
        raise ValueError(
            f"E_template ({E_template}) must be >= N_template*k_neighbors "
            f"({N_template*k_neighbors}). Reconvert model with larger e_template "
            f"or reduce k_neighbors."
        )

    pos_dtype = template_batch["positions"].dtype
    edge_dtype = template_batch["edge_index"].dtype
    shift_dtype = template_batch["shifts"].dtype
    us_dtype = template_batch["unit_shifts"].dtype
    node_attrs_dtype = template_batch["node_attrs"].dtype
    node_idx_dtype = template_batch["node_attrs_index"].dtype
    cell_dtype = template_batch["cell"].dtype

    # Finite far shift to neutralize invalid/padded edges safely (float32 friendly)
    far = jnp.asarray(r_cutoff + 2.0 * dr_threshold + 1.0, dtype=shift_dtype)
    FAR_SHIFT = jnp.array([far, 0.0, 0.0], dtype=shift_dtype)

    # Z -> species index lookup (index into config["atomic_numbers"])
    atomic_numbers = tuple(int(x) for x in config["atomic_numbers"])
    max_Z = int(max(atomic_numbers))
    Z_to_index = -jnp.ones((max_Z + 1,), dtype=jnp.int32)
    Z_to_index = Z_to_index.at[jnp.array(atomic_numbers, dtype=jnp.int32)].set(
        jnp.arange(len(atomic_numbers), dtype=jnp.int32)
    )
    z_atomic = jnp.asarray(z_atomic, dtype=jnp.int32)

    head_value = None
    if include_head and ("head" in template_batch):
        head_value = template_batch["head"].astype(jnp.int32).reshape(-1)

    # Cell / inv cell (static w.r.t. positions)
    cell_1x3x3 = _make_cell_1x3x3(box, cell_dtype)
    cell = cell_1x3x3[0]
    inv_cell = jnp.linalg.inv(cell)

    @jax.jit
    def make_batch_from_firstk(R_real, neighbor_idx):
        """
        Build a MACE batch dict of fixed template shapes using:
        - nodes: real atoms padded to N_template, padded nodes masked via node_attrs=0
        - edges: FIRST k neighbor slots per atom (N*k edges), then pad to E_template

        IMPORTANT: jax-md neighbor lists use sentinel index == N for empty slots.
        We must mask with (idx < N), not just (idx >= 0).
        """
        R_real = jnp.asarray(R_real, dtype=pos_dtype)
        neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)

        N = R_real.shape[0]
        M = neighbor_idx.shape[1]
        E_real = N * k_neighbors

        # ---- Positions (pad to N_template) ----
        positions = jnp.zeros((N_template, 3), dtype=pos_dtype)
        positions = positions.at[:N].set(R_real)

        # ---- Node attrs ----
        species_idx = jnp.zeros((N_template,), dtype=jnp.int32)
        species_idx_real = Z_to_index[z_atomic[:N]]  # (N,)
        species_idx = species_idx.at[:N].set(species_idx_real)

        node_mask = (jnp.arange(N_template, dtype=jnp.int32) < N).astype(node_attrs_dtype)
        node_attrs = jax.nn.one_hot(species_idx, num_classes=C_node, dtype=node_attrs_dtype)
        node_attrs = node_attrs * node_mask[:, None]

        node_attrs_index = jnp.where(
            node_mask > 0, species_idx, jnp.zeros_like(species_idx)
        ).astype(node_idx_dtype)

        # ---- Validity mask (handle sentinel idx == N) ----
        valid_slot = (neighbor_idx >= 0) & (neighbor_idx < N)

        # Build receivers0 with invalid slots -> 0 (safe gather index)
        receivers0 = jnp.where(valid_slot, neighbor_idx, jnp.zeros_like(neighbor_idx))

        # Forbid self edges (i -> i)
        self_edge = receivers0 == jnp.arange(N, dtype=jnp.int32)[:, None]
        valid_slot = valid_slot & (~self_edge)

        # Rebuild receivers0 after masking
        receivers0 = jnp.where(valid_slot, neighbor_idx, jnp.zeros_like(neighbor_idx))

        # ---- Take FIRST k neighbor slots (no ranking) ----
        if M < k_neighbors:
            extra = k_neighbors - M
            recv_p = jnp.concatenate(
                [receivers0, jnp.zeros((N, extra), dtype=receivers0.dtype)], axis=1
            )
            valid_p = jnp.concatenate(
                [valid_slot, jnp.zeros((N, extra), dtype=valid_slot.dtype)], axis=1
            )
        else:
            recv_p = receivers0
            valid_p = valid_slot

        receivers_k = recv_p[:, :k_neighbors]
        valid_k = valid_p[:, :k_neighbors]

        senders = jnp.repeat(jnp.arange(N, dtype=jnp.int32), k_neighbors)
        receivers = receivers_k.reshape(-1).astype(jnp.int32)
        valid_e = valid_k.reshape(-1)

        receivers = jnp.where(valid_e, receivers, senders)

        # ---- Differentiable minimum-image displacement for selected edges ----
        Ri_e = R_real[senders]
        Rj_e = R_real[receivers]

        dR_min_e = jax.vmap(displacement_fn)(Ri_e, Rj_e)  # jax-md convention
        dR_model = -dR_min_e                               # what MACE expects

        delta = (Rj_e - Ri_e)
        shifts = dR_model - delta

        unit_shifts_int = jnp.rint(shifts @ inv_cell).astype(jnp.int32)
        unit_shifts_int = lax.stop_gradient(unit_shifts_int)

        unit_shifts = unit_shifts_int.astype(us_dtype)
        unit_shifts = lax.stop_gradient(unit_shifts)

        shifts = (unit_shifts @ cell).astype(shift_dtype)
        shifts = lax.stop_gradient(shifts)

        shifts = jnp.where(valid_e[:, None], shifts, FAR_SHIFT[None, :])
        unit_shifts = jnp.where(
            valid_e[:, None], unit_shifts, jnp.zeros((E_real, 3), dtype=us_dtype)
        )

        # ---- Pad edges to E_template (tail padding) ----
        pad_n = E_template - E_real

        send_pad = jnp.zeros((pad_n,), dtype=jnp.int32)
        recv_pad = jnp.zeros((pad_n,), dtype=jnp.int32)
        us_pad = jnp.zeros((pad_n, 3), dtype=us_dtype)
        sh_pad = jnp.tile(FAR_SHIFT[None, :], (pad_n, 1))

        send2 = jnp.concatenate([senders, send_pad], axis=0)
        recv2 = jnp.concatenate([receivers, recv_pad], axis=0)
        us2 = jnp.concatenate([unit_shifts, us_pad], axis=0)
        sh2 = jnp.concatenate([shifts, sh_pad], axis=0)

        edge_index = jnp.stack([send2, recv2], axis=0).astype(edge_dtype)

        out = {
            "positions": positions,
            "node_attrs": node_attrs,
            "node_attrs_index": node_attrs_index,
            "edge_index": edge_index,
            "shifts": sh2.astype(shift_dtype),
            "unit_shifts": us2.astype(us_dtype),
            "batch": template_batch["batch"],
            "ptr": template_batch["ptr"],
            "cell": cell_1x3x3,
        }
        if head_value is not None:
            out["head"] = head_value
        return out

    @jax.jit
    def energy_fn(R, *, neighbors=None, neighbor_idx=None):
        if neighbor_idx is None:
            if neighbors is None:
                raise ValueError("Provide either neighbors=... or neighbor_idx=...")
            neighbor_idx = neighbors.idx

        batch = make_batch_from_firstk(R, neighbor_idx)

        # NNX: call the module directly (no variables/apply)
        out = jax_model(batch)

        e = out["energy"] if isinstance(out, dict) and "energy" in out else out
        return _as_scalar(e)

    return neighbor_fn, shift_fn, energy_fn

