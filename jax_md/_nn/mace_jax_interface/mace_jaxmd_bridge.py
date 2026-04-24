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


def _to_cartesian(R, cell, fractional_coordinates: bool):
    R = jnp.asarray(R)
    if fractional_coordinates:
        return R @ cell
    return R


def make_mace_jaxmd_energy(
    *,
    jax_model,
    template_batch: dict,
    config: dict,
    box,
    z_atomic,                      # (N_real,) atomic numbers for real atoms only
    r_cutoff: float,
    dr_threshold: float,
    k_neighbors: int = 64,
    capacity_multiplier: float = 2.0,
    include_head: bool = True,
    fractional_coordinates: bool = False,
):
    """
    MACE(JAX) potential wrapped for jax-md with neighbor lists.

    Returns:
      neighbor_fn, shift_fn, energy_fn, freeze_graph_fn, make_fixed_graph_energy_fn

    energy_fn signature:
      energy_fn(R, *, box=None, neighbors=None, neighbor_idx=None) -> scalar

    freeze_graph_fn signature:
      freeze_graph_fn(R, *, box=None, neighbors=None, neighbor_idx=None) -> dict

    make_fixed_graph_energy_fn signature:
      make_fixed_graph_energy_fn(fixed_graph) -> fixed_energy_fn
      fixed_energy_fn(R, *, box=None) -> scalar

    Notes
    -----
    - If fractional_coordinates=False:
        R is interpreted as Cartesian coordinates.
    - If fractional_coordinates=True:
        R is interpreted as fractional coordinates in the unit cell.
        The MACE model still receives Cartesian positions/shifts internally.
    """

    box_default = jnp.asarray(box)

    # Space / shift / neighbor list setup
    if fractional_coordinates:
        displacement_fn_fixed, shift_fn = space.periodic_general(
            box_default,
            fractional_coordinates=True,
        )
        neighbor_fn = partition.neighbor_list(
            displacement_fn_fixed,
            box_default,
            r_cutoff=r_cutoff,
            dr_threshold=dr_threshold,
            mask=True,
            capacity_multiplier=capacity_multiplier,
            fractional_coordinates=True,
        )
    else:
        displacement_fn_fixed, shift_fn = space.periodic(box_default)
        neighbor_fn = partition.neighbor_list(
            displacement_fn_fixed,
            box_default,
            r_cutoff=r_cutoff,
            dr_threshold=dr_threshold,
            mask=True,
            capacity_multiplier=capacity_multiplier,
        )

    # Template sizes / dtypes expected by converted MACE model
    N_template = int(template_batch["positions"].shape[0])
    E_template = int(template_batch["edge_index"].shape[1])
    C_node = int(template_batch["node_attrs"].shape[1])

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

    far = jnp.asarray(r_cutoff + 2.0 * dr_threshold + 1.0, dtype=shift_dtype)
    FAR_SHIFT = jnp.array([far, 0.0, 0.0], dtype=shift_dtype)

    # Z -> species index lookup
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

    def _runtime_displacement_fn(box_now):
        box_arr = jnp.asarray(box_now)
        if fractional_coordinates:
            return space.periodic_general(
                box_arr,
                fractional_coordinates=True,
            )[0]
        else:
            if box_arr.shape == (3,):
                return space.periodic(box_arr)[0]
            elif box_arr.shape == (3, 3):
                return space.periodic_general(box_arr)[0]
            else:
                raise ValueError(f"Unexpected runtime box shape: {box_arr.shape}")

    @jax.jit
    def make_batch_from_firstk(R_real, neighbor_idx, box_now):
        """
        Build a fixed-shape MACE batch using:
        - real atoms padded to N_template
        - first k neighbor slots per atom from jax-md neighbor list
        - runtime box for displacement/cell/shifts
        """
        R_input = jnp.asarray(R_real, dtype=pos_dtype)
        neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)

        cell_1x3x3 = _make_cell_1x3x3(box_now, cell_dtype)
        cell = cell_1x3x3[0]
        inv_cell = jnp.linalg.inv(cell)

        disp_now = _runtime_displacement_fn(box_now)

        N = R_input.shape[0]
        M = neighbor_idx.shape[1]
        E_real = N * k_neighbors

        R_cart = _to_cartesian(R_input, cell, fractional_coordinates).astype(pos_dtype)

        # ---- Positions ----
        positions = jnp.zeros((N_template, 3), dtype=pos_dtype)
        positions = positions.at[:N].set(R_cart)

        # ---- Node attrs ----
        species_idx_real = Z_to_index[z_atomic[:N]]

        species_idx = jnp.zeros((N_template,), dtype=jnp.int32)
        species_idx = species_idx.at[:N].set(species_idx_real)

        node_attrs = jnp.zeros((N_template, C_node), dtype=node_attrs_dtype)
        node_attrs = node_attrs.at[:N].set(
            jax.nn.one_hot(
                species_idx_real,
                num_classes=C_node,
                dtype=node_attrs_dtype,
            )
        )

        node_attrs_index = jnp.zeros((N_template,), dtype=node_idx_dtype)
        node_attrs_index = node_attrs_index.at[:N].set(
            species_idx_real.astype(node_idx_dtype)
        )

        # ---- Valid slots ----
        valid_slot = (neighbor_idx >= 0) & (neighbor_idx < N)
        receivers0 = jnp.where(valid_slot, neighbor_idx, jnp.zeros_like(neighbor_idx))

        self_edge = receivers0 == jnp.arange(N, dtype=jnp.int32)[:, None]
        valid_slot = valid_slot & (~self_edge)
        receivers0 = jnp.where(valid_slot, neighbor_idx, jnp.zeros_like(neighbor_idx))

        # ---- Take first k slots ----
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

        # ---- Edge geometry ----
        Ri_in = R_input[senders]
        Rj_in = R_input[receivers]
        dR_min_in = jax.vmap(disp_now)(Ri_in, Rj_in)

        # IMPORTANT:
        # In fractional_coordinates=True mode, periodic_general already returns
        # a physical-space displacement, so no additional @ cell is needed.
        dR_model = -dR_min_in

        delta_cart = R_cart[receivers] - R_cart[senders]

        # Exact continuous shifts: crucial for force/stress consistency
        exact_shifts = (dR_model - delta_cart).astype(shift_dtype)

        # Integer image bookkeeping
        unit_shifts_int = jnp.rint(exact_shifts @ inv_cell).astype(jnp.int32)
        unit_shifts_int = lax.stop_gradient(unit_shifts_int)

        unit_shifts = unit_shifts_int.astype(us_dtype)
        unit_shifts = lax.stop_gradient(unit_shifts)

        shifts = lax.stop_gradient(exact_shifts)

        # Mask invalid edges
        shifts = jnp.where(valid_e[:, None], shifts, FAR_SHIFT[None, :])
        unit_shifts = jnp.where(
            valid_e[:, None], unit_shifts, jnp.zeros((E_real, 3), dtype=us_dtype)
        )

        # ---- Pad edges ----
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
    def freeze_graph_fn(R, *, box=None, neighbors=None, neighbor_idx=None):
        if neighbor_idx is None:
            if neighbors is None:
                raise ValueError("Provide either neighbors=... or neighbor_idx=...")
            neighbor_idx = neighbors.idx

        box_now = box_default if box is None else box

        cell_1x3x3 = _make_cell_1x3x3(box_now, cell_dtype)
        cell = cell_1x3x3[0]
        inv_cell = jnp.linalg.inv(cell)

        disp_now = _runtime_displacement_fn(box_now)

        R_input = jnp.asarray(R, dtype=pos_dtype)
        neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)

        N = R_input.shape[0]
        M = neighbor_idx.shape[1]

        valid_slot = (neighbor_idx >= 0) & (neighbor_idx < N)
        receivers0 = jnp.where(valid_slot, neighbor_idx, jnp.zeros_like(neighbor_idx))

        self_edge = receivers0 == jnp.arange(N, dtype=jnp.int32)[:, None]
        valid_slot = valid_slot & (~self_edge)
        receivers0 = jnp.where(valid_slot, neighbor_idx, jnp.zeros_like(neighbor_idx))

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

        R_cart = _to_cartesian(R_input, cell, fractional_coordinates).astype(pos_dtype)

        Ri_in = R_input[senders]
        Rj_in = R_input[receivers]
        dR_min_in = jax.vmap(disp_now)(Ri_in, Rj_in)

        # IMPORTANT:
        # In fractional_coordinates=True mode, periodic_general already returns
        # a physical-space displacement, so no additional @ cell is needed.
        dR_model = -dR_min_in

        delta_cart = R_cart[receivers] - R_cart[senders]

        exact_shifts = (dR_model - delta_cart).astype(shift_dtype)
        unit_shifts_int = jnp.rint(exact_shifts @ inv_cell).astype(jnp.int32)
        unit_shifts_int = lax.stop_gradient(unit_shifts_int)

        return {
            "neighbor_idx": neighbor_idx,
            "senders": senders,
            "receivers": receivers,
            "valid_e": valid_e,
            "unit_shifts_int": unit_shifts_int,
        }

    def make_fixed_graph_energy_fn(fixed_graph):
        """
        Freeze graph topology and integer image bookkeeping from a reference
        configuration, then return an energy function that only updates:
        - positions
        - cell
        - continuous shifts = unit_shifts @ cell

        This is intended for stress / pressure calculations where the graph
        must remain fixed under infinitesimal strain.
        """
        senders = jnp.asarray(fixed_graph["senders"], dtype=jnp.int32)
        receivers = jnp.asarray(fixed_graph["receivers"], dtype=jnp.int32)
        valid_e = jnp.asarray(fixed_graph["valid_e"], dtype=bool)
        unit_shifts_int = jnp.asarray(fixed_graph["unit_shifts_int"], dtype=jnp.int32)

        E_real = int(senders.shape[0])

        if E_real > E_template:
            raise ValueError(
                f"Fixed graph has {E_real} edges, but template only supports "
                f"{E_template}. Reconvert model with a larger e_template."
            )

        pad_n = E_template - E_real

        send_pad = jnp.zeros((pad_n,), dtype=jnp.int32)
        recv_pad = jnp.zeros((pad_n,), dtype=jnp.int32)
        us_pad = jnp.zeros((pad_n, 3), dtype=us_dtype)
        sh_pad = jnp.tile(FAR_SHIFT[None, :], (pad_n, 1))

        send2 = jnp.concatenate([senders, send_pad], axis=0)
        recv2 = jnp.concatenate([receivers, recv_pad], axis=0)
        edge_index = jnp.stack([send2, recv2], axis=0).astype(edge_dtype)

        @jax.jit
        def fixed_energy_fn(R, *, box=None):
            box_now = box_default if box is None else box

            R_input = jnp.asarray(R, dtype=pos_dtype)
            N = R_input.shape[0]

            cell_1x3x3 = _make_cell_1x3x3(box_now, cell_dtype)
            cell = cell_1x3x3[0]

            R_cart = _to_cartesian(R_input, cell, fractional_coordinates).astype(pos_dtype)

            # ---- Positions ----
            positions = jnp.zeros((N_template, 3), dtype=pos_dtype)
            positions = positions.at[:N].set(R_cart)

            # ---- Node attrs ----
            species_idx_real = Z_to_index[z_atomic[:N]]

            species_idx = jnp.zeros((N_template,), dtype=jnp.int32)
            species_idx = species_idx.at[:N].set(species_idx_real)

            node_attrs = jnp.zeros((N_template, C_node), dtype=node_attrs_dtype)
            node_attrs = node_attrs.at[:N].set(
                jax.nn.one_hot(
                    species_idx_real,
                    num_classes=C_node,
                    dtype=node_attrs_dtype,
                )
            )

            node_attrs_index = jnp.zeros((N_template,), dtype=node_idx_dtype)
            node_attrs_index = node_attrs_index.at[:N].set(
                species_idx_real.astype(node_idx_dtype)
            )

            # ---- Frozen graph, runtime shifts ----
            unit_shifts = unit_shifts_int.astype(us_dtype)
            unit_shifts = lax.stop_gradient(unit_shifts)

            shifts = (unit_shifts @ cell).astype(shift_dtype)

            shifts = jnp.where(valid_e[:, None], shifts, FAR_SHIFT[None, :])
            unit_shifts = jnp.where(
                valid_e[:, None], unit_shifts, jnp.zeros((E_real, 3), dtype=us_dtype)
            )

            us2 = jnp.concatenate([unit_shifts, us_pad], axis=0)
            sh2 = jnp.concatenate([shifts, sh_pad], axis=0)

            batch = {
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
                batch["head"] = head_value

            out = jax_model(batch)
            e = out["energy"] if isinstance(out, dict) and "energy" in out else out
            return _as_scalar(e)

        return fixed_energy_fn

    @jax.jit
    def energy_fn(R, *, box=None, neighbors=None, neighbor_idx=None):
        if neighbor_idx is None:
            if neighbors is None:
                raise ValueError("Provide either neighbors=... or neighbor_idx=...")
            neighbor_idx = neighbors.idx

        box_now = box_default if box is None else box
        batch = make_batch_from_firstk(R, neighbor_idx, box_now)

        out = jax_model(batch)
        e = out["energy"] if isinstance(out, dict) and "energy" in out else out
        return _as_scalar(e)

    return neighbor_fn, shift_fn, energy_fn, freeze_graph_fn, make_fixed_graph_energy_fn
