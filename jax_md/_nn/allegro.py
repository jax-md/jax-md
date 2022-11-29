from functools import partial
from typing import Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jraph
from jax_md import partition, space

import jax
import jax.numpy as jnp
from jax import vmap


def bessel(d: jnp.ndarray, c: float, n: int) -> jnp.ndarray:
    n = jnp.arange(1, n + 1)
    d = d[..., None]
    x = jnp.where(d == 0, 1, d)
    return jnp.where(
        d == 0,
        jnp.sqrt(2 / c) * n * jnp.pi / c,
        jnp.sqrt(2 / c) * jnp.sin(n * jnp.pi / c * x) / x,
    )


def normalized_bessel(d: jnp.ndarray, n: int) -> jnp.ndarray:
    with jax.ensure_compile_time_eval():
        r = jnp.linspace(0.0, 1.0, 1000)
        b = bessel(r, 1.0, n)
        mu = jnp.trapz(b, r, axis=0)
        sig = jnp.trapz((b - mu) ** 2, r, axis=0) ** 0.5
    return (bessel(d, 1.0, n) - mu) / sig


def u(d: jnp.ndarray, p: int) -> jnp.ndarray:
    return e3nn.poly_envelope(p - 1, 2)(d)


def safe_norm(x: jnp.ndarray, axis: int = None) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis)
    return jnp.where(x2 == 0, 1, x2) ** 0.5


def safe_spherical_harmonics(lmax, r):
    """nan-safe spherical harmonics."""
    return e3nn.spherical_harmonics(
        e3nn.Irreps.spherical_harmonics(lmax),
        r / safe_norm(r, axis=-1)[..., None],
        False,
    )


def print_std(message: str, x: e3nn.IrrepsArray):
    x = x.reshape((-1, x.shape[-1]))
    std = e3nn.mean(e3nn.norm(x - e3nn.mean(x, axis=0), squared=True)) ** 0.5
    print(f"{message}: {x.shape} {x.irreps} {std.array}")


def layer(
    x: jnp.ndarray,
    V: e3nn.IrrepsArray,
    r: jnp.ndarray,
    *,
    edge_src: jnp.ndarray,
    features_out: int,
    n_out: int,
    irreps_out: e3nn.Irreps,
    p: int,
    Y: e3nn.IrrepsArray,
    num_neighbors: float,
) -> e3nn.IrrepsArray:
    r"""Allegro layer.

    Args:
        x (``jnp.ndarray``): scalars, array of shape ``(edge, features)``
        V (``e3nn.IrrepArray``): non-scalars, array of shape ``(edge, n, irreps)``
        r (``jnp.ndarray``): relative vectors, array of shape ``(edge, 3)``
        edge_src (``jnp.ndarray``): array of integers
        features_out (``int``): number of scalar features in output
        n_out (``int``): number of non-scalar irreps in output
        irreps_out (``e3nn.Irreps``): irreps of output
    """
    assert x.shape[0] == V.shape[0] == r.shape[0]
    assert (x.ndim, V.ndim, r.ndim) == (2, 3, 2)
    irreps_out = e3nn.Irreps(irreps_out)
    n = V.shape[1]

    d = safe_norm(r, axis=1)  # (edge,)

    # lmax = V.irreps.lmax + irreps_out.lmax
    # Y = safe_spherical_harmonics(lmax, r)

    w = e3nn.MultiLayerPerceptron([n], None)(x)  # (edge, n)
    wY = w[:, :, None] * Y[:, None, :]  # (edge, n, irreps)
    wY = e3nn.index_add(edge_src, wY, map_back=True) / jnp.sqrt(
        num_neighbors
    )  # (edge, n, irreps)

    V = e3nn.tensor_product(
        wY, V, filter_ir_out="0e" + irreps_out
    ).simplify()  # (edge, n, irreps)

    if "0e" in V.irreps:
        # Note: full_tensor_product gives ordered irreps and 0e is always the first
        assert V.irreps[0].ir == "0e"
        Vx, V = V[:, :, V.irreps[:1]], V[:, :, V.irreps[1:]]
        x = jnp.concatenate([x, Vx.array.reshape((x.shape[0], -1))], axis=1)

    x = e3nn.MultiLayerPerceptron(
        [features_out, features_out, features_out], jax.nn.silu
    )(
        x
    )  # (edge, features_out)
    x = u(d, p)[:, None] * x  # (edge, features_out)

    V = e3nn.Linear(irreps_out, n_out)(V)  # (edge, n_out, irreps_out)

    return (x, V)


def allegro_impl(
    graph: jraph.GraphsTuple,
    *,
    r_cut: float = 7.0,
    p: int = 6,
    features: int = 1024,
    n: int = 128,
    irreps: e3nn.Irreps = "0o + 1o + 1e + 2e + 2o + 3o + 3e",
    num_layers: int = 3,
    num_radial_basis: int = 8,
    num_species: int = 118,
    irreps_out: e3nn.Irreps = "0e",
    num_neighbors: float = 1.0,
) -> e3nn.IrrepsArray:
    z = graph.nodes

    if isinstance(graph.edges, tuple):
        dr, edge_input = graph.edges
    else:
        dr = graph.edges
        edge_input = e3nn.IrrepsArray.zeros("", (dr.shape[0],))

    edge_src, edge_dst = graph.senders, graph.receivers

    irreps = e3nn.Irreps(irreps)
    irreps_out = e3nn.Irreps(irreps_out)

    assert z.ndim == 1
    assert dr.shape == edge_src.shape + (3,)

    dr /= r_cut
    d = safe_norm(dr, axis=1)  # (edge,)
    x = jnp.concatenate(
        [
            normalized_bessel(d, num_radial_basis),
            jax.nn.one_hot(z[edge_src], num_species),
            jax.nn.one_hot(z[edge_dst], num_species),
        ],
        axis=1,
    )
    x = e3nn.MultiLayerPerceptron(
        [features // 8, features // 4, features // 2, features], jax.nn.silu
    )(
        x
    )  # (edge, features)
    x = u(d, p)[:, None] * x  # (edge, features)

    Y = safe_spherical_harmonics(2 * irreps.lmax, dr)  # (edge, irreps)
    V = Y[:, e3nn.Irreps.spherical_harmonics(irreps.lmax)]  # only up to lmax
    V = e3nn.IrrepsArray.cat([V, edge_input], axis=1)  # (edge, irreps)

    w = e3nn.MultiLayerPerceptron([n], None)(x)  # (edge, n)
    V = w[:, :, None] * V[:, None, :]  # (edge, n, irreps)

    for _ in range(num_layers):
        y, V = layer(
            x,
            V,
            dr,
            edge_src=edge_src,
            features_out=features,
            n_out=n,
            irreps_out=irreps,
            p=p,
            Y=Y,
            num_neighbors=num_neighbors,
        )

        alpha = 1 / 2
        x = (x + alpha * y) / jnp.sqrt(1 + alpha**2)

    x = e3nn.MultiLayerPerceptron([128], None)(x)  # (edge, 128)

    xV = e3nn.IrrepsArray.cat(
        [e3nn.IrrepsArray("128x0e", x), V.repeat_mul_by_last_axis()]
    )
    xV = e3nn.Linear(irreps_out)(xV)  # (edge, irreps_out)

    return xV


# jraph version.


def jraph_allegro(
    *,
    r_cut: float = 7.0,
    p: int = 6,
    features: int = 1024,
    n: int = 128,
    irreps: e3nn.Irreps = "0o + 1o + 1e + 2e + 2o + 3o + 3e",
    num_layers: int = 3,
    num_radial_basis: int = 8,
    num_species: int = 118,
    irreps_out: e3nn.Irreps = "0e",
) -> float:
    @hk.without_apply_rng
    @hk.transform
    def allegro_fn(graph):
        z, r = graph.nodes
        dr = r[graph.receivers] - r[graph.senders]

        if graph.edges is None:
            graph = graph._replace(nodes=z, edges=dr)
        else:
            graph = graph._replace(nodes=z, edges=(dr, graph.edges))

        edges = allegro_impl(
            graph,
            r_cut=r_cut,
            p=p,
            features=features,
            n=n,
            irreps=irreps,
            num_layers=num_layers,
            num_radial_basis=num_radial_basis,
            num_species=num_species,
            irreps_out=irreps_out,
        )
        graph = graph._replace(nodes=(z, r), edges=edges)
        return graph

    return allegro_fn


# Pure e3nn version.


def e3nn_allegro(
    *,
    r_cut: float = 7.0,
    p: int = 6,
    features: int = 1024,
    n: int = 128,
    irreps: e3nn.Irreps = "0o + 1o + 1e + 2e + 2o + 3o + 3e",
    num_layers: int = 3,
    num_radial_basis: int = 8,
    num_species: int = 118,
    irreps_out: e3nn.Irreps = "0e",
    num_edges: Optional[int] = None,
) -> float:
    @hk.without_apply_rng
    @hk.transform
    def allegro_fn(z: jnp.ndarray, r: jnp.ndarray) -> e3nn.IrrepsArray:
        num_nodes = len(z)

        z = jnp.concatenate((z, jnp.zeros((1,), dtype=z.dtype)))
        edge_src, edge_dst = e3nn.radius_graph(
            r, r_cut, size=num_edges, fill_src=num_nodes, fill_dst=0
        )

        dr = r[edge_dst] - r[edge_src]

        graph = jraph.GraphsTuple(
            nodes=z,
            edges=dr,
            receivers=edge_dst,
            senders=edge_src,
            globals=None,
            n_node=jnp.array([num_nodes, 1]),
            n_edge=jnp.array(
                [jnp.sum(edge_src < num_nodes), jnp.sum(edge_src == num_nodes)]
            ),
        )
        edge_output = allegro_impl(
            graph,
            r_cut=r_cut,
            p=p,
            features=features,
            n=n,
            irreps=irreps,
            num_layers=num_layers,
            num_radial_basis=num_radial_basis,
            num_species=num_species,
            irreps_out=irreps_out,
        )
        graph_count = len(graph.n_node)
        graph_idx = jnp.repeat(
            jnp.arange(graph_count), graph.n_edge, total_repeat_length=len(edge_output)
        )
        return jnp.squeeze(
            e3nn.index_add(graph_idx, edge_output, out_dim=graph_count)[:-1].array
        )

    return allegro_fn


# JAX MD versions.

# There are two versions of the function. One for use during deployment. This takes
# a displacement function and a box. It returns a function to build the neighbor list
# and a neural network module that takes positions and a neighbor list to produce an
# energy.
#
# The second version of the function is for use during training. It takes a position
# and a jraph GraphsTuple. Here it is understood that both the position and GraphsTuple
# have data for an entire batch of inputs arranged flat (as described in the jraph
# documentation). Usually, in practice, the graphs would be precomputed and cached
# on disk. It is necessary to pass the positions through again in case we want to
# train on forces.


def jaxmd_allegro(
    displacement: space.DisplacementFn,
    box: jnp.ndarray,
    species: jnp.ndarray,
    *,
    r_cut: float = 7.0,
    dr_threshold: float = 1.0,
    p: int = 6,
    features: int = 1024,
    n: int = 128,
    irreps: e3nn.Irreps = "0o + 1o + 1e + 2e + 2o + 3o + 3e",
    num_layers: int = 3,
    num_radial_basis: int = 8,
    num_species: int = 118,
    irreps_out: e3nn.Irreps = "0e",
    **neighbor_kwargs,
) -> Tuple[partition.NeighborFn, hk.Module]:
    z = species
    neighbor_fn = partition.neighbor_list(
        displacement,
        box,
        r_cut,
        dr_threshold,
        format=partition.Sparse,
        **neighbor_kwargs,
    )

    @hk.without_apply_rng
    @hk.transform
    def allegro_fn(r: jnp.ndarray, neighbor: partition.NeighborList, **kwargs) -> float:
        d = partial(displacement, **kwargs)
        dr = vmap(d)(r[neighbor.idx[0]], r[neighbor.idx[1]])

        graph = partition.to_jraph(neighbor, nodes=z, edges=dr)

        edge_output = allegro_impl(
            graph,
            r_cut=r_cut,
            p=p,
            features=features,
            n=n,
            irreps=irreps,
            num_layers=num_layers,
            num_radial_basis=num_radial_basis,
            num_species=num_species,
            irreps_out=irreps_out,
        )
        graph_count = len(graph.n_node)
        graph_idx = jnp.repeat(
            jnp.arange(graph_count), graph.n_edge, total_repeat_length=len(edge_output)
        )
        return jnp.squeeze(
            e3nn.index_add(graph_idx, edge_output, out_dim=graph_count)[:-1].array
        )

    return neighbor_fn, allegro_fn


def jaxmd_allegro_train(
    displacement: space.DisplacementFn,
    *,
    r_cut: float = 7.0,
    p: int = 6,
    features: int = 1024,
    n: int = 128,
    irreps: e3nn.Irreps = "0o + 1o + 1e + 2e + 2o + 3o + 3e",
    num_layers: int = 3,
    num_radial_basis: int = 8,
    num_species: int = 118,
    irreps_out: e3nn.Irreps = "0e",
) -> hk.Module:
    @hk.without_apply_rng
    @hk.transform
    def allegro_fn(r: jnp.ndarray, graph: jraph.GraphsTuple, **kwargs) -> jnp.ndarray:
        d = partial(displacement, **kwargs)
        dr = vmap(d)(r[graph.senders], r[graph.receivers])
        graph = graph._replace(edges=dr)
        edge_output = allegro_impl(
            graph,
            r_cut=r_cut,
            p=p,
            features=features,
            n=n,
            irreps=irreps,
            num_layers=num_layers,
            num_radial_basis=num_radial_basis,
            num_species=num_species,
            irreps_out=irreps_out,
        )
        graph_count = len(graph.n_node)
        graph_idx = jnp.repeat(
            jnp.arange(graph_count), graph.n_edge, total_repeat_length=len(edge_output)
        )
        return jnp.squeeze(
            e3nn.index_add(graph_idx, edge_output, out_dim=graph_count)[:-1].array
        )

    return allegro_fn
