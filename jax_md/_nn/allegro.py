from typing import Optional

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph


def normalized_bessel(d: jnp.ndarray, n: int) -> jnp.ndarray:
    with jax.ensure_compile_time_eval():
        r = jnp.linspace(0.0, 1.0, 1000, dtype=d.dtype)
        b = e3nn.bessel(r, n)
        mu = jnp.trapz(b, r, axis=0)
        sig = jnp.trapz((b - mu) ** 2, r, axis=0) ** 0.5
    return (e3nn.bessel(d, n) - mu) / sig


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


class AllegroLayer(nn.Module):
    @nn.compact
    def __call__(
        self,
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

        w = e3nn.flax.MultiLayerPerceptron((n,))(x)  # (edge, n)
        wY = w[:, :, None] * Y[:, None, :]  # (edge, n, irreps)
        wY = e3nn.index_add(edge_src, wY, map_back=True) / jnp.sqrt(
            num_neighbors
        )  # (edge, n, irreps)

        V = e3nn.tensor_product(
            wY, V, filter_ir_out="0e" + irreps_out
        )  # (edge, n, irreps)

        if "0e" in V.irreps:
            x = jnp.concatenate([x, V.filter(keep="0e").axis_to_mul().array], axis=1)
            V = V.filter(drop="0e")

        x = e3nn.flax.MultiLayerPerceptron(
            (features_out, features_out, features_out),
            jax.nn.silu,
            output_activation=False,
        )(
            x
        )  # (edge, features_out)
        x = u(d, p)[:, None] * x  # (edge, features_out)

        V = V.axis_to_mul()  # (edge, n * irreps)
        V = e3nn.flax.Linear(n_out * irreps_out)(V)  # (edge, n_out * irreps_out)
        V = V.mul_to_axis(n_out)  # (edge, n_out, irreps_out)

        return (x, V)


class Allegro(nn.Module):
    r_cut: float
    num_neighbors: float
    p: int = 6
    features: int = 1024
    n: int = 128
    irreps: e3nn.Irreps = e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e")
    num_layers: int = 3
    num_radial_basis: int = 8
    irreps_out: e3nn.Irreps = e3nn.Irreps("0e")

    @nn.compact
    def __call__(
        self,
        node_attrs: jnp.ndarray,  # jax.nn.one_hot(z, num_species)
        edge_src: jnp.ndarray,
        edge_dst: jnp.ndarray,
        edge_vectors: jnp.ndarray,
        edge_features: Optional[e3nn.IrrepsArray] = None,
    ) -> e3nn.IrrepsArray:
        dr = edge_vectors
        irreps = e3nn.Irreps(self.irreps)
        irreps_out = e3nn.Irreps(self.irreps_out)

        assert dr.shape == edge_src.shape + (3,)

        dr /= self.r_cut
        d = safe_norm(dr, axis=1)  # (edge,)
        x = jnp.concatenate(
            [
                normalized_bessel(d, self.num_radial_basis),
                node_attrs[edge_src],
                node_attrs[edge_dst],
            ],
            axis=1,
        )
        x = e3nn.flax.MultiLayerPerceptron(
            (self.features // 8, self.features // 4, self.features // 2, self.features),
            jax.nn.silu,
            output_activation=False,
        )(
            x
        )  # (edge, features)
        x = u(d, self.p)[:, None] * x  # (edge, features)

        Y = safe_spherical_harmonics(2 * irreps.lmax, dr)  # (edge, irreps)
        V = Y.slice_by_mul[: irreps.lmax + 1]  # only up to lmax

        if edge_features is not None:
            V = e3nn.concatenate([V, edge_features])  # (edge, irreps)

        w = e3nn.flax.MultiLayerPerceptron((self.n,))(x)  # (edge, n)
        V = w[:, :, None] * V[:, None, :]  # (edge, n, irreps)

        for _ in range(self.num_layers):
            y, V = AllegroLayer()(
                x,
                V,
                dr,
                edge_src=edge_src,
                features_out=self.features,
                n_out=self.n,
                irreps_out=irreps,
                p=self.p,
                Y=Y,
                num_neighbors=self.num_neighbors,
            )

            alpha = 1 / 2
            x = (x + alpha * y) / jnp.sqrt(1 + alpha**2)

        x = e3nn.flax.MultiLayerPerceptron((128,))(x)  # (edge, 128)

        xV = e3nn.concatenate([e3nn.IrrepsArray("128x0e", x), V.axis_to_mul()])
        xV = e3nn.flax.Linear(irreps_out)(xV)  # (edge, irreps_out)

        return xV


class AllegroEnergyModel(nn.Module):
    r_cut: float
    num_neighbors: float
    p: int = 6
    features: int = 1024
    n: int = 128
    irreps: e3nn.Irreps = e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e")
    num_layers: int = 3
    num_radial_basis: int = 8

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        node_attrs: e3nn.IrrepsArray = graph.nodes  # [n_atoms, irreps_attributes]
        edge_src: jnp.ndarray = graph.senders  # [n_edges]
        edge_dst: jnp.ndarray = graph.receivers  # [n_edges]
        edge_vectors: jnp.ndarray = graph.edges  # [n_edges, 3]

        edge_outputs = Allegro(
            r_cut=self.r_cut,
            num_neighbors=self.num_neighbors,
            p=self.p,
            features=self.features,
            n=self.n,
            irreps=self.irreps,
            num_layers=self.num_layers,
            num_radial_basis=self.num_radial_basis,
        )(
            node_attrs,
            edge_src,
            edge_dst,
            edge_vectors,
        )

        return e3nn.scatter_sum(edge_outputs, nel=graph.n_edge)
