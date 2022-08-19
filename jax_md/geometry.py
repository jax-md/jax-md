# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code to instantiate different systems.
"""

from typing import Callable

from jax_md import util
from jax_md import space
from jax_md import dataclasses

import jax.numpy as jnp
from functools import partial, reduce


# Types


Array = util.Array
Box = space.Box


# Regions


@dataclasses.dataclass
class Region:
    region_fn: Callable[[Array], Array]
    min_bound: Array
    max_bound: Array

    def __call__(self, R: Array) -> Array:
        return self.region_fn(R)


def rect(min_bound: Array, max_bound: Array) -> Region:
    center = (min_bound + max_bound) / 2
    size = (max_bound - min_bound) / 2

    @partial(jnp.vectorize, signature="(d)->()")
    def region_fn(R: Array) -> Array:
        assert R.ndim == 1
        assert R.shape[0] == min_bound.shape[0]
        assert R.shape[0] == max_bound.shape[0]
        return jnp.min(size - jnp.abs(R - center))

    return Region(region_fn, min_bound, max_bound)


def ball(center: Array, radius: Array) -> Region:
    @partial(jnp.vectorize, signature="(d)->()")
    def region_fn(R: Array) -> Array:
        (dim,) = R.shape
        assert dim == center.shape[0]
        return radius - jnp.linalg.norm(R - center)

    return Region(region_fn, center - radius, center + radius)


def cylinder(start: Array, end: Array, radius: Array) -> Region:
    n = end - start
    length = jnp.linalg.norm(n)
    n /= length
    center = (start + end) / 2

    @partial(jnp.vectorize, signature="(d)->()")
    def region_fn(R: Array) -> Array:
        (dim,) = R.shape
        assert dim == 3
        dR = R - center
        dR_dot_n = jnp.dot(dR, n)
        r_parallel = length / 2 - dR_dot_n
        r_perp = radius - jnp.linalg.norm(dR - dR_dot_n * n)
        return jnp.min(r_parallel, r_perp)

    return Region(
        region_fn,
    )


def translate(region: Region, shift: Array) -> Region:
    return Region(
        lambda R: region(R - shift), region.min_bound + shift, region.max_bound + shift
    )


def transform(region: Region, affine: Array) -> Region:
    inv_affine = jnp.linalg.inv(affine)
    return Region(
        lambda R: region(space.transform(inv_affine, R)),
        region.min_bound,
        region.max_bound,
    )


def union(*regions) -> Region:
    def region_fn(R: Array) -> Array:
        return reduce(lambda x, y: jnp.maximum(x, y), [r(R) for r in regions])

    min_bound = reduce(
        lambda x, y: jnp.minimum(x, y), [r.min_bound for r in regions], jnp.inf
    )
    max_bound = reduce(
        lambda x, y: jnp.maximum(x, y), [r.max_bound for r in regions], -jnp.inf
    )
    return Region(region_fn, min_bound, max_bound)


def intersection(*regions) -> Region:
    def region(R: Array) -> Array:
        return reduce(lambda x, y: jnp.minimum(x, y), [r(R) for r in regions])

    min_bound = reduce(
        lambda x, y: jnp.maximum(x, y), [r.min_bound for r in regions], -jnp.inf
    )
    max_bound = reduce(
        lambda x, y: jnp.minimum(x, y), [r.max_bound for r in regions], jnp.inf
    )
    return Region(region, min_bound, max_bound)


def complement(region: Region) -> Region:
    def not_region(R: Array) -> Array:
        return -region(R)

    # TODO: Is there a nicer way to keep track of this? Currently we just assume
    # that the complement is unbounded. Obviously this is not true and it loses
    # information; for example, the complement of the complement will still be
    # unbounded.
    min_bound = -jnp.inf * jnp.ones_like(region.min_bound)
    max_bound = jnp.inf * jnp.ones_like(region.max_bound)
    return Region(not_region, min_bound, max_bound)


def difference(a: Region, b: Region) -> Region:
    return intersection(a, complement(b))


def is_in(R: Array, region: Region) -> bool:
    return region(R) >= 0


# Lattice


def lattice(
    R: Array, unit_cell: Box, region: Region, fraction_coordinates: bool = True
) -> Array:
    dim = R.shape[-1]
    size = region.max_bound - region.min_bound
    assert unit_cell.shape == (dim, dim)
    assert region.min_bound.shape == (dim,)
    assert jnp.allclose(unit_cell, jnp.triu(unit_cell))

    if dim == 2:
        sx, sy = size
        dx, dy = unit_cell.T
        nx = jnp.ceil(sx / dx[0])
        ny = jnp.ceil(sy / dy[1])
        sx = jnp.ceil(ny * dy[0] / dx[0])
        Xs, Ys = jnp.meshgrid(jnp.arange(-sx, nx), jnp.arange(ny))
        shift = jnp.stack((Xs, Ys), axis=-1).reshape((-1, dim))
        R = R[:, None, :] + shift[None, :, [0]] * dx + shift[None, :, [1]] * dy
    elif dim == 3:
        sx, sy, sz = size
        dx, dy, dz = unit_cell.T
        nx = jnp.ceil(sx / dx[0])
        ny = jnp.ceil(sy / dy[1])
        nz = jnp.ceil(sz / dz[2])
        sx = jnp.maximum(jnp.ceil(ny * dy[0] / dx[0]), jnp.ceil(nz * dz[0] / dx[0]))
        sy = jnp.ceil(nz * dz[1] / dy[1])
        Xs, Ys, Zs = jnp.meshgrid(
            jnp.arange(-sx, nx), jnp.arange(-sy, ny), jnp.arange(nz)
        )
        shift = jnp.stack((Xs, Ys, Zs), axis=-1).reshape((-1, dim))
        R = (
            R[:, None, :]
            + shift[None, :, [0]] * dx
            + shift[None, :, [1]] * dy
            + shift[None, :, [2]] * dz
        )
    else:
        raise ValueError()

    R = jnp.reshape(R, (-1, dim)) + region.min_bound[None, :]
    mask = is_in(R, region)
    R = R[mask] - region.min_bound[None, :]
    return R
