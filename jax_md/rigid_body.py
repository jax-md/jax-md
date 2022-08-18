# Copyright 2022 Google LLC
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


"""Code to simulate rigid bodies in two- and three-dimensions.

This code contains a number of pieces that handle different parts of
rigid body simulations.

To start with, we include some quaternion utilities for representing oriented
bodies in three dimensions.

Rigid body simulations are split into two components.

1) The state of rigid bodies are represented by a dataclass containing a
center-of-mass position and an orientation. Along with this type
representation, the core simulation functions are overloaded to automatically
allow deterministic NVE and NVT simulations to work with state composed of
RigidBody objects (see `simulation.py` for details). If you need any other
simulation environments, please raise a github issue.

One subtlety of the type system that we use here is that a host of related
quantities are represented by RigidBody objects. For example, the momentum
is represented by a RigidBody containing the linear momentum and angular
momentum, while the mass is a RigidBody containing the total mass and the
moment of inertia. This allows us to naturally use JAX's tree_map utilities
to jointly map over the different related quantities. Additionally, forces
inherit the RigidBody type with a center-of-mass force and torque.

2) Interactions between rigid bodies are specified. This is largely responsible
for dictating the shape of the rigid body. While arbitrary interactions are
possible, we include utility functions for producing rigid bodies that are
made by the union of point-like particles. This captures many common models
of rigid molecules and colloids. These functions work by providing a
RigidPointUnion object that specifies the location of point particles in the
body frame along with a pointwise energy function. This approach works with or
without neighbor lists and yields a function that computes the total energy on
a system of rigid bodies.
"""

from typing import Optional, Tuple, Any, Union, Callable

from absl import logging

import numpy as onp

import jax
from jax import vmap
from jax import ops
from jax import random
import jax.numpy as jnp
from jax_md import dataclasses, util, space, partition, quantity, simulate
from functools import partial, reduce

from jax.tree_util import tree_map, tree_reduce

import operator


DType = Any
Array = util.Array
PyTree = Any
f64 = util.f64
f32 = util.f32
KeyArray = random.KeyArray
NeighborListFns = partition.NeighborListFns
ShiftFn = space.ShiftFn
UpdateFn = simulate.UpdateFn


"""Quaternion Utilities.

The quaternion utilities are divided into private helper functions and public
functions. The public versions of the function take `Quaternion` objects which
help enforce type safety. The private version of the functions take raw arrays,
but use JAX's vectorize utilities to automatically vectorize over any number
of additional dimensions.

To compute derivatives of quaternions as the tangent space of S^3 we follow the
perspective outlined in
New Langevin and Gradient Thermostats for Rigid Body Dynamics
R. L. Davidchack, T. E. Ouldridge, and M. V. Tretyakov
J. Chem. Phys. 142, 144114 (2015)

TODO: We should make sure that all publicly exposed quaternion operations have
correct derivatives, at some point.
"""


def _is_float64(x: Array) -> bool:
  return x.dtype in [jnp.float64, onp.float64]


@partial(jnp.vectorize, signature='(q),(q)->(q)')
def _quaternion_multiply(lhs: Array, rhs: Array) -> Array:
  wl, xl, yl, zl = lhs
  wr, xr, yr, zr = rhs

  dtype = f64 if _is_float64(lhs) or _is_float64(rhs) else f32

  return jnp.array([
      -xl * xr - yl * yr - zl * zr + wl * wr,
      xl * wr + yl * zr - zl * yr + wl * xr,
      -xl * zr + yl * wr + zl * xr + wl * yr,
      xl * yr - yl * xr + zl * wr + wl * zr
  ], dtype=dtype)


@partial(jnp.vectorize, signature='(q)->(q)')
def _quaternion_conjugate(q: Array) -> Array:
  w, x, y, z = q
  return jnp.array([w, -x, -y, -z], dtype=q.dtype)


def _quaternion_rotate_raw(q: Array, v: Array) -> Array:
  """Rotates a vector by a given quaternion."""
  if q.shape != (4,):
    raise ValueError('quaternion_rotate expects quaternion to have '
                     f'4-dimensions. Found {q.shape}.')
  if v.shape != (3,):
    raise ValueError('quaternion_rotate expects vector to have '
                     f'three-dimensions. Found {v.shape}.')

  v = jnp.concatenate([jnp.zeros((1,), v.dtype), v])
  q = _quaternion_multiply(q, _quaternion_multiply(v, _quaternion_conjugate(q)))
  return q[1:]


@jax.custom_vjp
def _quaternion_rotate(q: Array, v: Array) -> Array:
  return _quaternion_rotate_raw(q, v)

def _quaternion_rotate_fwd(q: Array, v: Array) -> Array:
  return _quaternion_rotate(q, v), (q, v)

def _quaternion_rotate_bwd(res, g: Array) -> Tuple[Array, Array]:
  q, v = res
  _, vjp_fn = jax.vjp(_quaternion_rotate_raw, q, v)
  dq, dv = vjp_fn(g)
  return dq - (q @ dq) * q, dv

_quaternion_rotate.defvjp(_quaternion_rotate_fwd, _quaternion_rotate_bwd)


def _random_quaternion(key: KeyArray, dtype: DType) -> Array:
  """Generate a random quaternion of a given dtype."""
  rnd = random.uniform(key, (3,), minval=0.0, maxval=1.0, dtype=dtype)

  r1 = jnp.sqrt(1.0 - rnd[0])
  r2 = jnp.sqrt(rnd[0])
  pi2 = jnp.pi * 2.0
  t1 = pi2 * rnd[1]
  t2 = pi2 * rnd[2]
  return jnp.array([jnp.cos(t2) * r2,
                    jnp.sin(t1) * r1,
                    jnp.cos(t1) * r1,
                    jnp.sin(t2) * r2],
                   dtype)


@dataclasses.dataclass
class Quaternion:
  """An object representing a quaternion.

  Data is stored in a vector array, but this class exposes several
  convenience features including quaternion multiplication and conjugation. It
  also changes the size property to return the number of degrees of freedom
  of the quaternion (which is three since we expect the quaternion to be
  normalized.

  Attributes:
    vec: An array containing the underlying jax.numpy representation.
  """
  vec: Array

  @property
  def size(self) -> int:
    return 3 * reduce(operator.mul, self.vec.shape[:-1], 1)

  @property
  def ndim(self) -> Tuple[int, ...]:
    return self.vec.ndim

  def conj(self):
    return Quaternion(_quaternion_conjugate(self.vec))

  def __mul__(self, qp: 'Quaternion') -> 'Quaternion':
    return Quaternion(_quaternion_multiply(self.vec, qp.vec))

  def __rmul__(self, qp: 'Quaternion') -> 'Quaternion':
    return Quaternion(_quaternion_multiply(qp.vec, self.vec))

  def __getitem__(self, idx):
    if self.vec.ndim == 1:
      # NOTE: This will not catch the case where `idx` indexes into b
      raise ValueError('Quaternions do not support indexing into their '
                       'spatial dimension. If you want this behavior then '
                       'access the underying `vec` attribute directly.')
    return Quaternion(self.vec[idx])


def quaternion_rotate(q: Quaternion, v: Array) -> Array:
  """Rotates a vector by a given quaternion."""
  return jnp.vectorize(_quaternion_rotate, signature='(q),(d)->(d)')(q.vec, v)


def random_quaternion(key: KeyArray, dtype: DType) -> Quaternion:
  """Generate a random quaternion of a given dtype."""
  rand_quat = partial(_random_quaternion, dtype=dtype)
  rand_quat = jnp.vectorize(rand_quat, signature='(k)->(q)')
  return Quaternion(rand_quat(key))


def tree_map_no_quat(fn: Callable[..., Any], tree: Any, *rest: Any):
  """Tree map over a PyTree treating Quaternions as leaves."""
  return tree_map(fn, tree, *rest,
                  is_leaf=lambda node: isinstance(node, Quaternion))


"""Rigid body simulation functions.

This section contains classes and functions to simulate rigid bodies and to
compute various observables for simulations of rigid bodies. The structure
of the code is as follows:

1) We have the rigid body dataclass which contains the data necessary to
describe the state of rigid bodies during a simulation.

2) We have a number of helper functions to transform between the world
space reference frame and the body reference frame along with corresponding
transformations for angular momentum.

3) We have code to compute various physical observables of rigid body
collections. These are analogous to several functions in the `quantity.py` file
namely: `kinetic_energy` and `temperature`.

4) Finally, we have functions that overide the single_dispatch functions in
`simulate.py` that allow NVE and NVT simulations to work with rigid bodies.
"""


@dataclasses.dataclass
class RigidBody:
  """Defines a body described by a position and orientation.

  One subtlety about the use of RigidBody objects in JAX MD is that they
  are used to describe several different related concepts. In general the
  `RigidBody` object contains two pieces of data: the `center` containing
  information about the center of mass of the body and `orientation` containing
  information about the orientation of the body. In practice, this means that
  `RigidBody` objects are used to describe a number quantities that all have a
  center-of-mass and orientational components.

  For example, the instantaneous state of a rigid body might be described by a
  `RigidBody` containing center-of-mass position and orientation. The momentum
  of the body will be described by a `RigidBody` containing the linear momentum
  and the angular momentum. The force on the body will be described by a
  `RigidBody` containing linear force and torque. Finally, the mass of the body
  will be described by a `RigidBody` containing the total mass and the angular
  momentum.

  When used in conjunction with automatic differentiation or simulation
  environments, forces and velocities will also be of type `RigidBody`. In
  these cases the orientation should be interpreted as torque and angular
  momentum respectively.

  Attributes:
    center: An array of two- or three-dimensional positions giving the center
      position of the body.
    orientation: In two-dimensions this will be an array of angles. In three-
      dimensions this will be a set of quaternions.
  """

  center: Array
  orientation: Union[Array, Quaternion]

  def __getitem__(self, idx):
    return RigidBody(self.center[idx], self.orientation[idx])


util.register_custom_simulation_type(RigidBody)


@partial(jnp.vectorize, signature='(d)->(k,k)')
def _space_to_body_rotation(q: Array) -> Array:
  q2 = q ** 2
  w, x, y, z = q
  w2, x2, y2, z2 = q2
  return jnp.array([
    [w2 + x2 - y2 - z2, 2 * (x * y + w * z), 2 * (x * z - w * y)],
    [2 * (x * y - w * z), w2 - x2 + y2 - z2, 2 * (y * z + w * x)],
    [2 * (x * z + w * y), 2 * (y * z - w * x), w2 - x2 - y2 + z2]
  ], q.dtype)


def space_to_body_rotation(q: Quaternion) -> Array:
  """Returns an affine transformation from world space to the body frame."""
  return _space_to_body_rotation(q.vec)


@partial(jnp.vectorize, signature='(d)->(d,d)')
def _S(q: Array) -> Array:
  return jnp.array([
      [q[0], -q[1], -q[2], -q[3]],
      [q[1], q[0], -q[3], q[2]],
      [q[2], q[3], q[0], -q[1]],
      [q[3], -q[2], q[1], q[0]]
  ], q.dtype)


def S(q: Quaternion) -> Array:
  """From Miller III et al., S(q) is defined so that \dot q = 1/2S(q)\omega.

  Thus S(q) is the affine transformation that relates time derivatives of
  quaternions to angular velocities.
  """
  return _S(q.vec)


def conjugate_momentum_to_angular_momentum(orientation: Quaternion,
                                           momentum: Quaternion
                                           ) -> Array:
  """Convert from the conjugate momentum of a quaternion to angular momentum.

  Simulations involving quaternions typically proceed by integrating Hamilton's
  equations with an extended Hamiltonian,
    H(p, q) = 1/8 p^T S(q) D S(q)^T p + \phi(q)
  where q is the orientation and p is the conjugate momentum variable.
  Note (!!) unlike in problems involving only positional degrees of freedom, it
  is not the case here that dq/dt = p / m. The conjugate momentum is defined
  only by the Legendre transformation.

  This means that you cannot compute the angular velocity by simply
  transforming the conjugate momentum as you would the time-derivative of q.
  Compare, for example equation (2.13) and (2.15) in [1].

  [1] Symplectic quaternion scheme for biophysical molecular dynamics
  Miller, Eleftheriou, Pattnaik, Ndirango, Newns, and Martyna
  J. Chem. Phys. 116 20 (2002)
  """
  # NOTE: Here we are stripping the zeroth component of the angular moment.
  # however, it would be good to add a test that this is explicitly zero.
  @partial(jnp.vectorize, signature='(d),(d)->(k)')
  def wrapped_fn(q: Array, m: Array) -> Array:
    return (0.5 * _S(q).T @ m)[1:]
  return wrapped_fn(orientation.vec, momentum.vec)


def angular_momentum_to_conjugate_momentum(orientation: Quaternion,
                                           omega: Array
                                           ) -> Quaternion:
  """Transforms angular momentum vector to a conjugate momentum quaternion."""
  @partial(jnp.vectorize, signature='(d),(k)->(d)')
  def wrapped_fn(q: Array, o: Array) -> Array:
    o = jnp.concatenate((jnp.zeros((1,), dtype=q.dtype), o))
    return 2 * _S(q) @ o
  return Quaternion(wrapped_fn(orientation.vec, omega))


"""Quantity Functions.

These functions are analogues of functions in `quantity.py` except that they
work with RigidBody objects rather than linear positions / velocities.
"""


def canonicalize_momentum(position: RigidBody, momentum: RigidBody
                          ) -> RigidBody:
  """Convert quaternion conjugate momentum to angular momentum."""
  orientation = position.orientation
  p = momentum.orientation
  if isinstance(orientation, Quaternion):
    p = conjugate_momentum_to_angular_momentum(orientation, p)
  return RigidBody(momentum.center, p)


def kinetic_energy(position: RigidBody, momentum: RigidBody, mass: RigidBody
                   ) -> float:
  """Computes the kinetic energy of a system with some momenta."""
  momentum = canonicalize_momentum(position, momentum)
  ke = tree_map(lambda m, p: 0.5 * util.high_precision_sum(p**2 / m),
                mass, momentum)
  return tree_reduce(operator.add, ke, 0.0)


def temperature(position: RigidBody, momentum: RigidBody, mass: RigidBody
                ) -> float:
  """Computes the temperature of a system with some momenta."""
  dof = quantity.count_dof(momentum)
  momentum = canonicalize_momentum(position, momentum)
  ke = tree_map(lambda m, p: util.high_precision_sum(p**2 / m) / dof,
                mass, momentum)
  return tree_reduce(operator.add, ke, 0.0)


def get_moment_of_inertia_diagonal(I: Array, eps=1e-5):
  """Raises a ValueError if the moment of inertia tensor is not diagonal."""
  I_diag = vmap(jnp.diag)(I)
  # NOTE: Here epsilon has to take into account numerical error from
  # diagonalization. Maybe there's a more systematic way to figure this
  # out. It might also be worth always trying to diagonalize at float64.
  # NOTE: This will not work if the moment of inertia tensor is not known
  # ahead of a JIT. Maybe worth removing this check and relying on the fact
  # that helper functions always diagonalize the moment of inertia.
  try:
    if jnp.any(jnp.abs(I - vmap(jnp.diag)(I_diag)) > eps):
      max_dev = jnp.max(jnp.abs(I - vmap(jnp.diag)(I_diag)))
      raise ValueError('Expected diagonal moment of inertia.'
                       f'Maximum deviation: {max_dev}. Tolerance: {eps}.')
  except jax.errors.ConcretizationTypeError:
    logging.info('Skipping moment of inertia diagonalization check inside of'
                 'JIT. Make sure your moment of inertia is diagonal.')
  return I_diag


"""Simulation Single Dispatch Extension Functions.

This code overides the core simulation functions in `simulate.py` to allow
simulations to work with RigidBody objects. See `simulate.py` for a detailed
description of the use of single dispatch in simulation functions.

These functions are based on Miller III et al [1], which uses the
Suzuki-Trotter decomposition to identify a factorization of the Liouville
operator for Rigid Body motion. This factorization is compatible with either
the NVE or NVT ensemble (but is not compatible with NPT).
"""


@quantity.count_dof.register
def _(position: RigidBody) -> int:
  sizes = tree_map_no_quat(lambda x: x.size, position)
  return tree_reduce(lambda accum, x: accum + x, sizes, 0)


@simulate.initialize_momenta.register
def _(R: RigidBody, mass: RigidBody, key: Array, kT: float):
  center_key, angular_key = random.split(key)

  P_center = jnp.sqrt(mass.center * kT) * random.normal(center_key,
                                                        R.center.shape,
                                                        dtype=R.center.dtype)
  P_center = P_center - jnp.mean(P_center, axis=0, keepdims=True)

  # A the moment we assume that rigid body objects are either 2d or 3d. At some
  # point it might be worth expanding this definition to include other kinds of
  # oriented bodies.
  if isinstance(R.orientation, Quaternion):
    scale = jnp.sqrt(mass.orientation * kT)
    center = R.center
    P_angular = scale * random.normal(angular_key,
                                      center.shape,
                                      dtype=center.dtype)
    P_orientation = angular_momentum_to_conjugate_momentum(R.orientation,
                                                           P_angular)
  else:
    scale = jnp.sqrt(mass.orientation * kT)
    shape, dtype = R.orientation.shape, R.orientation.dtype
    P_orientation = scale * random.normal(angular_key, shape, dtype=dtype)

  return RigidBody(P_center, P_orientation)


def _rigid_body_3d_update(shift_fn: ShiftFn, m_rot: int) -> UpdateFn:
  """A symplectic update function for 3d rigid bodies."""
  def com_update_fn(R, P, F, M, dt, **kwargs):
    return shift_fn(R, P * dt / M, **kwargs)

  def free_rotor(k, dt, quat, p_quat, M):
    delta = dt / m_rot

    P = [
         vmap(lambda q: jnp.array([-q[1], q[0], q[3], -q[2]])),
         vmap(lambda q: jnp.array([-q[2], -q[3], q[0], q[1]])),
         vmap(lambda q: jnp.array([-q[3], q[2], -q[1], q[0]])),
    ]

    if M.ndim == 1:
      Mk = M[k]
    elif M.ndim == 2:
      Mk = M[:, [k]]
    else:
      raise NotImplementedError()
    zeta = delta * jnp.einsum('ij,ij->i', p_quat, P[k](quat) / (4 * Mk))
    zeta = zeta[:, None]
    quat = jnp.cos(zeta) * quat + jnp.sin(zeta) * P[k](quat)
    p_quat = jnp.cos(zeta) * p_quat + jnp.sin(zeta) * P[k](p_quat)

    return quat, p_quat

  def quaternion_update_fn(R, P, F, M, dt, **kwargs):
    dt_2 = dt / 2
    if not (isinstance(R, Quaternion) and isinstance(P, Quaternion)):
      raise ValueError('For 3d rigid bodies, orientations must be quaternions.'
                       f'Found {type(R)} for positions and {type(P)} for '
                       'momenta.')
    R = R.vec
    P = P.vec
    for _ in range(m_rot):
      R, P = free_rotor(2, dt_2, R, P, M)
      R, P = free_rotor(1, dt_2, R, P, M)
      R, P = free_rotor(0, dt, R, P, M)
      R, P = free_rotor(1, dt_2, R, P, M)
      R, P = free_rotor(2, dt_2, R, P, M)
    return Quaternion(R), Quaternion(P)

  def update_fn(R, P, F, M, dt, **kwargs):
    R_cm = com_update_fn(R.center, P.center, F.center, M.center, dt, **kwargs)
    R_or, P_or = quaternion_update_fn(
      R.orientation,
      P.orientation,
      F.orientation,
      M.orientation,
      dt,
      **kwargs
    )
    R = dataclasses.replace(R, center=R_cm, orientation=R_or)
    P = dataclasses.replace(P, orientation=P_or)
    return R, P, F, M

  return update_fn


def _rigid_body_2d_update(shift_fn: ShiftFn) -> UpdateFn:
  """A symplectic update function for 2d rigid bodies."""
  def com_update_fn(R, P, F, M, dt, **kwargs):
    return shift_fn(R, P * dt / M, **kwargs)

  def orientation_update_fn(R, P, F, M, dt, **kwargs):
    return R + dt * P / M

  def update_fn(R, P, F, M, dt, **kwargs):
    R_cm = com_update_fn(R.center, P.center, F.center, M.center, dt, **kwargs)
    R_or = orientation_update_fn(
      R.orientation,
      P.orientation,
      F.orientation,
      M.orientation,
      dt,
      **kwargs
    )
    return dataclasses.replace(R, center=R_cm, orientation=R_or), P, F, M

  return update_fn


@simulate.inner_update_fn.register
def _(R: RigidBody, shift_fn: Callable[..., Array], m_rot=1, **unused_kwargs):
  if isinstance(R.orientation, Quaternion):
    return _rigid_body_3d_update(shift_fn, m_rot=m_rot)
  else:
    return _rigid_body_2d_update(shift_fn)


@simulate.canonicalize_mass.register
def _(mass: RigidBody) -> RigidBody:
  if len(mass.center) == 1:
    return RigidBody(mass.center[0], mass.orientation)
  elif len(mass.center) > 1:
    return RigidBody(mass.center[:, None], mass.orientation)
  raise NotImplementedError(
    'Center of mass must be either a scalar or a vector. Found an array of '
    f'shape {mass.center.shape}.')


@simulate.kinetic_energy.register
def _(R: RigidBody, P: RigidBody, M: RigidBody) -> Array:
  return kinetic_energy(R, P, M)


"""Rigid bodies as unions of point-like particles.

All of the preceding code is valid for any rigid body. Now, we provide a set
of tools for easily constructing energy functions for one class of rigid
bodies. In particular, we provide utilities for defining rigid bodies as
rigid unions of point-like particles. These point-like particles can have
arbitrary interactions between them (which we refer to as the point-species).
Additionally, different rigid point unions can be put into the same simulation.

The rigid point union is synonymous with the shape of the body. Of course this
represents a small subset of the total possible set of rigid body potentials
and it would be interesting to explore other possibilities.
"""


@dataclasses.dataclass
class RigidPointUnion:
  """.. _rigid_body_union:

  Defines a rigid collection of point-like masses glued together.

  This class describes a rigid body as a collection of point-like particles
  rigidly arranged in space. These points can have variable masses. Rigid
  bodies interact by specifying well-defined pair potentials between the
  different points. This is a common model for rigid molecules and colloids.

  To avoid a singularity in the case of a rigid body with a single point, the
  particles are represented by disks in two-dimensions and spheres in
  three-dimensions so that each point-mass has a moment of inertia,
  :math:`I_{disk} = r^2/2` in two-dimensions and :math:`I_{sphere} = 2r^2/5`
  in three-dimensions.

  Each point can optionally be described by an integer specifying its species
  (that we will refer to as a "point species"). Different point species
  typically interact with different potential parameters.

  Additionally, this class can store multiple different shapes packed together
  that get referenced by a "shape species". In this case `total_points` refers
  to the total number of points among all the shapes while `shape_count` refers
  to the number of different kinds of shapes.

  Attributes:
    points: An array of shape `(total_points, spatial_dim)` specifying the
      position of the points making up each rigid shape.
    masses: An array of shape `(total_points,)` specifying the mass of each
      point in the union.
    point_count: An array of shape `(shape_count,)` specifying the number of
      points in each shape.
    point_offset: An array of shape `(shape_count,)` specifying the starting
      index in the `points` array for each shape.
    point_species: An optional array of shape `(total_points,)` specifying
      the species of each point making up the rigid shape.
    point_radius: A float specifying the radius for the disk / sphere used
      in computing the moment of inertia for each point-like particle.
  """

  points: Array
  masses: Array
  point_count: Array
  point_offset: Array
  point_species: Optional[Array] = None
  point_radius: float = f32(0.5)

  def dimension(self) -> int:
    """Returns the spatial dimension of the shape."""
    return self.points.shape[-1]

  def _sum_over_shapes(self, x):
    shape_count = len(self.point_count)
    shape_idx = jnp.repeat(jnp.arange(shape_count), self.point_count,
                           total_repeat_length=len(self.points))
    return ops.segment_sum(x, shape_idx, shape_count,)

  def moment_of_inertia(self) -> Array:
    """Compute the moment of inertia for each shape in the collection."""
    ndim = self.dimension()
    dtype = self.points.dtype
    if ndim == 2:
      I_disk = 1 / 2 * self.point_radius ** 2
      @vmap
      def per_particle(point, mass):
        return mass * ((point[0] ** 2 + point[1] ** 2) + I_disk)
      return self._sum_over_shapes(per_particle(self.points, self.masses))
    elif ndim == 3:
      I_sphere = 2 / 5 * self.point_radius ** 2
      @vmap
      def per_particle(point, mass):
        Id = jnp.eye(3, dtype=dtype)
        diagonal = jnp.sum(point**2) * Id
        off_diagonal = point[:, None] * point[None, :]
        return mass * ((diagonal - off_diagonal) + Id * I_sphere)
      return self._sum_over_shapes(per_particle(self.points, self.masses))
    else:
      raise ValueError('Rigid bodies are only defined in two- and three-'
                       'dimensions.')

  def mass(self, shape_species: Optional[Array]=None) -> RigidBody:
    """Get a RigidBody with the mass and moment of inertia for each shape.

    Arguments:
      shape_species: An optional array of integers specifying a mixture of
        different shapes. If specified then the mass object will contain
        a mass and moment of inertia for each shape in the collection.
    """
    ndim = self.dimension()
    if ndim == 2:
      if shape_species is not None:
        return RigidBody(self._sum_over_shapes(self.masses)[shape_species],
                         self.moment_of_inertia()[shape_species])

      return RigidBody(self._sum_over_shapes(self.masses),
                       self.moment_of_inertia())
    elif ndim == 3:
      # In three-dimensions, we grab the diagonal of the moment of inertia
      # assuming (and checking) that it is properly diagonalized.
      I_diag = get_moment_of_inertia_diagonal(self.moment_of_inertia())
      if shape_species is not None:
        return RigidBody(self._sum_over_shapes(self.masses)[shape_species],
                         I_diag[shape_species])
      return RigidBody(self._sum_over_shapes(self.masses), I_diag)
    raise ValueError('Rigid bodies only defined for two- and three-dimensions.'
                     f' Found {ndim}.')

  def __getitem__(self, idx: int) -> 'RigidPointUnion':
    """Extract a single shape from the collection of shapes."""
    start = self.point_offset[idx]
    end = start + self.point_count[idx]
    return RigidPointUnion(self.points[start : end],
                           self.masses[start : end],
                           jnp.array([self.point_count[idx]]),
                           jnp.array([0]),
                           None if self.point_species is None else
                           self.point_species[start : end])


def _transform_to_diagonal_frame(shape: RigidPointUnion) -> RigidPointUnion:
  """Transform points to zero center of mass and diagonal moment of inertia."""
  ndim = shape.dimension()
  assert len(shape.point_count) == 1

  if ndim == 2:
    total_mass = shape._sum_over_shapes(shape.masses[:, None] * shape.points)
    com = total_mass / shape.point_count[:, None]
    return shape.set(points=shape.points - com)
  elif ndim == 3:
    total_mass = jnp.sum(shape.masses)
    I, = shape.moment_of_inertia()

    I_diag, U = jnp.linalg.eigh(I)

    points = jnp.einsum('ni,ij->nj', shape.points, U)
    return RigidPointUnion(points,
                          shape.masses,
                          shape.point_count,
                          shape.point_offset)

  raise ValueError('Rigid bodies only defined for two- or three-dimensions'
                   f' found shape of dimension={ndim}.')


def point_union_shape(points: Array, masses: Array) -> RigidPointUnion:
  """Construct a rigid body out of points and masses.

  See :ref:`rigid_body_union` for details.

  Arguments:
    points: An array point point positions.
    masses: An array of particle masses.

  Returns:
    A RigidPointUnion shape object specifying the shape rotated so that the
    moment of inertia tensor is diagonal.
  """
  if jnp.isscalar(masses) or masses.shape == ():
    masses = masses * jnp.ones((len(points),), points.dtype)
  shape = RigidPointUnion(points=points,
                          masses=masses,
                          point_count=jnp.array([len(points)]),
                          point_offset=jnp.array([0]))
  return _transform_to_diagonal_frame(shape)


def concatenate_shapes(*shapes) -> RigidPointUnion:
  """Concatenate a list of RigidPointUnions into a single RigidPointUnion."""
  shape_tuples = zip(*[dataclasses.astuple(s) for s in shapes])
  points, masses, point_count, point_offset, point_species, _ = shape_tuples
  any_point_species = any(x is not None for x in point_species)
  if any_point_species and not all(x is not None for x in point_species):
    raise ValueError('Either all shapes should have point species or none '
                     'should have point species.')
  if (any_point_species and
      not all(isinstance(x, (Array, onp.ndarray)) for x in point_species)):
    raise ValueError('All point species should be specified as `onp.ndarray` '
                     'since the species must be known statically at compile '
                     'time.')
  point_count = jnp.concatenate(point_count)
  return RigidPointUnion(
      points=jnp.concatenate(points),
      masses=jnp.concatenate(masses),
      point_species=(None if point_species[0] is None
                     else jnp.concatenate(point_species)),
      point_count=point_count,
      point_offset=jnp.concatenate([jnp.array([0]),
                                    jnp.cumsum(point_count)[:-1]])
  )


# Change of Basis Transformations (Rigid Body Frame to World Frame)


@partial(jnp.vectorize, signature='()->(d,d)')
def rotation2d(theta: Array) -> Array:
  """Builds a two-dimensional rotation matrix from an angle."""
  return jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                    [jnp.sin(theta),  jnp.cos(theta)]])


def transform(body: RigidBody, shape: RigidPointUnion) -> Array:
  """Transform a rigid point union from body frame to world frame."""
  if isinstance(body.orientation, Quaternion):
    offset = quaternion_rotate(body.orientation, shape.points)
  else:
    offset = space.raw_transform(rotation2d(body.orientation), shape.points)
  return body.center[None, :] + offset


def union_to_points(body: RigidBody,
                    shape: RigidPointUnion,
                    shape_species: Optional[onp.ndarray]=None
                    ) -> Tuple[Array, Optional[Array]]:
  """Transforms points in a RigidPointUnion to world space."""
  if shape_species is None:
    position = vmap(transform, (0, None))(body, shape)
    point_species = shape.point_species
    if point_species is not None:
      point_species = shape.point_species[None, :]
      point_species = jnp.broadcast_to(point_species, position.shape[:-1])
      point_species = jnp.reshape(point_species, (-1,))
    position = jnp.reshape(position, (-1, position.shape[-1]))
    return position, point_species
  elif isinstance(shape_species, onp.ndarray):
    shape_species_types = onp.unique(shape_species)
    shape_species_count = len(shape_species_types)
    assert (len(shape.point_count) == shape_species_count and
            onp.max(shape_species_types) == shape_species_count - 1 and
            onp.min(shape_species_types) == 0)
    shape = tree_map(lambda x: onp.array(x), shape)

    point_position = []
    point_species = []

    for s in range(shape_species_count):
      cur_shape = shape[s]
      pos = vmap(transform, (0, None))(body[shape_species == s], cur_shape)

      ps = cur_shape.point_species
      if ps is not None:
        ps = cur_shape.point_species[None, :]
        ps = jnp.broadcast_to(ps, pos.shape[:-1])
        point_species += [jnp.reshape(ps, (-1,))]

      pos = jnp.reshape(pos, (-1, pos.shape[-1]))
      point_position += [pos]
    point_position = jnp.concatenate(point_position)
    point_species = jnp.concatenate(point_species) if point_species else None
    return point_position, point_species
  else:
    raise NotImplementedError('Shape species must either be None or of type '
                              'onp.ndarray since it must be specified ahead '
                              f'of compilation. Found {type(shape_species)}.')

# Energy Functions


def point_energy(energy_fn: Callable[..., Array],
                 shape: RigidPointUnion,
                 shape_species: Optional[onp.ndarray]=None
                 ) -> Callable[..., Array]:
  """Produces a RigidBody energy given a pointwise energy and a point union.

  This function takes takes a pointwise energy function that computes the
  energy of a set of particle positions along with a RigidPointUnion
  (optionally with shape species information) and produces a new energy
  function that computes the energy of a collection of rigid bodies.

  Args:
    energy_fn: An energy function that takes point positions and produces a
      scalar energy function.
    shape: A RigidPointUnion shape that contains one or more shapes defined as
      a union of point masses.
    shape_species: An optional array specifying the composition of the system
      in terms of shapes.

  Returns:
    An energy function that takes a `RigidBody` and produces a scalar energy
    energy.
  """
  def wrapped_energy_fn(body, **kwargs):
    pos, point_species = union_to_points(body, shape, shape_species)
    if point_species is None:
      return energy_fn(pos, **kwargs)
    return energy_fn(pos, species=point_species, **kwargs)
  return wrapped_energy_fn


def point_energy_neighbor_list(energy_fn: Callable[..., Array],
                               neighbor_fn: NeighborListFns,
                               shape: RigidPointUnion,
                               shape_species: Optional[onp.ndarray]=None
                               ) -> Tuple[NeighborListFns,
                                          Callable[..., Array]]:
  """Produces a RigidBody energy given a pointwise energy and a point union.

  This function takes takes a pointwise energy function that computes the
  energy of a set of particle positions using neighbor lists, a `neighbor_fn`
  that builds and updates neighbor lists (see `partition.py` for details),
  along with a RigidPointUnion (optionally with shape species information) and
  produces a new energy function that computes the energy of a collection of
  rigid bodies using neighbor lists and a `neighbor_fn` that is responsible for
  building and updating the neighbor lists.

  Args:
    energy_fn: An energy function that takes point positions along with a set
      of neighbors and produces a scalar energy function.
    neighbor_fn: A neighbor list function that creates and updates a neighbor
      list among points.
    shape: A RigidPointUnion shape that contains one or more shapes defined as
      a union of point masses.
    shape_species: An optional array specifying the composition of the system
      in terms of shapes.

  Returns:
    An energy function that takes a `RigidBody` and produces a scalar energy
    energy.
  """

  def wrapped_energy_fn(body, neighbor, **kwargs):
    pos, species = union_to_points(body, shape, shape_species)
    return energy_fn(pos, neighbor=neighbor, species=species, **kwargs)

  def neighbor_allocate_fn(body, **kwargs):
    pos, species = union_to_points(body, shape, shape_species)
    nbrs = neighbor_fn.allocate(pos, **kwargs)
    nbrs = dataclasses.replace(nbrs, update_fn=neighbor_update_fn)
    return nbrs

  def neighbor_update_fn(body, neighbor, **kwargs):
    pos, species = union_to_points(body, shape, shape_species)
    return neighbor_fn.update(pos, neighbor, **kwargs)

  wrapped_neighbor_fns = partition.NeighborListFns(neighbor_allocate_fn,
                                                    neighbor_update_fn)

  return wrapped_neighbor_fns, wrapped_energy_fn


# Predefined RigidPointUnion shapes.

# 2D Shapes.
monomer = point_union_shape(onp.array([[0.0, 0.0]], f32), f32(1.0))
dimer = point_union_shape(onp.array([[0.0, 0.5], [0.0, -0.5]], f32), f32(1.0))
trimer = point_union_shape(
    onp.array([[0, onp.sqrt(1 - 0.5 ** 2) - 0.5],
               [0.5, -0.5],
               [-0.5, -0.5]], f32),
    f32(1.0))
square = point_union_shape(
    onp.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], f32),
    f32(1.0))

# 3D Shapes.
tetrahedron = point_union_shape(
  onp.array([[1.0, 1.0, 1.0],
             [ 1.0, -1.0, -1.0],
             [-1.0,  1.0, -1.0],
             [-1.0, -1.0, 1.0]], f32) * f32(0.5),
    f32(1.0))
octohedron = point_union_shape(
  onp.array([[1.0, 0.0, 0.0],
             [-1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, -1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, -1.0]], f32) * f32(0.5),
    f32(1.0))
