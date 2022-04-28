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


from typing import Optional, Tuple, Any, Union, Callable

import numpy as onp

import jax
from jax import vmap
from jax import ops
from jax import random
import jax.numpy as jnp
from jax_md import dataclasses, util, space, partition, quantity, simulate
from functools import partial

from jax.tree_util import tree_map, tree_reduce

import operator


DType = Any
Array = util.Array
PyTree = Any
f64 = util.f64
f32 = util.f32
KeyArray = random.KeyArray
NeighborListFns = partition.NeighborListFns
ShiftFn = Any


# Quaternion Utilities


def is_float64(x: Array) -> bool:
  return x.dtype in [jnp.float64, onp.float64]


@partial(jnp.vectorize, signature='(q),(q)->(q)')
def _quaternion_multiply(lhs: Array, rhs: Array) -> Array:
  wl, xl, yl, zl = lhs
  wr, xr, yr, zr = rhs

  dtype = f64 if is_float64(lhs) or is_float64(rhs) else f32

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


@partial(jnp.vectorize, signature='(q),(d)->(d)')
def _quaternion_apply(q: Array, v: Array) -> Array:
  if q.shape != (4,):
    raise ValueError('')
  if v.shape != (3,):
    raise ValueError('')

  v = jnp.concatenate([jnp.zeros((1,), v.dtype), v])
  q = _quaternion_multiply(q, _quaternion_multiply(v, _quaternion_conjugate(q)))
  return q[1:]


@dataclasses.dataclass
class Quaternion:
  vec: Array

  @property
  def size(self) -> int:
    return 3 * reduce(operator.mul, data.shape[:-1], 1)

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
    # TODO: Better error message.
    assert self.vec.ndim > 1
    return Quaternion(self.vec[idx])


def quaternion_apply(q: Quaternion, v: Array) -> Array:
  return _quaternion_apply(q.vec, v)


def random_quaternion(key: KeyArray, dtype: DType) -> Quaternion:
  rnd = random.uniform(key, (3,), minval=0.0, maxval=1.0, dtype=dtype)

  r1 = jnp.sqrt(1.0 - rnd[0])
  r2 = jnp.sqrt(rnd[0])
  pi2 = jnp.pi * 2.0
  t1 = pi2 * rnd[1]
  t2 = pi2 * rnd[2]
  return Quaternion(
    jnp.array([jnp.cos(t2) * r2,
               jnp.sin(t1) * r1,
               jnp.cos(t1) * r1,
               jnp.sin(t2) * r2],
              dtype)
  )


def tree_map_no_quat(fn: Callable[..., Any], tree: Any, *rest: Any):
  return tree_map(fn, tree, *rest, lambda node: isinstance(node, Quaternion))


# Rigid Body Types


@dataclasses.dataclass
class RigidBody:
  center: Array
  orientation: Union[Array, Quaternion]

  @property
  def moment_of_inertia(self) -> Union[Array, Quaternion]:
    return self.orientation

  def __getitem__(self, idx):
    return RigidBody(self.center[idx], self.orientation[idx])


@dataclasses.dataclass
class RigidBodyShape:
  points: Array        # (total_points, spatial_dim)
  masses: Array        # (total_points,)
  point_count: Array   # (number_of_body_types,)
  point_offset: Array  # (number_of_body_types,)
  point_species: Optional[Array] = None  # (total_points,)
  point_radius: float = f32(0.5)

  def dimension(self):
    return self.points.shape[-1]

  def sum_over_shapes(self, x):
    shape_count = len(self.point_count)
    shape_idx = jnp.repeat(jnp.arange(shape_count), self.point_count)
    return ops.segment_sum(x, shape_idx, shape_count)

  def moment_of_inertia(self):
    ndim = self.dimension()
    if ndim == 2:
      I_disk = 1 / 2 * self.point_radius ** 2
      @vmap
      def per_particle(point, mass):
        return mass * (point[0] ** 2 + point[1] ** 2) + I_disk
      return self.sum_over_shapes(per_particle(self.points, self.masses))
    elif ndim == 3:
      I_sphere = 2 / 5 * self.point_radius ** 2
      @vmap
      def per_particle(point, mass):
        diagonal = jnp.linalg.norm(point) ** 2 * jnp.eye(point.shape[-1], dtype=point.dtype)
        off_diagonal = point[:, None] * point[None, :]
        return mass * ((diagonal - off_diagonal) + jnp.eye(3, dtype=point.dtype) * I_sphere)
      return self.sum_over_shapes(per_particle(self.points, self.masses))
    else:
      raise ValueError('Rigid bodies are only defined in two- and three-'
                       'dimensions.')

  def mass(self, species: Optional[Array]=None):
    ndim = self.dimension()
    if ndim == 2:
      if species is not None:
        return RigidBody(self.sum_over_shapes(self.masses)[species],
                         self.moment_of_inertia()[species])

      return RigidBody(self.sum_over_shapes(self.masses),
                       self.moment_of_inertia())
    elif ndim == 3:
      I = self.moment_of_inertia()
      I_diag = vmap(jnp.diag)(I)
      # NOTE: Here epsilon has to take into account numerical error from
      # diagonalization. Maybe there's a more systematic way to figure this
      # out. It might also be worth always trying to diagonalize at float64.
      eps = 1e-5
      if jnp.any(jnp.abs(I - vmap(jnp.diag)(I_diag)) > eps):
        max_dev = jnp.max(jnp.abs(I - vmap(jnp.diag)(I_diag)))
        raise ValueError('Expected diagonal moment of inertia.'
                         f'Maximum deviation: {max_dev}. Tolerance: {eps}.')
      if species is not None:
        return RigidBody(self.sum_over_shapes(self.masses)[species],
                         I_diag[species])
      return RigidBody(self.sum_over_shapes(self.masses), I_diag)
    raise ValueError()

  def radius(self):
    return jnp.max(jnp.linalg.norm(self.points, axis=-1))

  def __getitem__(self, idx):
    start = self.point_offset[idx]
    end = start + self.point_count[idx]
    return RigidBodyShape(self.points[start : end],
                          self.masses[start : end],
                          jnp.array([self.point_count[idx]]),
                          jnp.array([0]),
                          None if self.point_species is None else
                          self.point_species[start : end])


def transform_to_diagonal_frame(shape: RigidBodyShape) -> RigidBodyShape:
  ndim = shape.dimension()
  assert len(shape.point_count) == 1

  if ndim == 2:
    total_mass = shape.sum_over_shapes(shape.masses[:, None] * shape.points)
    com = total_mass / shape.point_count[:, None]
    return shape.set(points=shape.points - com)
  elif ndim == 3:
    total_mass = jnp.sum(shape.masses)
    I, = shape.moment_of_inertia()

    I_diag, U = jnp.linalg.eigh(I)

    points = jnp.einsum('ni,ij->nj', shape.points, U)
    return RigidBodyShape(points,
                          shape.masses,
                          shape.point_count,
                          shape.point_offset)

  raise ValueError('Rigid bodies only defined for two- or three-dimensions'
                   f' found shape of dimension={ndim}.')


def rigid_body_shape(points: Array, masses: Array) -> RigidBodyShape:
  if jnp.isscalar(masses) or masses.shape == ():
    masses = masses * jnp.ones((len(points),), points.dtype)
  shape = RigidBodyShape(points=points,
                         masses=masses,
                         point_count=jnp.array([len(points)]),
                         point_offset=jnp.array([0]))
  return transform_to_diagonal_frame(shape)


def concatenate_shapes(*shapes):
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
  return RigidBodyShape(
      points=jnp.concatenate(points),
      masses=jnp.concatenate(masses),
      point_species=(None if point_species[0] is None
                     else jnp.concatenate(point_species)),
      point_count=point_count,
      point_offset=jnp.concatenate([jnp.array([0]),
                                    jnp.cumsum(point_count)[:-1]])
  )

# TODO: Maybe move the shapes somewhere else (end of the file?)
# 2D Shapes
monomer = rigid_body_shape(onp.array([[0.0, 0.0]], f32), f32(1.0))
dimer = rigid_body_shape(onp.array([[0.0, 0.5], [0.0, -0.5]], f32), f32(1.0))
trimer = rigid_body_shape(
    onp.array([[0, onp.sqrt(1 - 0.5 ** 2) - 0.5],
               [0.5, -0.5],
               [-0.5, -0.5]], f32),
    f32(1.0))
square = rigid_body_shape(
    onp.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], f32),
    f32(1.0))

# 3D Shapes
tetrahedron = rigid_body_shape(
  onp.array([[1.0, 1.0, 1.0],
             [ 1.0, -1.0, -1.0],
             [-1.0,  1.0, -1.0],
             [-1.0, -1.0, 1.0]], f32) * f32(0.5),
    f32(1.0))
octohedron = rigid_body_shape(
  onp.array([[1.0, 0.0, 0.0],
             [-1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, -1.0, 0.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, -1.0]], f32) * f32(0.5),
    f32(1.0))


# Change of Basis Transformations (Rigid Body Frame to World Frame)


@partial(jnp.vectorize, signature='()->(d,d)')
def rotation(theta: Array) -> Array:
  return jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                    [jnp.sin(theta),  jnp.cos(theta)]])


@jax.custom_vjp
def _transform3d(body: Tuple[Array, Quaternion],
                 shape: RigidBodyShape
                 ) -> Array:
  center, orientation = body
  return center[None, :] + quaternion_apply(orientation, shape.points)


def _transform3d_fwd(body: Tuple[Array, Quaternion], shape: RigidBodyShape
                     ) -> Tuple[Array, Tuple[Quaternion, RigidBodyShape]]:
  center, orientation = body
  return (_transform3d((center, orientation), shape), (orientation, shape))


def _transform3d_bwd(res: Tuple[Array, RigidBodyShape], F_particle: Array
                     ) -> Tuple[Tuple[Array, Array], RigidBodyShape]:
  orientation, shape = res
  F_com = jnp.sum(F_particle, axis=0)

  mass = shape.masses[:, None]
  com_mass = jnp.sum(mass)

  F_space = F_particle - F_com[None, :] * (mass / com_mass)
  A_body = space_to_body_rotation(orientation)
  F_body = jnp.einsum('jk,ak->aj', A_body, F_space)

  tau3 = jnp.sum(jnp.cross(shape.points, F_body), axis=0)
  tau4 = jnp.concatenate((jnp.zeros((1,), F_particle.dtype), tau3))

  F_quat = 2 * S(orientation) @ tau4

  _, vjp_fn = jax.vjp(partial(quaternion_apply, orientation), shape.points)

  return (F_com, F_quat), RigidBodyShape(vjp_fn(F_particle)[0],
                                         jnp.zeros_like(shape.masses),
                                         shape.point_count,
                                         shape.point_offset,
                                         shape.point_species,
                                         shape.point_radius)
_transform3d.defvjp(_transform3d_fwd, _transform3d_bwd)


def transform3d(body: RigidBody, shape: RigidBodyShape) -> Array:
  return _transform3d((body.center, body.orientation), shape)


def transform(body: RigidBody, shape: RigidBodyShape) -> Array:
  if isinstance(body.orientation, Quaternion):
    return transform3d(body, shape)
  else:
    offset = space.raw_transform(rotation(body.orientation), shape.points)
    return body.center[None, :] + offset


def bodies_to_points(body: RigidBody,
                     shape: RigidBodyShape,
                     shape_species: Optional[onp.ndarray]=None
                     ) -> Tuple[Array, Optional[Array]]:
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
  return _space_to_body_rotation(q.vec)


@partial(jnp.vectorize, signature='(d)->(d,d)')
def _S(q: Array) -> Array:
  """From Miller III et al., S(q) is defined so that \dot q = 1/2S(q)\omega.

  If we were to include the 1/2 in the definition of S (which we might do) then
  it would be the matrix that takes you from the tangents in the body frame
  (angular velocites) to tangents in quaternions (space frame?)
  """
  return jnp.array([
      [q[0], -q[1], -q[2], -q[3]],
      [q[1], q[0], -q[3], q[2]],
      [q[2], q[3], q[0], -q[1]],
      [q[3], -q[2], q[1], q[0]]
  ], q.dtype)


def S(q: Quaternion) -> Array:
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

  This means that you cannot compute the angular velocity by simply transforming
  the conjugate momentum as you would the time-derivative of q. Compare, for
  example equation (2.13) and (2.15) in [1].

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
  @partial(jnp.vectorize, signature='(d),(k)->(d)')
  def wrapped_fn(q: Array, o: Array) -> Array:
    o = jnp.concatenate((jnp.zeros((1,), dtype=q.dtype), o))
    return 0.5 * _S(q) @ o
  return Quaternion(wrapped_fn(orientation.vec, omega))


# Energy Functions


def energy(energy_fn: Callable[..., Array],
           shape: RigidBodyShape,
           shape_species: onp.ndarray=None) -> Callable[..., Array]:
  def wrapped_energy_fn(body, **kwargs):
    pos, point_species = bodies_to_points(body, shape, shape_species)
    if point_species is None:
      return energy_fn(pos, **kwargs)
    return energy_fn(pos, species=point_species, **kwargs)
  return wrapped_energy_fn


def energy_neighbor_list(energy_fn: Callable[..., Array],
                         neighbor_fn: NeighborListFns,
                         shape: RigidBodyShape,
                         shape_species: onp.ndarray=None
                         ) -> Tuple[NeighborListFns,
                                               Callable[..., Array]]:
  def wrapped_energy_fn(body, neighbor, **kwargs):
    pos, species = bodies_to_points(body, shape, shape_species)
    return energy_fn(pos, neighbor=neighbor, species=species, **kwargs)

  def neighbor_allocate_fn(body, **kwargs):
    pos, species = bodies_to_points(body, shape, shape_species)
    nbrs = neighbor_fn.allocate(pos, **kwargs)
    nbrs = dataclasses.replace(nbrs, update_fn=neighbor_update_fn)
    return nbrs

  def neighbor_update_fn(body, neighbor, **kwargs):
    pos, species = bodies_to_points(body, shape, shape_species)
    return neighbor_fn.update(pos, neighbor, **kwargs)

  wrapped_neighbor_fns = partition.NeighborListFns(neighbor_allocate_fn,
                                                    neighbor_update_fn)

  return wrapped_neighbor_fns, wrapped_energy_fn


# Quantity Functions


def canonicalize_momentum(position: RigidBody, momentum: RigidBody) -> RigidBody:
  orientation = position.orientation
  p = momentum.orientation
  if isinstance(orientation, Quaternion):
    p = conjugate_momentum_to_angular_momentum(orientation, p)
  return RigidBody(momentum.center, p)


def kinetic_energy(position: PyTree, momentum: PyTree, mass: Array) -> float:
  momentum = canonicalize_momentum(position, momentum)
  ke = tree_map(lambda m, p: 0.5 * util.high_precision_sum(p**2 / m),
                mass, momentum)
  return tree_reduce(operator.add, ke, 0.0)


def temperature(position: PyTree,
                momentum: PyTree,
                mass: Array=f32(1.0)) -> float:
  """Computes the temperature of a system with some momenta."""
  dof = quantity.count_dof(momentum)
  momentum = canonicalize_momentum(position, momentum)
  ke = tree_map(lambda m, p: util.high_precision_sum(p**2 / m) / dof,
                mass, momentum)
  return tree_reduce(operator.add, ke, 0.0)


# Simulation Single Dispatch Extension Functions


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


def rigid_body_3d_update(shift_fn: ShiftFn, m_rot: int
                         ) -> Tuple[Callable, Callable]:
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
    # TODO: Better check for quaternions.
    if not (isinstance(R, Quaternion) and isinstance(P, Quaternion)):
      raise ValueError()
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


def rigid_body_2d_update(shift_fn):
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


@simulate.get_update_fn.register
def _(R: RigidBody, shift_fn: Callable[..., Array], m_rot=1, **unused_kwargs):
  if isinstance(R.orientation, Quaternion):
    return rigid_body_3d_update(shift_fn, m_rot=m_rot)
  else:
    return rigid_body_2d_update(shift_fn)


@quantity.canonicalize_mass.register
def _(mass: RigidBody) -> RigidBody:
  if len(mass.center) == 1:
    return RigidBody(mass.center[0], mass.orientation)
  elif len(mass.center) > 1:
    return RigidBody(mass.center[:, None], mass.orientation)
  raise NotImplementedError()

@simulate.kinetic_energy.register
def _(R: RigidBody, P: RigidBody, M: RigidBody) -> Array:
  return kinetic_energy(R, P, M)
