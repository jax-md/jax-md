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

"""Code to calculate the elastic modulus tensor for athermal systems.

  The elastic modulus tensor describes a material's response to different
  boundary deformations. Specifically, for a small deformation given by a
  symmetric strain tensor e, the change in energy is
    U / V^0 = U^0/V^0 + s^0_{ij} e_{ji} + (1/2) C_{ijkl} e_{ij} e_{kl} + ...
  where V^0 is the volume, U^0 is the initial energy, s^0 is the residual stress
  tensor of the undeformed system, and C is the elastic modulus tensor. C is
  a fourth-rank tensor of shape (dimension,dimension,dimension,dimension), with
  the following symmetries.

    Minor symmetries:
      C_ijkl = C_jikl = C_ijlk

    Major symmetries:
      C_ijkl = C_lkij

  The minor symmetries are also reflected in the symmetric nature
  of stress and strain tensors:
    s_ij = s_ji
    e_ij = e_ji

  In general, there are 21 independent elastic constants in 3 dimension (6 in 2
  dimensions). While systems with additional symmetries (e.g. isotropic,
  orthotropic, etc.) can be expressed with fewer constants, we do not assume any
  such additional symmetries.

  At zero temperature, the response of every particle to a deformation can be
  calculated explicitly to linear order, enabling the exact calculation of
  the elastic modulus tensor without the need of any finite differences or any
  approximations.

"""




from functools import partial
from typing import Dict, Callable, Union
from absl import logging

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, lax, grad, vmap, jacfwd, jacrev, jvp

from jax_md import quantity
from jax_md.util import Array
from jax_md.util import f32
from jax_md.util import f64

def _get_strain_tensor_list(dim, dtype) -> Array:
  if dim == 2:
    strain_tensors = jnp.array([[[1, 0],[0, 0]],
                                [[0, 0],[0, 1]],
                                [[0, 1],[1, 0]],
                                [[1, 0],[0, 1]],
                                [[1, 1],[1, 0]],
                                [[0, 1],[1, 1]]], dtype=dtype)
  elif dim == 3:
    strain_tensors = jnp.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                                [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                                [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
                                [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
                                [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                                [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                                [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                [[1, 0, 1], [0, 0, 0], [1, 0, 0]],
                                [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                                [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
                                [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                                [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
                                [[0, 0, 0], [0, 0, 1], [0, 1, 1]],
                                [[0, 0, 1], [0, 0, 0], [1, 0, 1]],
                                [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                                [[0, 0, 1], [0, 0, 1], [1, 1, 0]],
                                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                                [[0, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=dtype)
  else:
    raise AssertionError('not implemented for {} dimensions'.format(dim))
  return strain_tensors

def _convert_responses_to_elastic_constants(response_all: Array) -> Array:
  if response_all.shape[0] == 6:
    cxxxx = response_all[0]
    cyyyy = response_all[1]
    cxyxy = 0.25 * response_all[2]
    cxxyy = 0.5  * (response_all[3] - cxxxx - cyyyy)
    cxxxy = 0.25 * (response_all[4] - 4 * cxyxy - cxxxx)
    cyyxy = 0.25 * (response_all[5] - 4 * cxyxy - cyyyy)

    C = jnp.array(
        [[[[cxxxx, cxxxy], [cxxxy, cxxyy]],
          [[cxxxy, cxyxy], [cxyxy, cyyxy]]],
         [[[cxxxy, cxyxy], [cxyxy, cyyxy]],
          [[cxxyy, cyyxy], [cyyxy, cyyyy]]]])

  elif response_all.shape[0] == 21:
    cxxxx = response_all[0];
    cyyyy = response_all[1];
    czzzz = response_all[2];
    cyzyz = response_all[3] / 4.;
    cxzxz = response_all[4] / 4.;
    cxyxy = response_all[5] / 4.;
    cyyzz = (response_all[6] - cyyyy - czzzz) / 2.;
    cxxzz = (response_all[7] - cxxxx - czzzz) / 2.;
    cxxyy = (response_all[8] - cxxxx - cyyyy) / 2.;
    cxxyz = (response_all[9] - cxxxx - 4. * cyzyz) / 4.;
    cxxxz = (response_all[10] - cxxxx - 4. * cxzxz) / 4.;
    cxxxy = (response_all[11] - cxxxx - 4. * cxyxy) / 4.;
    cyyyz = (response_all[12] - cyyyy - 4. * cyzyz) / 4.;
    cyyxz = (response_all[13] - cyyyy - 4. * cxzxz) / 4.;
    cyyxy = (response_all[14] - cyyyy - 4. * cxyxy) / 4.;
    czzyz = (response_all[15] - czzzz - 4. * cyzyz) / 4.;
    czzxz = (response_all[16] - czzzz - 4. * cxzxz) / 4.;
    czzxy = (response_all[17] - czzzz - 4. * cxyxy) / 4.;
    cyzxz = (response_all[18] - 4. * cyzyz - 4. * cxzxz) / 8.;
    cyzxy = (response_all[19] - 4. * cyzyz - 4. * cxyxy) / 8.;
    cxzxy = (response_all[20] - 4. * cxzxz - 4. * cxyxy) / 8.;

    C = jnp.array(
         [[[[cxxxx, cxxxy, cxxxz],
            [cxxxy, cxxyy, cxxyz],
            [cxxxz, cxxyz, cxxzz]],
           [[cxxxy, cxyxy, cxzxy],
            [cxyxy, cyyxy, cyzxy],
            [cxzxy, cyzxy, czzxy]],
           [[cxxxz, cxzxy, cxzxz],
            [cxzxy, cyyxz, cyzxz],
            [cxzxz, cyzxz, czzxz]]],
          [[[cxxxy, cxyxy, cxzxy],
            [cxyxy, cyyxy, cyzxy],
            [cxzxy, cyzxy, czzxy]],
           [[cxxyy, cyyxy, cyyxz],
            [cyyxy, cyyyy, cyyyz],
            [cyyxz, cyyyz, cyyzz]],
           [[cxxyz, cyzxy, cyzxz],
            [cyzxy, cyyyz, cyzyz],
            [cyzxz, cyzyz, czzyz]]],
          [[[cxxxz, cxzxy, cxzxz],
            [cxzxy, cyyxz, cyzxz],
            [cxzxz, cyzxz, czzxz]],
           [[cxxyz, cyzxy, cyzxz],
            [cyzxy, cyyyz, cyzyz],
            [cyzxz, cyzyz, czzyz]],
           [[cxxzz, czzxy, czzxz],
            [czzxy, cyyzz, czzyz],
            [czzxz, czzyz, czzzz]]]])
  else:
    raise AssertionError('response_all has incorrect shape')
  return C

def athermal_moduli(energy_fn: Callable[..., Array],
                    tether_strength: float=1e-10,
                    gradient_check: Array=None,
                    cg_tol: float=1e-7,
                    check_convergence: bool=False
                    ) -> Callable[..., Array]:
  """ Setup calculation of elastic modulus tensor.

  Args:
    energy_fn: A function that computes the energy of the system. This
      function must take as an argument `perturbation` which perturbes the
      box shape. Any energy function constructed using `smap` or in `energy.py`
      with a standard space will satisfy this property.
    tether_strength: scalar. Strength of the "tether" applied to each particle,
      which can be necessary to make the Hessian matrix non-singular. Solving
      for the non-affine response of each particle requires that the Hessian is
      positive definite. However, there can often be zero modes (eigenvectors of
      the Hessian with zero eigenvalue) that do not couple to the boundary, and
      therefore do not affect the elastic constants despite the zero eigenvalue.
      The most common example is the global translational modes. To solve for
      the non-affine response, we consider the "tethered Hessian"
        H + tether_strength * I,
      where I is the identity matrix and tether_strength is a small constant.
    gradient_check: None or scalar. If not None, a check will be performed
      to guarantee that the maximum component of the gradient is less than
      gradient_check. In other words, that
        jnp.amax(jnp.abs(grad(energy_fn)(R, box=box))) < gradient_check == True
      NOTE: JAX currently does not support proper runtime error handling.
      Therefore, if this check fails, the calculation will return an array
      of jnp.nan's. It is the users responsibility, if they want to use this
      check, to then ensure that the returned array is not full of nans.
    cg_tol: scalar. Tolorance used when solving for the non-affine response.
    check_convergence: bool. If true, calculate_EMT will return a boolean
      flag specifiying if the cg solve routine converged to the desired
      tolorance. The default is False, but convergence checking is highly
      recommended especially when using 32-bit precision data.

  Return: A function to calculate the elastic modulus tensor

  """

  def calculate_emt(R: Array,
                    box: Array,
                    **kwargs) -> Array:
    """Calculate the elastic modulus tensor.

    energy_fn(R) corresponds to the state around which we are expanding

    Args:
      R: array of shape (N,dimension) of particle positions. This does not
        generalize to arbitrary dimensions and is only implemented for
          dimension == 2
          dimension == 3
      box: A box specifying the shape of the simulation volume. Used to infer
        the volume of the unit cell.

    Return: C or the tuple (C,converged)
      where C is the Elastic modulus tensor as an array of shape (dimension,
      dimension,dimension,dimension) that respects the major and minor
      symmetries, and converged is a boolean flag (see above).

    """
    if not (R.shape[-1] == 2 or R.shape[-1] == 3):
      raise AssertionError('Only implemented for 2d and 3d systems.')

    if R.dtype is not jnp.dtype('float64'):
      logging.warning('Elastic modulus calculations can sometimes lose '
                      'precision when not using 64-bit precision.')

    dim = R.shape[-1]

    def setup_energy_fn_general(strain_tensor):
      I = jnp.eye(dim, dtype=R.dtype)
      @jit
      def energy_fn_general(R, gamma):
        perturbation = I + gamma * strain_tensor
        return energy_fn(R, perturbation=perturbation, **kwargs)
      return energy_fn_general

    def get_affine_response(strain_tensor):
      energy_fn_general = setup_energy_fn_general(strain_tensor)
      d2U_dRdgamma = jacfwd(jacrev(energy_fn_general,argnums=0),argnums=1)(R,0.)
      d2U_dgamma2  = jacfwd(jacrev(energy_fn_general,argnums=1),argnums=1)(R,0.)
      return d2U_dRdgamma, d2U_dgamma2

    strain_tensors = _get_strain_tensor_list(dim, R.dtype)
    d2U_dRdgamma_all,d2U_dgamma2_all = vmap(get_affine_response)(strain_tensors)

    #Solve the system of equations.
    energy_fn_Ronly = partial(energy_fn, **kwargs)
    def hvp(f, primals, tangents):
      return jvp(grad(f), primals, tangents)[1]
    def hvp_specific_with_tether(v):
      return hvp(energy_fn_Ronly, (R,), (v,)) + tether_strength * v

    non_affine_response_all = jsp.sparse.linalg.cg(
        vmap(hvp_specific_with_tether),
        d2U_dRdgamma_all,
        tol=cg_tol
        )[0]
    #The above line should be functionally equivalent to:
    #H0=hessian(energy_fn)(R, box=box, **kwargs).reshape(R.size,R.size) \
    #    + tether_strength * jnp.identity(R.size)
    #non_affine_response_all = jnp.transpose(jnp.linalg.solve(
    #   H0,
    #   jnp.transpose(d2U_dRdgamma_all))
    #   )

    residual = jnp.linalg.norm(vmap(hvp_specific_with_tether)(
        non_affine_response_all) - d2U_dRdgamma_all
      )
    converged = residual / jnp.linalg.norm(d2U_dRdgamma_all) < cg_tol

    response_all = d2U_dgamma2_all - jnp.einsum("nij,nij->n",
                                                d2U_dRdgamma_all,
                                                non_affine_response_all)

    vol_0 = quantity.volume(dim, box)
    response_all = response_all / vol_0
    C = _convert_responses_to_elastic_constants(response_all)

    # JAX does not allow proper runtime error handling in jitted function.
    # Instead, if the user requests a gradient check and the check fails,
    # we convert C into jnp.nan's. While this doesn't raise an exception,
    # it at least is very "loud".
    if gradient_check is not None:
      maxgrad = jnp.amax(jnp.abs(grad(energy_fn)(R, **kwargs)))
      C = lax.cond(maxgrad > gradient_check,
                   lambda _: jnp.nan * C,
                   lambda _: C,
                   None)

    if check_convergence:
      return C, converged
    else:
      return C

  return calculate_emt


def _get_mandel_mapping_weight(dim, dtype):
  if dim == 2:
    m_map  = jnp.array([[0,0],[1,1],[0,1]], dtype=jnp.int8)
    weight = jnp.array([1,1,jnp.sqrt(2)], dtype=dtype)
    return m_map, weight
  elif dim == 3:
    m_map  = jnp.array([[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]], dtype=jnp.int8)
    weight = jnp.array([1,1,1,jnp.sqrt(2),jnp.sqrt(2),jnp.sqrt(2)], dtype=dtype)
    return m_map, weight
  else:
    raise AssertionError('dim must be 2 or 3')

def tensor_to_mandel(T: Array) -> Array:
  """ Convert a tensor to Mandel notation.

  Mandel notation is a way to represent symmetric second-rank tensors and
  fourth-rank tensors with minor symmetries in a reduced form. Pairs of indices
  are combined as follows:

      For tensors of shape (2,2) or (2,2,2,2):
      tensor indices        Mandel indices
      0,0 -------------->   0
      1,1 -------------->   1
      0,1 or 1,0 ------->   2

      For tensors of shape (3,3) or (3,3,3,3):
      tensor indices        Mandel indices
      0,0 -------------->   0
      1,1 -------------->   1
      2,2 -------------->   2
      1,2 or 2,1 ------->   3
      0,2 or 2,0 ------->   4
      0,1 or 1,0 ------->   5

  If mandel_index(i,j) performs the above index mapping, then the input T and
  output M satisfy
    M[mandel_index(i,j)] = T[i,j] * w
  or
    M[mandel_index(i,j), mandel_index(k,l)] = T[i,j,k,l] * w(i,j) * w(k,l)
  where
    w(i,j) = 1       if i==j
           = sqrt(2) if i!=j
  is a weight that is used to ensure proper contraction rules. Here (and only
  here) we do not assume major symmetries in fourth-rank tensors.

  Args:
    T: Array with 4 possible shapes:
      1. T.shape == (2,2)
         Convert a symmetric array of shape (2,2) to an array of shape (3,)
      2. T.shape == (3,3)
         Convert a symmetric array of shape (3,3) to an array of shape (6,)
      3. T.shape == (2,2,2,2)
         Convert a tensor of shape (2,2,2,2) with minor symmetries to an array
         of shape (3,3)
      4. T.shape == (3,3,3,3)
         Convert a tensor of shape (3,3,3,3) with minor symmetries to an array
         of shape (6,6)

  Output: Array of shape (3,), (6,), (3,3), or (6,6)

  see: https://sbrisard.github.io/janus/mandel.html (accessed 21 April, 2021)
  """
  dim = T.shape[0]
  if not (dim ==2 or dim == 3):
    raise AssertionError('dim must be 2 or 3')

  rank = len(T.shape)
  if not (rank == 2 or rank == 4):
    raise AssertionError('T must have rank 2 or 4')

  m_map, weight = _get_mandel_mapping_weight(dim, T.dtype)

  if rank == 2:
    extract = lambda idx, w: T[idx[0], idx[1]] * w
    M = vmap(extract, in_axes=(0,0))(m_map, weight)
  else:
    extract = lambda idx0, idx1, w0, w1: T[
        idx0[0], idx0[1], idx1[0], idx1[1]
        ] * w0 * w1
    M = vmap(
          vmap(extract,
            in_axes=(0,None,0,None)),
          in_axes=(None,0,None,0))(m_map, m_map, weight, weight)
  return M

def mandel_to_tensor(M: Array) -> Array:
  """ Perform the inverse of M = tensor_to_mandel(T).

  Args:
    M: Array of shape (3,), (6,), (3,3), or (6,6)

  Output: Array of shape (2,2), (3,3), (2,2,2,2), or (3,3,3,3)
  """
  mandel_dim = M.shape[0]
  if not (mandel_dim == 3 or mandel_dim == 6):
    raise AssertionError('M.shape[0] must be 3 or 6')

  rank = len(M.shape)
  if not (rank == 1 or rank == 2):
    raise AssertionError('T must have rank 1 or 2')

  def mandel_index(i, j):
    return lax.cond(i==j,
                    lambda ij: ij[0],
                    lambda ij: mandel_dim - ij[0] - ij[1],
                    (i,j))

  if mandel_dim == 3:
    dimension = 2
  else:
    dimension = 3

  tensor_range = jnp.arange(dimension)
  _, weight = _get_mandel_mapping_weight(dimension, M.dtype)

  if rank == 1:
    def extract(i, j):
      idx = mandel_index(i, j)
      return M[idx] / weight[idx]
    T = vmap(
          vmap(extract,
            in_axes=(0,None)),
          in_axes=(None,0))(tensor_range, tensor_range)
  else:
    def extract(i, j, k, l):
      idx0 = mandel_index(i, j)
      idx1 = mandel_index(k, l)
      return M[idx0, idx1] / (weight[idx0] * weight[idx1])
    T = vmap(
          vmap(
            vmap(
              vmap(extract,
                in_axes=(0,None,None,None)),
              in_axes=(None,0,None,None)),
            in_axes=(None,None,0,None)),
          in_axes=(None,None,None,0))(tensor_range,
                                      tensor_range,
                                      tensor_range,
                                      tensor_range)
  return T



@partial(jit,static_argnums=(1,))
def _extract_elements(C, as_dict):
  if C.shape[0] == 2:
    indices = ( [0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1],
                [0, 1, 0, 1, 0, 1],
                [0, 1, 1, 1, 1, 1])
    clist = C[ indices ]
    if as_dict:
      names = ['cxxxx','cyyyy','cxyxy','cxxyy','cxxxy','cyyxy']
      return dict(zip(names, clist))
    else:
      return clist

  elif C.shape[0] == 3:
    indices = ( [0, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2, 1, 2, 1, 1],
                [0, 1, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 0],
                [0, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2])
    clist = C[ indices ]
    if as_dict:
      names = ['cxxxx','cyyyy','czzzz','cyzyz','cxzxz','cxyxy','cyyzz','cxxzz',
               'cxxyy','cxxyz','cxxxz','cxxxy','cyyyz','cyyxz','cyyxy','czzyz',
               'czzxz','czzxy','cyzxz','cyzxy','cxzxy']
      return dict(zip(names, clist))
    else:
      return clist
  else:
    raise AssertionError('C has wrong shape')

def extract_elements(C: Array) -> Dict:
  """ Convert an elastic modulus tensor into a list of unique elements.

      In 2d, these are:
      cxxxx,cyyyy,cxyxy,cxxyy,cxxxy,cyyxy

      In 3d, these are:
      cxxxx,cyyyy,czzzz,cyzyz,cxzxz,cxyxy,cyyzz,cxxzz,cxxyy,cxxyz,cxxxz,cxxxy,
      cyyyz,cyyxz,cyyxy,czzyz,czzxz,czzxy,cyzxz,cyzxy,cxzxy

  Args:
    C: A previously calculated elastic modulus tensor represented as an
      array of shape (spatial_dimension,spatial_dimension,spatial_dimension,
      spatial_dimension), where spatial_dimension is either 2 or 3. C must
      satisfy both the major and minor symmetries, but this is not checked.
  Return: a dict of the 6 (21) unique elastic constants in 2 (3) dimensions.
  """
  return _extract_elements(C,True)

def extract_isotropic_moduli(C: Array) -> Dict:
  """ Extract commonly used isotropic constants.

  There are a number of important constants used to describe the linear
  elastic behavior of isotropic systems, including the bulk modulud, B,
  the shear modulus, G, the longitudinal modulus, M, the Young's modulus,
  E, and the Poisson's ratio, nu. This convenience function extracts them
  from an elastic modulus tensor C.

  Angle averaged quantities: While these quantities are defined for
  isotropic systems, one can still define an "angle-averaged shear
  modulus", for example, that averages over all possible shear
  deformations. This can be useful for systems that are statistically
  isotropic but where a particular realization is slightly anisotropic.
  The precise definitions are as follows:

  First, we define the "response", R, to a certain strain tensor e to be
    R = 2 * (U / V^0 - U^0/V^0 - s^0_{ij} e_{ji}) = C_{ijkl} e_{ij} e_{kl}

  Bulk modulus, B:
  This is the response to the rotationally invariant strain tensor:
      e = (1/2) * ( 1 0 )     or      e = (1/3) * ( 1 0 0 )
                  ( 0 1 )                         ( 0 1 0 )
                                                  ( 0 0 1 )

  Shear modulus, G:
  This is the response to the strain tensor:
      e = (1/2) * ( 0 1 )     or      e = (1/2) * ( 0 1 0 )
                  ( 1 0 )                         ( 1 0 0 )
                                                  ( 0 0 0 )
  averaged over all possible orientations. For perfectly isotropic
  systems, it should be equal to C[0,1,0,1].

  Longitudinal modulus, M:
  This is the response to the strain tensor:
      e = ( 1 0 )     or      e = ( 1 0 0 )
          ( 0 0 )                 ( 0 0 0 )
                                  ( 0 0 0 )
  averaged over all possible orientations. For perfectly isotropic
  systems, it should be equal to C[0,0,0,0].

  Young's modulus, E:
  This is a measure of tensile stiffness and is calculated a using well-
  known expression in terms of B and G.

  Poisson's ratio:
  This is a measure of deformations in directions perpendicular to an
  applied load and is calculated a using well-known expression in terms
  of B and G.

  Args:
    C: A previously calculated elastic modulus tensor represented as an
      array of shape (spatial_dimension,spatial_dimension,spatial_dimension,
      spatial_dimension), where spatial_dimension is either 2 or 3. C must
      satisfy both the major and minor symmetries, but this is not checked.

  Return: a dictionary containing the elastic constants.

  """
  if C.shape[0] == 2:
    cxxxx,cyyyy,cxyxy,cxxyy,cxxxy,cyyxy = _extract_elements(C,False)
    B = (cxxxx + cyyyy + 2. * cxxyy) / 4.
    G = (4. * cxyxy + cxxxx + cyyyy - 2. * cxxyy) / 8.
    M = B + G
    E = 4 * B * G / (B + G)
    nu = (B - G) / (B + G)

  elif C.shape[0] == 3:
    cxxxx,cyyyy,czzzz,cyzyz,cxzxz,cxyxy,cyyzz,cxxzz,cxxyy, \
    cxxyz,cxxxz,cxxxy,cyyyz,cyyxz,cyyxy,czzyz,czzxz,czzxy, \
    cyzxz,cyzxy,cxzxy = _extract_elements(C,False)
    B = (cxxxx + 2 * cxxyy + 2 * cxxzz + cyyyy + 2 * cyyzz + czzzz) / 9.
    G = (cxxxx - cxxyy - cxxzz + 3 * cxyxy + 3 * cxzxz + cyyyy - cyyzz
         + 3 * cyzyz + czzzz) / 15.
    M = B + 4 * G / 3
    E = 9 * B * G / (3 * B + G)
    nu = (3 * B - 2 * G) / (2 * (3 * B + G))
  else:
    raise AssertionError('C has incorrect shape')

  return {'B':B, 'G':G, 'M':M, 'E':E, 'nu':nu}







