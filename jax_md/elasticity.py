from functools import partial
from typing import Dict, Callable, List, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, lax, grad, vmap, jacfwd, jacrev, jvp

from jax_md.util import Array
from jax_md.util import f32
from jax_md.util import f64


def _get_strain_tensor_list(box: Array) -> Array:
  if box.shape == (2,2):
    strain_tensors = jnp.array([[[1., 0.],[0., 0.]],
                            [[0., 0.],[0., 1.]],
                            [[0., 1.],[1., 0.]],
                            [[1., 0.],[0., 1.]],
                            [[1., 1.],[1., 0.]],
                            [[0., 1.],[1., 1.]]], dtype=f64)
  elif box.shape == (3,3):
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
                                [[0, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=f64)
  else:
    raise AssertionError('not implemented in {} dimensions'.format(dimension))
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
    cyzyz = response_all[3]/4.;
    cxzxz = response_all[4]/4.;
    cxyxy = response_all[5]/4.;
    cyyzz = (response_all[6] - cyyyy - czzzz)/2.;
    cxxzz = (response_all[7] - cxxxx - czzzz)/2.;
    cxxyy = (response_all[8] - cxxxx - cyyyy)/2.;
    cxxyz = (response_all[9] - cxxxx - 4.*cyzyz)/4.;
    cxxxz = (response_all[10] - cxxxx - 4.*cxzxz)/4.;
    cxxxy = (response_all[11] - cxxxx - 4.*cxyxy)/4.;
    cyyyz = (response_all[12] - cyyyy - 4.*cyzyz)/4.;
    cyyxz = (response_all[13] - cyyyy - 4.*cxzxz)/4.;
    cyyxy = (response_all[14] - cyyyy - 4.*cxyxy)/4.;
    czzyz = (response_all[15] - czzzz - 4.*cyzyz)/4.;
    czzxz = (response_all[16] - czzzz - 4.*cxzxz)/4.;
    czzxy = (response_all[17] - czzzz - 4.*cxyxy)/4.;
    cyzxz = (response_all[18] - 4.*cyzyz - 4.*cxzxz)/8.;
    cyzxy = (response_all[19] - 4.*cyzyz - 4.*cxyxy)/8.;
    cxzxy = (response_all[20] - 4.*cxzxz - 4.*cxyxy)/8.;

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
    raise AssertionError('incorrect shape for array of responses')
  return C

def AthermalElasticModulusTensor(energy_fn: Callable[..., Array], 
                                 tether_strength: float=1e-10,
                                 gradient_check: Array=None, 
                                 cg_tol: float=1e-7
                                 ) -> Callable[..., Array]:
  """ Setup calculation of elastic modulus tensor for a 2d or 3d athermal system.

  The elastic modulus tensor describes a material's response to different 
  boundary deformations. Specifically, for small deformation given by a 
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

  Args:
    energy_fn: an energy function that is created using periodic_general so
      that it can be called with and differentiated with respect to a 'box' 
      argument. energy_fn(R, box=box) corresponds to the state around which we 
      are expanding
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
  
  Return: A function to calculate the elastic modulus tensor

    
  TODO:
    - generalize to work with force_fn functions? I'm not sure this is possible

  """

  def calculate_EMT(R: Array,
                    box: Array, 
                    **kwargs) -> Array:
    """Calculate the elastic modulus tensor

    energy_fn(R, box=box) corresponds to the state around which we are expanding
      
    Args:
      R: array of shape (N,dimension) of particle positions. This does not
        generalize to arbitrary dimensions and is only implemented for
          dimension == 2
          dimension == 3
      box: array of shape (dimension,dimension) representing the current box of 
        the system.
    
    Return: Elastic modulus tensor as an array of shape (dimension,dimension,
      dimension,dimension) that respects the major and minor symmetries
    """
    if len(box.shape) != 2:
      raise AssertionError('box must be a 2 dimensional array.')
    if box.shape[0] != box.shape[1]:
      raise AssertionError('box must be a square array.')
    if R.shape[-1] != box.shape[0]:
      raise AssertionError('inconsistent dimensions. R corresponds to a {}-dimensional \
      system but box corresponds to a {}-dimensional system.'.format(R.shape[-1], box.shape[0]))

    if not (R.shape[-1] == 2 or R.shape[-1] == 3):
      raise AssertionError('Only implemented for 2d and 3d systems.')

    def setup_energy_fn_general(strain_tensor):
      @jit
      def energy_fn_general(R, gamma):
        new_box = jnp.matmul(jnp.eye(strain_tensor.shape[0]) + gamma * strain_tensor, box)
        return energy_fn(R, box=new_box, **kwargs)
      return energy_fn_general
    
    def get_affine_response(strain_tensor):
      energy_fn_general = setup_energy_fn_general(strain_tensor)
      d2U_dRdgamma = jacfwd(jacrev(energy_fn_general,argnums=0),argnums=1)(R, 0.0).reshape(R.size)
      d2U_dgamma2  = jacfwd(jacrev(energy_fn_general,argnums=1),argnums=1)(R, 0.0)
      return d2U_dRdgamma, d2U_dgamma2

    strain_tensors = _get_strain_tensor_list(box)
    d2U_dRdgamma_all, d2U_dgamma2_all = vmap(get_affine_response)(strain_tensors)

    #solve the system of equations
    energy_fn_Ronly = partial(energy_fn, box=box, **kwargs)
    def hvp(f, primals, tangents):
      return jvp(grad(f), primals, tangents)[1]
    def hvp_specific_with_tether(v):
      return hvp(energy_fn_Ronly, (R,), (v.reshape(R.shape),)).reshape(v.shape) + tether_strength * v
    
    non_affine_response_all = jsp.sparse.linalg.cg(vmap(hvp_specific_with_tether),d2U_dRdgamma_all, tol=cg_tol)[0]
    #The above line should be functionally equivalent to:
    #H0=hessian(energy_fn)(R, box=box, **kwargs).reshape(R.size,R.size) + tether_strength * jnp.identity(R.size)
    #non_affine_response_all = jnp.transpose(jnp.linalg.solve(H0, jnp.transpose(d2U_dRdgamma_all)))

    def calc_response(d2U_dRdgamma,d2U_dgamma2,non_affine_response):
      return d2U_dgamma2 - jnp.dot(d2U_dRdgamma, non_affine_response)
    response_all = vmap(calc_response, in_axes=(0,0,0))(d2U_dRdgamma_all,d2U_dgamma2_all,non_affine_response_all)

    volume = box.diagonal().prod()
    response_all = response_all / volume
    C = _convert_responses_to_elastic_constants(response_all)
    
    #JAX does not allow proper runtime error handling in jitted function. 
    # Instead, if the user requests a gradient check and the check fails,
    # we convert C into jnp.nan's. While this doesn't raise an exception,
    # it at least is very "loud". 
    if gradient_check is not None:
      maxgrad = jnp.amax(jnp.abs(grad(energy_fn)(R, box=box, **kwargs)))
      C = lax.cond(maxgrad > gradient_check, lambda _: jnp.nan * C, lambda _: C, None)

    return C

  return calculate_EMT





def extract_elements(C: Array, as_dict: bool=True) -> Union[Dict,Array]:
  """ Convert an elastic modulus tensor into a list of 6 (21) unique elements
        in 2 (3) dimensions
      
      in 2d, these are:
      cxxxx,cyyyy,cxyxy,cxxyy,cxxxy,cyyxy

      in 3d, these are:
      cxxxx,cyyyy,czzzz,cyzyz,cxzxz,cxyxy,cyyzz,cxxzz,cxxyy,cxxyz,cxxxz,cxxxy,cyyyz,cyyxz,cyyxy,czzyz,czzxz,czzxy,cyzxz,cyzxy,cxzxy

  Args:
    C: A previously calculated elastic modulus tensor represented as an 
      array of shape (spatial_dimension,spatial_dimension,spatial_dimension,
      spatial_dimension), where spatial_dimension is either 2 or 3. C must 
      satisfy both the major and minor symmetries, but this is not checked.
  """
  if C.shape[0] == 2:
    indices = jnp.array([(0, 0, 0, 0),
                         (1, 1, 1, 1),
                         (0, 1, 0, 1),
                         (0, 0, 1, 1),
                         (0, 0, 0, 1),
                         (0, 1, 1, 1)]).transpose().tolist()
    clist = C[ tuple(indices) ] 
    if as_dict==True:
      names = ['cxxxx','cyyyy','cxyxy','cxxyy','cxxxy','cyyxy']
      return dict(zip(names, clist))
    else:
      return clist

  elif C.shape[0] == 3:
    indices = jnp.array([(0, 0, 0, 0),
                         (1, 1, 1, 1),
                         (2, 2, 2, 2),
                         (1, 2, 1, 2),
                         (0, 2, 0, 2),
                         (0, 1, 0, 1),
                         (1, 1, 2, 2),
                         (0, 0, 2, 2),
                         (0, 0, 1, 1),
                         (0, 0, 1, 2),
                         (0, 0, 0, 2),
                         (0, 0, 0, 1),
                         (1, 1, 1, 2),
                         (0, 2, 1, 1),
                         (0, 1, 1, 1),
                         (1, 2, 2, 2),
                         (0, 2, 2, 2),
                         (0, 1, 2, 2),
                         (0, 2, 1, 2),
                         (0, 1, 1, 2),
                         (0, 1, 0, 2)]).transpose().tolist()
    clist = C[ tuple(indices) ] 
    if as_dict:
      names = ['cxxxx','cyyyy','czzzz','cyzyz','cxzxz','cxyxy','cyyzz','cxxzz','cxxyy','cxxyz','cxxxz','cxxxy','cyyyz','cyyxz','cyyxy','czzyz','czzxz','czzxy','cyzxz','cyzxy','cxzxy']
      return dict(zip(names, clist))
    else:
      return clist
  else:
    raise AssertionError('C has wrong shape')

def _get_mandel_mapping_weight(DIM):
  if DIM == 2:
    m_map  = jnp.array([[0,0],[1,1],[0,1]], dtype=jnp.int8)
    weight = jnp.array([1,1,jnp.sqrt(2)], dtype=f64)
    return m_map, weight
  elif DIM == 3:
    m_map  = jnp.array([[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]], dtype=jnp.int8)
    weight = jnp.array([1,1,1,jnp.sqrt(2),jnp.sqrt(2),jnp.sqrt(2)], dtype=f64)
    return m_map, weight
  else:
    raise AssertionError('DIM must be 2 or 3')

def tensor_to_mandel(T: Array) -> Array:
  """ Convert a tensor to Mandel notation

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
  DIM = T.shape[0]
  if not (DIM==2 or DIM==3):
    raise AssertionError('DIM must be 2 or 3')
  
  rank = len(T.shape)
  if not (rank==2 or rank==4):
    raise AssertionError('T must have rank 2 or 4')
  
  m_map, weight = _get_mandel_mapping_weight(DIM)
  
  if rank == 2:
    extract = lambda idx, w: T[idx[0], idx[1]] * w
    M = vmap(extract, in_axes=(0,0))(m_map, weight)
  else:
    extract = lambda idx0, idx1, w0, w1: T[idx0[0], idx0[1], idx1[0], idx1[1]] * w0 * w1
    M = vmap(vmap(extract, in_axes=(0,None,0,None)), in_axes=(None,0,None,0))(m_map, m_map, weight, weight)
  return M

def mandel_to_tensor(M: Array) -> Array:
  """ Perform the inverse of M = tensor_to_mandel(T).

  Args:
    M: Array of shape (3,), (6,), (3,3), or (6,6)
  
  Output: Array of shape (2,2), (3,3), (2,2,2,2), or (3,3,3,3)
  """
  DIM = M.shape[0]
  if not (DIM==3 or DIM==6):
    raise AssertionError('DIM must be 3 or 6')
  
  rank = len(M.shape)
  if not (rank==1 or rank==2):
    raise AssertionError('T must have rank 1 or 2')

  def mandel_index(i,j):
    return lax.cond(i==j, lambda ij: ij[0], lambda ij: DIM-ij[0]-ij[1], (i,j))

  if DIM == 3:
    dimension = 2
  else:
    dimension = 3

  tensor_range = jnp.arange(dimension)
  _, weight = _get_mandel_mapping_weight(dimension)

  if rank == 1:
    def extract(i,j):
      idx = mandel_index(i,j)
      return M[idx] / weight[idx]
    T = vmap(vmap(extract, in_axes=(0,None)), in_axes=(None,0))(tensor_range, tensor_range)
  else:
    def extract(i,j,k,l):
      idx0 = mandel_index(i,j)
      idx1 = mandel_index(k,l)
      return M[idx0,idx1] / (weight[idx0] * weight[idx1])
    T = vmap(vmap(vmap(vmap(extract, in_axes=(0,None,None,None)), in_axes=(None,0,None,None)), in_axes=(None,None,0,None)), in_axes=(None,None,None,0))(tensor_range, tensor_range, tensor_range, tensor_range)
  return T



@partial(jit,static_argnums=(1,))
def _extract_elements(C, as_dict):
  """ Convert an elastic modulus tensor into a list of 6 (21) unique elements
        in 2 (3) dimensions
      
      in 2d, these are:
      cxxxx,cyyyy,cxyxy,cxxyy,cxxxy,cyyxy

      in 3d, these are:
      cxxxx,cyyyy,czzzz,cyzyz,cxzxz,cxyxy,cyyzz,cxxzz,cxxyy,cxxyz,cxxxz,cxxxy,cyyyz,cyyxz,cyyxy,czzyz,czzxz,czzxy,cyzxz,cyzxy,cxzxy

  Args:
    C: A previously calculated elastic modulus tensor represented as an 
      array of shape (spatial_dimension,spatial_dimension,spatial_dimension,
      spatial_dimension), where spatial_dimension is either 2 or 3. C must 
      satisfy both the major and minor symmetries, but this is not checked.
    as_dict: boolean. If true, return a dictionary with interpretable keys.
      If false, return an array that follows an internal convention.
  """
  if C.shape[0] == 2:
    indices = ( [0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1],
                [0, 1, 0, 1, 0, 1],
                [0, 1, 1, 1, 1, 1])
    clist = C[ indices ] 
    if as_dict==True:
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
      names = ['cxxxx','cyyyy','czzzz','cyzyz','cxzxz','cxyxy','cyyzz','cxxzz','cxxyy','cxxyz','cxxxz','cxxxy','cyyyz','cyyxz','cyyxy','czzyz','czzxz','czzxy','cyzxz','cyzxy','cxzxy']
      return dict(zip(names, clist))
    else:
      return clist
  else:
    raise AssertionError('C has wrong shape')

def extract_elements(C: Array) -> Dict:
  return _extract_elements(C,True)

def extract_isotropic_moduli(C: Array) -> Dict:
  """ There are a number of important constants used to describe the linear 
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

        First, we definte the "response", R, to a certain strain tensor e to be
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
    B = (cxxxx+cyyyy+2.*cxxyy) / 4.
    G = (4.*cxyxy + cxxxx+cyyyy-2.*cxxyy) / 8.
    M = B + G
    E = 4 * B * G / (B + G)
    nu = (B - G) / (B + G)
    
  elif C.shape[0] == 3:
    cxxxx,cyyyy,czzzz,cyzyz,cxzxz,cxyxy,cyyzz,cxxzz,cxxyy,cxxyz,cxxxz,cxxxy,cyyyz,cyyxz,cyyxy,czzyz,czzxz,czzxy,cyzxz,cyzxy,cxzxy = _extract_elements(C,False)
    B = (cxxxx + 2*cxxyy + 2*cxxzz + cyyyy + 2*cyyzz + czzzz) / 9.
    G = (cxxxx - cxxyy - cxxzz + 3*cxyxy + 3*cxzxz + cyyyy - cyyzz + 3*cyzyz + czzzz) / 15.
    M = B + 4 * G / 3
    E = 9 * B * G / (3 * B + G)
    nu = (3 * B - 2 * G) / (2 * (3 * B + G))
  else:
    raise AssertionError('C has incorrect shape')

  return {'B':B, 'G':G, 'M':M, 'E':E, 'nu':nu}







