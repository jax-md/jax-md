"""
Contains force field related code

Author: Mehmet Cagri Kaymak
"""
from jax_md import dataclasses, util
from dataclasses import fields
import jax
import jax.numpy as jnp

Array = util.Array

@dataclasses.dataclass
class ForceField(object):
  '''
  Container for ReaxFF parameters
  '''
  num_atom_types: int = dataclasses.static_field()
  name_to_index: dict = dataclasses.static_field()
  params_to_indices: dict = dataclasses.static_field()
  # these tuples are used to handle symmetric parameters in 3 and 4 body param
  # lists
  body3_indices_src: tuple = dataclasses.static_field()
  body3_indices_dst: tuple = dataclasses.static_field()
  body4_indices_src: tuple = dataclasses.static_field()
  body4_indices_dst: tuple = dataclasses.static_field()
  # self energies for each atom type
  self_energies: Array
  # overall energy shift
  shift: Array

  low_tap_rad: Array = dataclasses.static_field()
  up_tap_rad: Array = dataclasses.static_field()

  cutoff: Array = dataclasses.static_field()
  cutoff2: Array = dataclasses.static_field()
  hb_close_cutoff: Array = dataclasses.static_field()
  hb_far_cutoff: Array = dataclasses.static_field()

  body2_params_mask: Array = dataclasses.static_field()
  body3_params_mask: Array = dataclasses.static_field()
  body4_params_mask: Array = dataclasses.static_field()
  # since 4 body interactions are created from 3-body, we need to extend
  # 3 body mask based on the 4-body interactions to not miss any 4 body inter.
  body34_params_mask: Array = dataclasses.static_field()
  hb_params_mask: Array = dataclasses.static_field()

  global_params: Array

  electronegativity: Array
  idempotential: Array
  gamma: Array

  rvdw: Array
  p1co: Array
  p1co_off: Array
  p1co_off_mask: Array = dataclasses.static_field()

  eps: Array
  p2co: Array
  p2co_off: Array
  p2co_off_mask: Array = dataclasses.static_field()

  alf: Array
  p3co: Array
  p3co_off: Array
  p3co_off_mask: Array = dataclasses.static_field()

  vop: Array
  amas: Array

  rat: Array
  rob1: Array
  rob1_off: Array
  rob1_off_mask: Array = dataclasses.static_field()

  rapt: Array
  rob2: Array
  rob2_off: Array
  rob2_off_mask: Array = dataclasses.static_field()

  vnq: Array
  rob3: Array
  rob3_off: Array
  rob3_off_mask: Array = dataclasses.static_field()

  ptp: Array
  pdp: Array
  popi: Array
  pdo: Array
  bop1: Array
  bop2: Array

  de1: Array
  de2: Array
  de3: Array
  psp: Array
  psi: Array

  aval: Array
  vval3: Array
  bo131: Array
  bo132: Array
  bo133: Array
  ovc: Array = dataclasses.static_field()
  v13cor: Array = dataclasses.static_field()

  softcut: Array # acks2 parameter
  softcut_2d: Array # softcut_2d[i,j] = 0.5 * (softcut[i] + softcut[j])

  stlp: Array
  valf: Array
  vval1: Array
  vval2: Array
  vval3: Array
  vval4: Array

  vkac: Array
  th0: Array
  vka: Array
  vkap: Array
  vka3: Array
  vka8: Array
  vval2: Array

  vlp1: Array
  valp1: Array
  vovun: Array
  vover: Array

  v1: Array
  v2: Array
  v3: Array
  v4: Array
  vconj: Array

  nphb: Array = dataclasses.static_field()
  rhb: Array
  dehb: Array
  vhb1: Array
  vhb2: Array

  # global parameters
  vdw_shiedling: Array
  trip_stab4: Array
  trip_stab5: Array
  trip_stab8: Array
  trip_stab11: Array
  over_coord1: Array
  over_coord2: Array
  val_par3: Array
  val_par15: Array
  val_par17: Array
  val_par18: Array
  val_par20: Array
  val_par21: Array
  val_par22: Array
  val_par31: Array
  val_par34: Array
  val_par39: Array
  par_16: Array
  par_6: Array
  par_7: Array
  par_9: Array
  par_10: Array
  par_32: Array
  par_33: Array
  par_24: Array
  par_25: Array
  par_26: Array
  par_28: Array

  par_35: Array #ACKS2

  @classmethod
  def init_from_arg_dict(cls, kwargs):
    field_set = {f.name for f in fields(cls) if f.init}
    filtered_kwargs = {k : v for k, v in kwargs.items() if k in field_set}
    if len(filtered_kwargs) != len(field_set):
      print("Missing arguments")
    else:
      return cls(**filtered_kwargs)

    return cls(**filtered_kwargs)

  def fill_symm(force_field):
    '''
    Fills the parameter arrays based on the symmetries
    '''
    # 2 body-params
    # for now global
    num_atoms = force_field.num_atom_types
    body_2_indices = jnp.tril_indices(num_atoms,k=-1)
    body_3_indices_src = force_field.body3_indices_src
    body_3_indices_dst = force_field.body3_indices_dst
    body_4_indices_src = force_field.body4_indices_src
    body_4_indices_dst = force_field.body4_indices_dst

    replace_dict = {}

    body_2_attr = ["p1co", "p2co", "p3co",
                   "p1co_off","p2co_off","p3co_off",
                   "rob1", "rob2", "rob3",
                   "rob1_off","rob2_off","rob3_off",
                   "ptp", "pdp", "popi",
                   "pdo", "bop1", "bop2",
                   "de1", "de2", "de3",
                   "psp", "psi", "vover"]
    for attr in body_2_attr:
      arr = getattr(force_field, attr)
      arr = arr.at[body_2_indices].set(arr.transpose()[body_2_indices])
      replace_dict[attr] = arr

    body_3_attr = ["vval2","vkac", "th0", "vka", "vkap", "vka3", "vka8"]
    for attr in body_3_attr:
      arr = getattr(force_field, attr)
      arr = arr.at[body_3_indices_dst].set(arr[body_3_indices_src])
      replace_dict[attr] = arr

    body_4_attr = ["v1","v2", "v3", "v4", "vconj"]
    for attr in body_4_attr:
      arr = getattr(force_field, attr)
      arr = arr.at[body_4_indices_dst].set(arr[body_4_indices_src])
      replace_dict[attr] = arr

    force_field = dataclasses.replace(force_field, **replace_dict)

    return force_field

  def fill_off_diag(force_field):
    '''
    Fills the off-diagonal entries in the parameter arrays
    '''
    num_rows = force_field.num_atom_types
    rat = force_field.rat
    rapt = force_field.rapt
    vnq = force_field.vnq
    rvdw = force_field.rvdw
    eps = force_field.eps
    alf = force_field.alf
    rob1_off = force_field.rob1_off
    rob2_off = force_field.rob2_off
    rob3_off = force_field.rob3_off
    rob1_off_mask = force_field.rob1_off_mask
    rob2_off_mask = force_field.rob2_off_mask
    rob3_off_mask = force_field.rob3_off_mask
    p1co_off = force_field.p1co_off
    p2co_off = force_field.p2co_off
    p3co_off = force_field.p3co_off
    p1co_off_mask = force_field.p1co_off_mask
    p2co_off_mask = force_field.p2co_off_mask
    p3co_off_mask = force_field.p3co_off_mask

    softcut = force_field.softcut

    mat1 = rat.reshape(1,-1)
    mat1 = jnp.tile(mat1,(num_rows,1))
    mat1_tr = mat1.transpose()
    rob1_temp = (mat1 + mat1_tr) * 0.5
    rob1_temp = jnp.where(mat1 > 0.0, rob1_temp, 0.0)
    rob1_temp = jnp.where(mat1_tr > 0.0, rob1_temp, 0.0)

    mat1 = rapt.reshape(1,-1)
    mat1 = jnp.tile(mat1,(num_rows,1))
    mat1_tr = mat1.transpose()
    rob2_temp = (mat1 + mat1_tr) * 0.5
    rob2_temp = jnp.where(mat1 > 0.0, rob2_temp, 0.0)
    rob2_temp = jnp.where(mat1_tr > 0.0, rob2_temp, 0.0)

    mat1 = vnq.reshape(1,-1)
    mat1 = jnp.tile(mat1,(num_rows,1))
    mat1_tr = mat1.transpose()
    rob3_temp = (mat1 + mat1_tr) * 0.5
    rob3_temp = jnp.where(mat1 > 0.0, rob3_temp, 0.0)
    rob3_temp = jnp.where(mat1_tr > 0.0, rob3_temp, 0.0)

    p1co_temp = 4.0 * rvdw.reshape(-1,1).dot(rvdw.reshape(1,-1))
    p1co_temp = util.safe_mask(p1co_temp > 0, jnp.sqrt, p1co_temp)
    p2co_temp = eps.reshape(-1,1).dot(eps.reshape(1,-1))
    p2co_temp = util.safe_mask(p2co_temp > 0, jnp.sqrt, p2co_temp)
    p3co_temp = alf.reshape(-1,1).dot(alf.reshape(1,-1))
    p3co_temp = util.safe_mask(p3co_temp > 0, jnp.sqrt, p3co_temp)

    rob1 = jnp.where(rob1_off_mask == 0, rob1_temp, rob1_off)
    rob2 = jnp.where(rob2_off_mask == 0, rob2_temp, rob2_off)
    rob3 = jnp.where(rob3_off_mask == 0, rob3_temp, rob3_off)

    p1co = jnp.where(p1co_off_mask == 0, p1co_temp, p1co_off * 2.0)
    p2co = jnp.where(p2co_off_mask == 0, p2co_temp, p2co_off)
    p3co = jnp.where(p3co_off_mask == 0, p3co_temp, p3co_off)

    softcut_2d = 0.5 * (softcut.reshape(-1,1) + softcut.reshape(1,-1))

    force_field = dataclasses.replace(force_field,
                                      rob1=rob1,
                                      rob2=rob2,
                                      rob3=rob3,
                                      p1co=p1co,
                                      p2co=p2co,
                                      p3co=p3co,
                                      softcut_2d=softcut_2d)
    return force_field







