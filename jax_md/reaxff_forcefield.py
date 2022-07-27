"""
Contains force field related code

Author: Mehmet Cagri Kaymak
"""


import numpy as onp
import jax.numpy as np
import jax
import pickle
from jax_md import dataclasses, util
from dataclasses import fields

Array = util.Array
# it fixes nan values issue, from: https://github.com/google/jax/issues/1052
def vectorized_cond(pred, true_fun, false_fun, operand):
  # true_fun and false_fun must act elementwise (i.e. be vectorized)
  #how to use: grad(lambda x: vectorized_cond(x > 0.5, lambda x: np.arctan2(x, x), lambda x: 0., x))(0.)
  true_op = np.where(pred, operand, 0)
  false_op = np.where(pred, 0, operand)
  return np.where(pred, true_fun(true_op), false_fun(false_op))


#https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
@jax.custom_jvp
def safe_sqrt(x):
  return np.sqrt(x)

@safe_sqrt.defjvp
def safe_sqrt_jvp(primals, tangents):
  x = primals[0]
  x_dot = tangents[0]
  #print(x[0])
  primal_out = safe_sqrt(x)
  tangent_out = 0.5 * x_dot / np.where(x > 0, primal_out, np.inf)
  return primal_out, tangent_out


#TODO: this part needs to be removed
CLOSE_NEIGH_CUTOFF=5.0

@dataclasses.dataclass
class ForceField(object):
    num_atom_types: int = dataclasses.static_field()
    name_to_index: dict = dataclasses.static_field()
    params_to_indices: dict = dataclasses.static_field()

    body3_indices_src: tuple = dataclasses.static_field()
    body3_indices_dst: tuple = dataclasses.static_field()
    body4_indices_src: tuple = dataclasses.static_field()
    body4_indices_dst: tuple = dataclasses.static_field()

    low_tap_rad: Array = dataclasses.static_field()
    up_tap_rad: Array = dataclasses.static_field()

    cutoff: Array = dataclasses.static_field()
    cutoff2: Array = dataclasses.static_field()

    body2_params_mask: Array = dataclasses.static_field()
    body3_params_mask: Array = dataclasses.static_field()
    body4_params_mask: Array = dataclasses.static_field()
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

    @classmethod
    def init_from_arg_dict(cls, kwargs):
        field_set = {f.name for f in fields(cls) if f.init}
        filtered_kwargs = {k : v for k, v in kwargs.items() if k in field_set}
        if len(filtered_kwargs) != len(field_set):
            print("Missing arguments")
        else:
            return cls(**filtered_kwargs)

        return cls(**filtered_kwargs)

def symm_force_field(FF_field_dict):
    # 2 body-params
    # for now global
    body_2_indices = np.tril_indices(FF_field_dict["num_atom_types"],k=-1)
    body_3_indices_src = FF_field_dict["body3_indices_src"]
    body_3_indices_dst = FF_field_dict["body3_indices_dst"]
    body_4_indices_src = FF_field_dict["body4_indices_src"]
    body_4_indices_dst = FF_field_dict["body4_indices_dst"]
    body_2_attr = ["p1co", "p2co", "p3co",
                   "p1co_off","p2co_off","p3co_off",
                   "rob1", "rob2", "rob3",
                   "rob1_off","rob2_off","rob3_off",
                   "ptp", "pdp", "popi",
                   "pdo", "bop1", "bop2",
                   "de1", "de2", "de3",
                   "psp", "psi", "vover"]
    for attr in body_2_attr:
        arr = FF_field_dict[attr]
        arr = jax.ops.index_update(arr,
                body_2_indices, arr.transpose()[body_2_indices])
        FF_field_dict[attr] = arr

    body_3_attr = ["vval2","vkac", "th0", "vka", "vkap", "vka3", "vka8"]
    for attr in body_3_attr:
        arr = FF_field_dict[attr]
        arr = jax.ops.index_update(arr,
                body_3_indices_dst, arr[body_3_indices_src])
        FF_field_dict[attr] = arr

    body_4_attr = ["v1","v2", "v3", "v4", "vconj"]
    for attr in body_4_attr:
        arr = FF_field_dict[attr]
        arr = jax.ops.index_update(arr,
                body_4_indices_dst, arr[body_4_indices_src])
        FF_field_dict[attr] = arr


def handle_offdiag(total_num_atom_types,
                   rat,rapt,vnq,
                   rob1_off,rob2_off,rob3_off,
                   rob1_off_mask,rob2_off_mask,rob3_off_mask,
                   rvdw,eps,alf,
                   p1co_off,p2co_off,p3co_off,
                   p1co_off_mask,p2co_off_mask,p3co_off_mask):
    '''
                          self.p1co_off_mask, #12
                              self.p2co_off_mask,
                              self.p3co_off_mask,

                              self.rob1_off_mask,#15
                              self.rob2_off_mask,
                              self.rob3_off_mask
    '''
    num_rows = total_num_atom_types

    mat1 = rat.reshape(1,-1)
    mat1 = np.tile(mat1,(num_rows,1))
    mat1_tr = mat1.transpose()
    rob1_temp = (mat1 + mat1_tr) * 0.5
    rob1_temp = np.where(mat1 > 0.0, rob1_temp, 0.0)
    rob1_temp = np.where(mat1_tr > 0.0, rob1_temp, 0.0)

    mat1 = rapt.reshape(1,-1)
    mat1 = np.tile(mat1,(num_rows,1))
    mat1_tr = mat1.transpose()
    rob2_temp = (mat1 + mat1_tr) * 0.5
    rob2_temp = np.where(mat1 > 0.0, rob2_temp, 0.0)
    rob2_temp = np.where(mat1_tr > 0.0, rob2_temp, 0.0)

    mat1 = vnq.reshape(1,-1)
    mat1 = np.tile(mat1,(num_rows,1))
    mat1_tr = mat1.transpose()
    rob3_temp = (mat1 + mat1_tr) * 0.5
    rob3_temp = np.where(mat1 > 0.0, rob3_temp, 0.0)
    rob3_temp = np.where(mat1_tr > 0.0, rob3_temp, 0.0)
    #TODO: gradient of sqrt. at 0 is nan, use safe sqrt
    p1co_temp = safe_sqrt(4.0 * rvdw.reshape(-1,1).dot(rvdw.reshape(1,-1)))
    p2co_temp = safe_sqrt(eps.reshape(-1,1).dot(eps.reshape(1,-1)))
    p3co_temp = safe_sqrt(alf.reshape(-1,1).dot(alf.reshape(1,-1)))


    rob1 = np.where(rob1_off_mask == 0, rob1_temp, rob1_off)
    rob2 = np.where(rob2_off_mask == 0, rob2_temp, rob2_off)
    rob3 = np.where(rob3_off_mask == 0, rob3_temp, rob3_off)

    p1co = np.where(p1co_off_mask == 0, p1co_temp, p1co_off * 2.0)
    p2co = np.where(p2co_off_mask == 0, p2co_temp, p2co_off)
    p3co = np.where(p3co_off_mask == 0, p3co_temp, p3co_off)

    return rob1, rob2, rob3, p1co, p2co, p3co

def preprocess_force_field(flattened_force_field, flattened_non_dif_params):
    return symm_force_field(handle_offdiag(flattened_force_field,flattened_non_dif_params),flattened_non_dif_params)

