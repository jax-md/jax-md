"""
Contains force field related code

Author: Mehmet Cagri Kaymak
"""
from jax_md import dataclasses, util
from dataclasses import fields

Array = util.Array

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


