"""
Contains helper functions ReaxFF

Author: Mehmet Cagri Kaymak
"""

import jax
import jax.numpy as jnp
import numpy as onp
from jax_md.reaxff.reaxff_forcefield import ForceField
from dataclasses import fields
from jax import custom_jvp
from frozendict import frozendict

@custom_jvp
def safe_sqrt(x):
  """Safe sqrt function (no nan gradients)."""
  return jnp.sqrt(x)

@safe_sqrt.defjvp
def safe_sqrt_jvp(primals, tangents):
  x = primals[0]
  x_dot = tangents[0]
  #print(x[0])
  primal_out = safe_sqrt(x)
  tangent_out = 0.5 * x_dot / jnp.where(x > 0, primal_out, jnp.inf)
  return primal_out, tangent_out

# it fixes nan values issue, from: https://github.com/google/jax/issues/1052
def vectorized_cond(pred, true_fun, false_fun, operand):
  # true_fun and false_fun must act elementwise (i.e. be vectorized)
  true_op = jnp.where(pred, operand, 0)
  false_op = jnp.where(pred, 0, operand)
  return jnp.where(pred, true_fun(true_op), false_fun(false_op))

def init_params_for_filler_atom_type(FF_field_dict):
  #TODO: make sure that index -1 doesnt belong to a real atom!!!
  
  FF_field_dict['rvdw'][-1] = 1
  FF_field_dict['eps'][-1] = 1
  FF_field_dict['alf'][-1] = 1

  FF_field_dict['vop'][-1] = 1

  FF_field_dict['gamma'][-1] = 1
  FF_field_dict['electronegativity'][-1] = 1
  FF_field_dict['idempotential'][-1] = 1

  FF_field_dict['bo131'][-1] = 1
  FF_field_dict['bo132'][-1] = 1
  FF_field_dict['bo133'][-1] = 1

def read_force_field(force_field_file,
                     cutoff2 = 1e-3,
                     hbond_close_cutoff = 0.01,
                     hbond_far_cutoff = 7.5,
                     dtype = jnp.float32):
  # to store all arguments together before creating the class
  FF_field_dict = {f.name:None for f in fields(ForceField) if f.init}
  FF_param_to_index = {}
  f = open(force_field_file, 'r')
  header = f.readline().strip()

  num_params = int(f.readline().strip().split()[0])
  global_params = onp.zeros(shape=(num_params,1), dtype=dtype)
  name_to_index = dict()
  body_3_indices_src = [[],[],[]]
  body_3_indices_dst = [[],[],[]]
  body_4_indices_src = [[],[],[],[]]
  body_4_indices_dst = [[],[],[],[]]

  for i in range(num_params):
    line = f.readline().strip()
    #to seperate the comment
    line = line.replace('!', ' ! ')
    global_params[i] = float(line.split()[0])
    
  FF_field_dict['low_tap_rad'] = global_params[11]
  FF_field_dict['up_tap_rad'] = global_params[12]
  FF_field_dict['vdw_shiedling'] = global_params[28]
  FF_field_dict['cutoff'] = global_params[29] * 0.01
  FF_field_dict['cutoff2'] = cutoff2
  FF_field_dict['hbond_close_cutff'] = hbond_close_cutoff
  FF_field_dict['hbond_far_cutoff'] = hbond_far_cutoff
  FF_field_dict['over_coord1'] = global_params[0]
  FF_field_dict['over_coord2'] = global_params[1]
  FF_field_dict['trip_stab4'] = global_params[3]
  FF_field_dict['trip_stab5'] = global_params[4]
  FF_field_dict['trip_stab8'] = global_params[7]
  FF_field_dict['trip_stab11'] = global_params[10]


  #FF_param_to_index[(1,12,1)] = ("low_tap_rad", (0,))
  #FF_param_to_index[(1,13,1)] = ("up_tap_rad", (0,))
  FF_param_to_index[(1,29,1)] = ("vdw_shiedling", (0,))
  #FF_param_to_index[(1,30,1)] = ("cutoff", (0,))
  FF_param_to_index[(1,1,1)] = ("over_coord1", (0,))
  FF_param_to_index[(1,2,1)] = ("over_coord2", (0,))
  FF_param_to_index[(1,4,1)] = ("trip_stab4", (0,))
  FF_param_to_index[(1,5,1)] = ("trip_stab5", (0,))
  FF_param_to_index[(1,8,1)] = ("trip_stab8", (0,))
  FF_param_to_index[(1,11,1)] = ("trip_stab11", (0,))

  FF_field_dict['val_par3'] = global_params[2]
  FF_field_dict['val_par15'] = global_params[14]
  FF_field_dict['par_16'] = global_params[15]
  FF_field_dict['val_par17'] = global_params[16]
  FF_field_dict['val_par18'] = global_params[17]
  FF_field_dict['val_par20'] = global_params[19]
  FF_field_dict['val_par21'] = global_params[20]
  FF_field_dict['val_par22'] = global_params[21]
  FF_field_dict['val_par31'] = global_params[30]
  FF_field_dict['val_par34'] = global_params[33]
  FF_field_dict['val_par39'] = global_params[38]

  FF_param_to_index[(1,3,1)] = ("val_par3", (0,))
  FF_param_to_index[(1,15,1)] = ("val_par15", (0,))
  FF_param_to_index[(1,16,1)] = ("par_16", (0,))
  FF_param_to_index[(1,17,1)] = ("val_par17", (0,))
  FF_param_to_index[(1,18,1)] = ("val_par18", (0,))
  FF_param_to_index[(1,20,1)] = ("val_par20", (0,))
  FF_param_to_index[(1,21,1)] = ("val_par21", (0,))
  FF_param_to_index[(1,22,1)] = ("val_par22", (0,))
  FF_param_to_index[(1,31,1)] = ("val_par31", (0,))
  FF_param_to_index[(1,34,1)] = ("val_par34", (0,))
  FF_param_to_index[(1,39,1)] = ("val_par39", (0,))

  # over under
  FF_field_dict['par_6'] = global_params[5]
  FF_field_dict['par_7'] = global_params[6]
  FF_field_dict['par_9'] = global_params[8]
  FF_field_dict['par_10'] = global_params[9]
  FF_field_dict['par_32'] = global_params[31]
  FF_field_dict['par_33'] = global_params[32]

  FF_param_to_index[(1,6,1)] = ("par_6", (0,))
  FF_param_to_index[(1,7,1)] = ("par_7", (0,))
  FF_param_to_index[(1,9,1)] = ("par_9", (0,))
  FF_param_to_index[(1,10,1)] = ("par_10", (0,))
  FF_param_to_index[(1,32,1)] = ("par_32", (0,))
  FF_param_to_index[(1,33,1)] = ("par_33", (0,))

  # torsion par_24,par_25, par_26,par_28
  FF_field_dict['par_24'] = global_params[23]
  FF_field_dict['par_25'] = global_params[24]
  FF_field_dict['par_26'] = global_params[25]
  FF_field_dict['par_28'] = global_params[27]

  FF_param_to_index[(1,24,1)] = ("par_24", (0,))
  FF_param_to_index[(1,25,1)] = ("par_25", (0,))
  FF_param_to_index[(1,26,1)] = ("par_26", (0,))
  FF_param_to_index[(1,28,1)] = ("par_28", (0,))

  #ACKS2
  FF_field_dict['par_35'] = global_params[34]
  FF_param_to_index[(1,35,1)] = ("par_35", (0,))


  real_num_atom_types = int(f.readline().strip().split()[0])
  num_atom_types = real_num_atom_types + 1 # 1 extra to store dummy atoms
  # self energies of the atoms
  FF_field_dict['self_energies'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['shift'] = onp.zeros(1,dtype=dtype)
  FF_field_dict['num_atom_types'] = num_atom_types
  
  # skip 3 lines of comment
  f.readline()
  f.readline()
  f.readline()

  atom_names = []
  line_ctr = 0

  # line 1
  FF_field_dict['rat'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['aval'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['amas'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['rvdw'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['eps'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['gamma'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['rapt'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['stlp'] = onp.zeros(num_atom_types,dtype=dtype)
  # line 2
  FF_field_dict['alf'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['vop'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['valf'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['valp1'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['electronegativity'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['idempotential'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['nphb'] = onp.zeros(num_atom_types,dtype=jnp.int32)
  # line 3
  FF_field_dict['vnq'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['vlp1'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['bo131'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['bo132'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['bo133'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['softcut'] = onp.zeros(num_atom_types,dtype=dtype)
  # line 4
  FF_field_dict['vovun'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['vval1'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['vval3'] = onp.zeros(num_atom_types,dtype=dtype)
  FF_field_dict['vval4'] = onp.zeros(num_atom_types,dtype=dtype)

  for i in range(real_num_atom_types):
    # first line
    line = f.readline().strip()
    split_line = line.split()
    atom_names.append(str(split_line[0]))
    name_to_index[atom_names[i]] = i

    FF_field_dict['rat'][i] = float(split_line[1])
    FF_field_dict['aval'][i] = float(split_line[2])
    FF_field_dict['amas'][i] = float(split_line[3])
    FF_field_dict['rvdw'][i] = float(split_line[4]) #vdw
    FF_field_dict['eps'][i] = float(split_line[5]) #vdw
    FF_field_dict['gamma'][i] = float(split_line[6]) #coulomb
    FF_field_dict['rapt'][i] = float(split_line[7])
    FF_field_dict['stlp'][i] = float(split_line[8]) #valency

    FF_param_to_index[(2,i+1,1)] = ("rat", (i,))
    FF_param_to_index[(2,i+1,2)] = ("aval", (i,))
    #FF_param_to_index[(2,i+1,3)] = ("amas", (i,))
    FF_param_to_index[(2,i+1,4)] = ("rvdw", (i,))
    FF_param_to_index[(2,i+1,5)] = ("eps", (i,))
    FF_param_to_index[(2,i+1,6)] = ("gamma", (i,))
    FF_param_to_index[(2,i+1,7)] = ("rapt", (i,))
    FF_param_to_index[(2,i+1,8)] = ("stlp", (i,))

    # second line
    line = f.readline().strip()
    split_line = line.split()
    FF_field_dict['alf'][i] = float(split_line[0]) #vdw
    FF_field_dict['vop'][i] = float(split_line[1]) #vdw
    FF_field_dict['valf'][i] = float(split_line[2]) # valency
    FF_field_dict['valp1'][i] = float(split_line[3]) #over-under coord
    FF_field_dict['electronegativity'][i] = float(split_line[5]) #coulomb
    # eta will be mult. by 2
    FF_field_dict['idempotential'][i] = float(split_line[6])
    # needed for hbond #needed to find acceptor-donor
    FF_field_dict['nphb'][i] = int(float(split_line[7]))

    FF_param_to_index[(2,i+1,9)] = ("alf", (i,))
    FF_param_to_index[(2,i+1,10)] = ("vop", (i,))
    FF_param_to_index[(2,i+1,11)] = ("valf", (i,))
    FF_param_to_index[(2,i+1,12)] = ("valp1", (i,))
    FF_param_to_index[(2,i+1,14)] = ("electronegativity", (i,))
    FF_param_to_index[(2,i+1,15)] = ("idempotential", (i,))
    # third line
    line = f.readline().strip()
    split_line = line.split()
    FF_field_dict['vnq'][i] = float(split_line[0])
    FF_field_dict['vlp1'][i] = float(split_line[1])
    FF_field_dict['bo131'][i] = float(split_line[3])
    FF_field_dict['bo132'][i] = float(split_line[4])
    FF_field_dict['bo133'][i] = float(split_line[5])
    FF_field_dict['softcut'][i] = float(split_line[6]) #ACKS2

    FF_param_to_index[(2,i+1,17)] = ("vnq", (i,))
    FF_param_to_index[(2,i+1,18)] = ("vlp1", (i,))
    FF_param_to_index[(2,i+1,20)] = ("bo131", (i,))
    FF_param_to_index[(2,i+1,21)] = ("bo132", (i,))
    FF_param_to_index[(2,i+1,22)] = ("bo133", (i,))
    FF_param_to_index[(2,i+1,23)] = ("softcut", (i,))

    # fourth line
    line = f.readline().strip()
    split_line = line.split()
    FF_field_dict['vovun'][i] = float(split_line[0]) #over-under coord
    FF_field_dict['vval1'][i] = float(split_line[1])
    FF_field_dict['vval3'][i] = float(split_line[3])
    FF_field_dict['vval4'][i] = float(split_line[4])

    FF_param_to_index[(2,i+1,25)] = ("vovun", (i,))
    FF_param_to_index[(2,i+1,26)] = ("vval1", (i,))
    FF_param_to_index[(2,i+1,28)] = ("vval3", (i,))
    FF_param_to_index[(2,i+1,29)] = ("vval4", (i,))

    # This part is moved to the related part in energy calculation
    #if FF_field_dict['amas'][i] < 21.0:
    #    FF_field_dict['vval3'][i] = FF_field_dict['valf'][i]
  FF_field_dict['name_to_index'] = name_to_index

  FF_field_dict['body2_params_mask'] =  onp.zeros((num_atom_types,
                                                   num_atom_types),
                                                  dtype=jnp.bool_)
  # line 1
  FF_field_dict['de1'] = onp.zeros((num_atom_types,
                                    num_atom_types),dtype=dtype)
  FF_field_dict['de2'] =  onp.zeros((num_atom_types,
                                     num_atom_types),dtype=dtype)
  FF_field_dict['de3'] =  onp.zeros((num_atom_types,
                                     num_atom_types),dtype=dtype)
  FF_field_dict['psi'] =  onp.zeros((num_atom_types,
                                     num_atom_types),dtype=dtype)
  FF_field_dict['pdo'] =  onp.zeros((num_atom_types,
                                     num_atom_types),dtype=dtype)
  FF_field_dict['v13cor'] =  onp.zeros((num_atom_types,
                                        num_atom_types),dtype=dtype)
  FF_field_dict['popi'] =  onp.zeros((num_atom_types,
                                      num_atom_types),dtype=dtype)
  FF_field_dict['vover'] =  onp.zeros((num_atom_types,
                                       num_atom_types),dtype=dtype)
  # line 2
  FF_field_dict['psp'] =  onp.zeros((num_atom_types,
                                     num_atom_types),dtype=dtype)
  FF_field_dict['pdp'] =  onp.zeros((num_atom_types,
                                     num_atom_types),dtype=dtype)
  FF_field_dict['ptp'] =  onp.zeros((num_atom_types,
                                     num_atom_types),dtype=dtype)
  FF_field_dict['bop1'] =  onp.zeros((num_atom_types,
                                      num_atom_types),dtype=dtype)
  FF_field_dict['bop2'] =  onp.zeros((num_atom_types,
                                      num_atom_types),dtype=dtype)
  FF_field_dict['ovc'] =  onp.zeros((num_atom_types,
                                     num_atom_types),dtype=dtype)
  line = f.readline().strip()
  num_bonds = int(line.split()[0])
  f.readline() # skip next line (comment)
  for b in range(num_bonds):
    # first line
    line = f.readline().strip()
    split_line = line.split()
    i = int(split_line[0]) - 1 # index starts at 0
    j = int(split_line[1]) - 1

    FF_field_dict['body2_params_mask'][i,j] = 1
    FF_field_dict['body2_params_mask'][j,i] = 1

    FF_field_dict['de1'][i,j] = float(split_line[2])
    FF_field_dict['de2'][i,j] = float(split_line[3])
    FF_field_dict['de3'][i,j] = float(split_line[4])
    FF_field_dict['psi'][i,j] = float(split_line[5])
    FF_field_dict['pdo'][i,j] = float(split_line[6])
    FF_field_dict['v13cor'][i,j] = float(split_line[7])
    FF_field_dict['popi'][i,j] = float(split_line[8])
    FF_field_dict['vover'][i,j] = float(split_line[9])

    FF_param_to_index[(3,b+1,1)] = ("de1", (i,j))
    FF_param_to_index[(3,b+1,2)] = ("de2", (i,j))
    FF_param_to_index[(3,b+1,3)] = ("de3", (i,j))
    FF_param_to_index[(3,b+1,4)] = ("psi", (i,j))
    FF_param_to_index[(3,b+1,5)] = ("pdo", (i,j))
    #FF_param_to_index[(3,b+1,6)] = ("v13cor", (i,j))
    FF_param_to_index[(3,b+1,7)] = ("popi", (i,j))
    FF_param_to_index[(3,b+1,8)] = ("vover", (i,j))
    # v13cor is static, so content needs to be finaized here
    # hence symm. part
    FF_field_dict['v13cor'][j,i] = FF_field_dict['v13cor'][i,j]
    # second line
    line = f.readline().strip()
    split_line = line.split()
    FF_field_dict['psp'][i,j] = float(split_line[0])
    FF_field_dict['pdp'][i,j] = float(split_line[1])
    FF_field_dict['ptp'][i,j] = float(split_line[2])
    FF_field_dict['bop1'][i,j] = float(split_line[4])
    FF_field_dict['bop2'][i,j] = float(split_line[5])
    FF_field_dict['ovc'][i,j] = float(split_line[6])
    # v13cor is static, so content needs to be finaized here
    FF_field_dict['ovc'][j,i] = FF_field_dict['ovc'][i,j]

    FF_param_to_index[(3,b+1,9)] = ("psp", (i,j))
    FF_param_to_index[(3,b+1,10)] = ("pdp", (i,j))
    FF_param_to_index[(3,b+1,11)] = ("ptp", (i,j))
    FF_param_to_index[(3,b+1,13)] = ("bop1", (i,j))
    FF_param_to_index[(3,b+1,14)] = ("bop2", (i,j))
    #FF_param_to_index[(3,b+1,8)] = ("ovc", (i,j))

  line = f.readline().strip()
  num_off_diag = int(line.split()[0])

  FF_field_dict['rob1_off'] = onp.zeros((num_atom_types,
                                         num_atom_types),dtype=dtype)
  FF_field_dict['rob1_off_mask'] =  onp.zeros((num_atom_types,
                                               num_atom_types),dtype=jnp.bool_)
  FF_field_dict['rob2_off'] =  onp.zeros((num_atom_types,
                                          num_atom_types),dtype=dtype)
  FF_field_dict['rob2_off_mask'] =  onp.zeros((num_atom_types,
                                               num_atom_types),dtype=jnp.bool_)
  FF_field_dict['rob3_off'] =  onp.zeros((num_atom_types,
                                          num_atom_types),dtype=dtype)
  FF_field_dict['rob3_off_mask'] =  onp.zeros((num_atom_types,
                                               num_atom_types),dtype=jnp.bool_)

  FF_field_dict['p1co_off'] =  onp.zeros((num_atom_types,
                                          num_atom_types),dtype=dtype)
  FF_field_dict['p1co_off_mask'] =  onp.zeros((num_atom_types,
                                               num_atom_types),dtype=jnp.bool_)
  FF_field_dict['p2co_off'] =  onp.zeros((num_atom_types,
                                          num_atom_types),dtype=dtype)
  FF_field_dict['p2co_off_mask'] =  onp.zeros((num_atom_types,
                                               num_atom_types),dtype=jnp.bool_)
  FF_field_dict['p3co_off'] =  onp.zeros((num_atom_types,
                                          num_atom_types),dtype=dtype)
  FF_field_dict['p3co_off_mask'] =  onp.zeros((num_atom_types,
                                               num_atom_types),dtype=jnp.bool_)
  for i in range(num_off_diag):
    line = f.readline().strip()
    split_line = line.split()
    nodm1 = int(split_line[0])
    nodm2 = int(split_line[1])
    deodmh = float(split_line[2])
    rodmh = float(split_line[3])
    godmh = float(split_line[4])
    rsig = float(split_line[5])
    rpi = float(split_line[6])
    rpi2 = float(split_line[7])
    #TODO: handle the mapping of the "params" later
    nodm1 = nodm1 - 1 #index starts from 0
    nodm2 = nodm2 - 1 #index starts from 0
    FF_field_dict['rob1_off'][nodm1,nodm2] = rsig
    FF_field_dict['rob2_off'][nodm1,nodm2] = rpi
    FF_field_dict['rob3_off'][nodm1,nodm2] = rpi2
    FF_field_dict['p1co_off'][nodm1,nodm2] = rodmh
    FF_field_dict['p2co_off'][nodm1,nodm2] = deodmh
    FF_field_dict['p3co_off'][nodm1,nodm2] = godmh

    if (rsig > 0
        and FF_field_dict['rat'][nodm1] > 0
        and FF_field_dict['rat'][nodm2] > 0):
      FF_field_dict['rob1_off_mask'][nodm1,nodm2] = 1
      FF_param_to_index[(4,i+1,4)] = ("rob1_off", (nodm1,nodm2))

    if (rpi > 0
        and FF_field_dict['rapt'][nodm1] > 0
        and FF_field_dict['rapt'][nodm2] > 0):
      FF_field_dict['rob2_off_mask'][nodm1,nodm2] = 1
      FF_param_to_index[(4,i+1,5)] = ("rob2_off", (nodm1,nodm2))

    if (rpi2 > 0
        and FF_field_dict['vnq'][nodm1] > 0
        and FF_field_dict['vnq'][nodm2] > 0):
      FF_field_dict['rob3_off_mask'][nodm1,nodm2] = 1
      FF_param_to_index[(4,i+1,6)] = ("rob3_off", (nodm1,nodm2))
    if (rodmh > 0):
      FF_field_dict['p1co_off_mask'][nodm1,nodm2] = 1
      FF_param_to_index[(4,i+1,2)] = ("p1co_off", (nodm1,nodm2))
    if (deodmh > 0):
      FF_field_dict['p2co_off_mask'][nodm1,nodm2] = 1
      FF_param_to_index[(4,i+1,1)] = ("p2co_off", (nodm1,nodm2))
    if (godmh > 0):
      FF_field_dict['p3co_off_mask'][nodm1,nodm2] = 1
      FF_param_to_index[(4,i+1,3)] = ("p3co_off", (nodm1,nodm2))
  # valency angle parameters
  line = f.readline().strip()
  num_val_params = int(line.split()[0])
  FF_field_dict['body3_params_mask'] =  onp.zeros((num_atom_types,
                                                   num_atom_types,
                                                   num_atom_types),
                                                  dtype=jnp.bool_)
  FF_field_dict['th0'] = onp.zeros((num_atom_types,
                                    num_atom_types,
                                    num_atom_types),
                                   dtype=dtype)
  FF_field_dict['vka'] = onp.zeros((num_atom_types,
                                    num_atom_types,
                                    num_atom_types),
                                   dtype=dtype)
  FF_field_dict['vka3'] = onp.zeros((num_atom_types,
                                    num_atom_types,
                                    num_atom_types),
                                   dtype=dtype)
  FF_field_dict['vka8'] = onp.zeros((num_atom_types,
                                    num_atom_types,
                                    num_atom_types),
                                   dtype=dtype)
  FF_field_dict['vkac'] = onp.zeros((num_atom_types,
                                    num_atom_types,
                                    num_atom_types),
                                   dtype=dtype)
  FF_field_dict['vkap'] = onp.zeros((num_atom_types,
                                    num_atom_types,
                                    num_atom_types),
                                   dtype=dtype)
  FF_field_dict['vval2'] = onp.zeros((num_atom_types,
                                    num_atom_types,
                                    num_atom_types),
                                   dtype=dtype)

  for val in range(num_val_params):
    line = f.readline().strip()
    split_line = line.split()
    ind1 = int(split_line[0])
    ind2 = int(split_line[1])
    ind3 = int(split_line[2])

    th0 = float(split_line[3])
    vka = float(split_line[4])
    vka3 = float(split_line[5])
    vka8 = float(split_line[6])
    vkac = float(split_line[7])
    vkap = float(split_line[8])
    vval2 = float(split_line[9])

    ind1 = ind1 - 1 #index starts from 0
    ind2 = ind2 - 1 #index starts from 0
    ind3 = ind3 - 1 #index starts from 0
    FF_field_dict['th0'][ind1,ind2,ind3] = th0
    FF_field_dict['vka'][ind1,ind2,ind3] = vka
    FF_field_dict['vka3'][ind1,ind2,ind3] = vka3
    FF_field_dict['vka8'][ind1,ind2,ind3] = vka8
    FF_field_dict['vkac'][ind1,ind2,ind3] = vkac
    FF_field_dict['vkap'][ind1,ind2,ind3] = vkap
    FF_field_dict['vval2'][ind1,ind2,ind3] = vval2

    FF_param_to_index[(5,val+1,1)] = ("th0", (ind1,ind2,ind3))
    FF_param_to_index[(5,val+1,2)] = ("vka", (ind1,ind2,ind3))
    FF_param_to_index[(5,val+1,3)] = ("vka3", (ind1,ind2,ind3))
    FF_param_to_index[(5,val+1,4)] = ("vka8", (ind1,ind2,ind3))
    FF_param_to_index[(5,val+1,5)] = ("vkac", (ind1,ind2,ind3))
    FF_param_to_index[(5,val+1,6)] = ("vkap", (ind1,ind2,ind3))
    FF_param_to_index[(5,val+1,7)] = ("vval2", (ind1,ind2,ind3))

    body_3_indices_dst[0].append(ind3)
    body_3_indices_dst[1].append(ind2)
    body_3_indices_dst[2].append(ind1)

    body_3_indices_src[0].append(ind1)
    body_3_indices_src[1].append(ind2)
    body_3_indices_src[2].append(ind3)

    if abs(vka) > 0.001:
      FF_field_dict['body3_params_mask'][ind1,ind2,ind3] = 1.0
      FF_field_dict['body3_params_mask'][ind3,ind2,ind1] = 1.0

  # torsion parameters
  line = f.readline().strip()
  num_tors_params = int(line.split()[0])
  FF_field_dict['body34_params_mask'] =  onp.zeros((num_atom_types,
                                                   num_atom_types,
                                                   num_atom_types),
                                                  dtype=jnp.bool_)
  FF_field_dict['body4_params_mask'] =  onp.zeros((num_atom_types,
                                                   num_atom_types,
                                                   num_atom_types,
                                                   num_atom_types),
                                                  dtype=jnp.bool_)
  FF_field_dict['v1'] = onp.zeros((num_atom_types,
                                   num_atom_types,
                                   num_atom_types,
                                   num_atom_types),
                                  dtype=dtype)
  FF_field_dict['v2'] = onp.zeros((num_atom_types,
                                   num_atom_types,
                                   num_atom_types,
                                   num_atom_types),
                                  dtype=dtype)
  FF_field_dict['v3'] = onp.zeros((num_atom_types,
                                   num_atom_types,
                                   num_atom_types,
                                   num_atom_types),
                                  dtype=dtype)
  FF_field_dict['v4'] = onp.zeros((num_atom_types,
                                   num_atom_types,
                                   num_atom_types,
                                   num_atom_types),
                                  dtype=dtype)
  FF_field_dict['vconj'] = onp.zeros((num_atom_types,
                                   num_atom_types,
                                   num_atom_types,
                                   num_atom_types),
                                  dtype=dtype)
  torsion_param_sets = set()
  lines_with_negative_vals = []
  for tors in range(num_tors_params):
    line = f.readline().strip()
    split_line = line.split()
    ind1 = int(split_line[0])
    ind2 = int(split_line[1])
    ind3 = int(split_line[2])
    ind4 = int(split_line[3])

    v1 = float(split_line[4])
    v2 = float(split_line[5])
    v3 = float(split_line[6])
    v4 = float(split_line[7])
    vconj = float(split_line[8])
    #v2bo = float(split_line[9])
    #v3bo = float(split_line[10])

    ind1 = ind1 - 1 #index starts from 0
    ind2 = ind2 - 1 #index starts from 0
    ind3 = ind3 - 1 #index starts from 0
    ind4 = ind4 - 1 #index starts from 0

    # if all parameters are 0, skip
    if v1 == 0.0 and v2 == 0.0 and v3 == 0.0 and v4 == 0.0 and vconj == 0.0:
      continue

    # TODO: handle 0 indices in the param. file later
    if (ind1 > -1 and ind4 > -1):
      if (ind1,ind2,ind3,ind4) in torsion_param_sets:
        print(f"[WARNING] 4-body parameters for ({ind1+1},{ind2+1},{ind3+1},{ind4+1}) appeared twice!")
        print("Might cause numerical inaccuracies!")
        print("Skipping the dublicate occurance...")
        continue
      FF_field_dict['v1'][ind1,ind2,ind3,ind4] = v1
      FF_field_dict['v2'][ind1,ind2,ind3,ind4] = v2
      FF_field_dict['v3'][ind1,ind2,ind3,ind4] = v3
      FF_field_dict['v4'][ind1,ind2,ind3,ind4] = v4
      FF_field_dict['vconj'][ind1,ind2,ind3,ind4] = vconj

      FF_param_to_index[(6,tors+1,1)] = ("v1", (ind1,ind2,ind3,ind4))
      FF_param_to_index[(6,tors+1,2)] = ("v2", (ind1,ind2,ind3,ind4))
      FF_param_to_index[(6,tors+1,3)] = ("v3", (ind1,ind2,ind3,ind4))
      FF_param_to_index[(6,tors+1,4)] = ("v4", (ind1,ind2,ind3,ind4))
      FF_param_to_index[(6,tors+1,5)] = ("vconj", (ind1,ind2,ind3,ind4))

      FF_field_dict['body4_params_mask'][ind1,ind2,ind3,ind4] = 1
      FF_field_dict['body4_params_mask'][ind4,ind3,ind2,ind1] = 1

      FF_field_dict['body34_params_mask'][ind1,ind2,ind3] = 1
      FF_field_dict['body34_params_mask'][ind3,ind2,ind1] = 1
      FF_field_dict['body34_params_mask'][ind2,ind3,ind4] = 1
      FF_field_dict['body34_params_mask'][ind4,ind3,ind2] = 1

      body_4_indices_dst[0].append(ind4)
      body_4_indices_dst[1].append(ind3)
      body_4_indices_dst[2].append(ind2)
      body_4_indices_dst[3].append(ind1)

      body_4_indices_src[0].append(ind1)
      body_4_indices_src[1].append(ind2)
      body_4_indices_src[2].append(ind3)
      body_4_indices_src[3].append(ind4)
      torsion_param_sets.add((ind1,ind2,ind3,ind4))
      torsion_param_sets.add((ind4,ind3,ind2,ind1))

    elif (ind1 == -1 and ind4 == -1):
      lines_with_negative_vals.append([tors,ind1,ind2,ind3,ind4,v1,v2,v3,v4,vconj])
    else:
      print(f"Invalid torsion parameter section, line:{tors+1}")
      return None

  # the lines with negative values affect mutliple types, so they need to
  # be processed at the end
  # if line dedicated to [ind1, ind2, ind3, ind4] is not available
  # then [-1, line2, line3, -1] will be used instead
  for vals in lines_with_negative_vals:
    tors,ind1,ind2,ind3,ind4,v1,v2,v3,v4,vconj = vals
    # Last index is reserved for this part
    sel_ind = FF_field_dict['num_atom_types'] - 1
    FF_field_dict['v1'][sel_ind,ind2,ind3,sel_ind] = v1
    FF_field_dict['v2'][sel_ind,ind2,ind3,sel_ind] = v2
    FF_field_dict['v3'][sel_ind,ind2,ind3,sel_ind] = v3
    FF_field_dict['v4'][sel_ind,ind2,ind3,sel_ind] = v4
    FF_field_dict['vconj'][sel_ind,ind2,ind3,sel_ind] = vconj

    FF_param_to_index[(6,tors+1,1)] = ("v1", (sel_ind,ind2,ind3,sel_ind))
    FF_param_to_index[(6,tors+1,2)] = ("v2", (sel_ind,ind2,ind3,sel_ind))
    FF_param_to_index[(6,tors+1,3)] = ("v3", (sel_ind,ind2,ind3,sel_ind))
    FF_param_to_index[(6,tors+1,4)] = ("v4", (sel_ind,ind2,ind3,sel_ind))
    FF_param_to_index[(6,tors+1,5)] = ("vconj", (sel_ind,ind2,ind3,sel_ind))

    for i in range(real_num_atom_types):
      for j in range(real_num_atom_types):
        if FF_field_dict['body4_params_mask'][i,ind2,ind3,j] == 0:
          body_4_indices_src[0].append(sel_ind)
          body_4_indices_src[1].append(ind2)
          body_4_indices_src[2].append(ind3)
          body_4_indices_src[3].append(sel_ind)

          body_4_indices_dst[0].append(i)
          body_4_indices_dst[1].append(ind2)
          body_4_indices_dst[2].append(ind3)
          body_4_indices_dst[3].append(j)
          FF_field_dict['body4_params_mask'][i,ind2,ind3,j] = 1

          FF_field_dict['body34_params_mask'][i,ind2,ind3] = 1
          FF_field_dict['body34_params_mask'][ind3,ind2,i] = 1
          FF_field_dict['body34_params_mask'][ind2,ind3,j] = 1
          FF_field_dict['body34_params_mask'][j,ind3,ind2] = 1

        if FF_field_dict['body4_params_mask'][j,ind3,ind2,i] == 0:
          body_4_indices_src[0].append(sel_ind)
          body_4_indices_src[1].append(ind2)
          body_4_indices_src[2].append(ind3)
          body_4_indices_src[3].append(sel_ind)

          body_4_indices_dst[0].append(j)
          body_4_indices_dst[1].append(ind3)
          body_4_indices_dst[2].append(ind2)
          body_4_indices_dst[3].append(i)
          FF_field_dict['body4_params_mask'][j,ind3,ind2,i] = 1

          FF_field_dict['body34_params_mask'][i,ind2,ind3] = 1
          FF_field_dict['body34_params_mask'][ind3,ind2,i] = 1
          FF_field_dict['body34_params_mask'][ind2,ind3,j] = 1
          FF_field_dict['body34_params_mask'][j,ind3,ind2] = 1

  # hbond parameters
  line = f.readline().strip()
  num_hbond_params = int(line.split()[0])
  FF_field_dict['hb_params_mask'] =  onp.zeros((num_atom_types,
                                                num_atom_types,
                                                num_atom_types),
                                               dtype=jnp.bool_)
  FF_field_dict['rhb'] = onp.zeros((num_atom_types,
                                    num_atom_types,
                                    num_atom_types),
                                   dtype=dtype)
  FF_field_dict['dehb'] = onp.zeros((num_atom_types,
                                     num_atom_types,
                                     num_atom_types),
                                    dtype=dtype)
  FF_field_dict['vhb1'] = onp.zeros((num_atom_types,
                                     num_atom_types,
                                     num_atom_types),
                                    dtype=dtype)
  FF_field_dict['vhb2'] = onp.zeros((num_atom_types,
                                     num_atom_types,
                                     num_atom_types),
                                    dtype=dtype)
  for i in range(num_hbond_params):
    line = f.readline().strip()
    split_line = line.split()

    ind1 = int(split_line[0]) - 1
    ind2 = int(split_line[1]) - 1
    ind3 = int(split_line[2]) - 1

    rhb = float(split_line[3])
    dehb = float(split_line[4])
    vhb1 = float(split_line[5])
    vhb2 = float(split_line[6])

    FF_field_dict['rhb'][ind1,ind2,ind3] = rhb
    FF_field_dict['dehb'][ind1,ind2,ind3] = dehb
    FF_field_dict['vhb1'][ind1,ind2,ind3] = vhb1
    FF_field_dict['vhb2'][ind1,ind2,ind3] = vhb2
    FF_field_dict['hb_params_mask'][ind1,ind2,ind3] = 1

    FF_param_to_index[(7,i+1,1)] = ("rhb", (ind1,ind2,ind3))
    FF_param_to_index[(7,i+1,2)] = ("dehb", (ind1,ind2,ind3))
    FF_param_to_index[(7,i+1,3)] = ("vhb1", (ind1,ind2,ind3))
    FF_param_to_index[(7,i+1,4)] = ("vhb2", (ind1,ind2,ind3))

  f.close()

  for i in range(3):
    body_3_indices_src[i] = onp.array(body_3_indices_src[i],dtype=onp.int32)
    body_3_indices_dst[i] = onp.array(body_3_indices_dst[i],dtype=onp.int32)

  for i in range(4):
    body_4_indices_src[i] = onp.array(body_4_indices_src[i],dtype=onp.int32)
    body_4_indices_dst[i] = onp.array(body_4_indices_dst[i],dtype=onp.int32)

  FF_field_dict['body3_indices_src'] = tuple(body_3_indices_src)
  FF_field_dict['body3_indices_dst'] = tuple(body_3_indices_dst)
  FF_field_dict['body4_indices_src'] = tuple(body_4_indices_src)
  FF_field_dict['body4_indices_dst'] = tuple(body_4_indices_dst)

  #TODO: this function call is not needed after the energy function is refactored
  init_params_for_filler_atom_type(FF_field_dict)


  # placeholders for params to be filled later
  FF_field_dict['rob1'] = onp.zeros_like(FF_field_dict['rob1_off'])
  FF_field_dict['rob2'] = onp.zeros_like(FF_field_dict['rob1_off'])
  FF_field_dict['rob3'] = onp.zeros_like(FF_field_dict['rob1_off'])

  FF_field_dict['p1co'] = onp.zeros_like(FF_field_dict['p1co_off'])
  FF_field_dict['p2co'] = onp.zeros_like(FF_field_dict['p1co_off'])
  FF_field_dict['p3co'] = onp.zeros_like(FF_field_dict['p1co_off'])

  FF_field_dict['softcut_2d'] = onp.zeros_like(FF_field_dict['p1co_off'])

  FF_field_dict['params_to_indices'] = frozendict(FF_param_to_index)

  FF_fields = ForceField.__dataclass_fields__
  for k in FF_field_dict:
    is_static = k in FF_fields and FF_fields[k].metadata.get('static', False)
    if (type(FF_field_dict[k]) == onp.ndarray):
      FF_field_dict[k] = jnp.array(FF_field_dict[k])
    elif type(FF_field_dict[k]) == float:
      FF_field_dict[k] = jnp.array(FF_field_dict[k], dtype=dtype)

  force_field = ForceField.init_from_arg_dict(FF_field_dict)

  return force_field
