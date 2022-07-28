"""
Contains helper functions ReaxFF

Author: Mehmet Cagri Kaymak
"""

import jax
import jax.numpy as jnp
import numpy as onp
from jax_md.reaxff_forcefield import ForceField
from jax_md.reaxff_forcefield import CLOSE_NEIGH_CUTOFF
from dataclasses import fields

# it fixes nan values issue, from: https://github.com/google/jax/issues/1052
def vectorized_cond(pred, true_fun, false_fun, operand):
  # true_fun and false_fun must act elementwise (i.e. be vectorized)
  true_op = jnp.where(pred, operand, 0)
  false_op = jnp.where(pred, 0, operand)
  return jnp.where(pred, true_fun(true_op), false_fun(false_op))

def calculate_bo_single(distance,
                rob1, rob2, rob3,
                ptp, pdp, popi, pdo, bop1, bop2,
                cutoff):
  '''
  This function is needed to find the cutoff for the bonded interactions
  during parameter optimization

  to get the highest bor (bond order potential):
      rob1:
      bop2: group 2, line 2, col. 6
      bop1: group 2, line 2, col. 5

      rob2:
      ptp: group 2, line 2, col. 3
      pdp: group 2, line 2, col. 2

      rob3:
      popi: group 2, line 1, col. 7
      pdo: group 2, line 1, col. 5
  '''

  rhulp = jnp.where(rob1 <= 0, 0, distance / rob1)
  rh2 = rhulp ** bop2
  ehulp = (1 + cutoff) * jnp.exp(bop1 * rh2)
  ehulp = jnp.where(rob1 <= 0, 0.0, ehulp)

  rhulp2 = jnp.where(rob2 == 0, 0, distance / rob2)
  rh2p = rhulp2 ** ptp
  ehulpp = jnp.exp(pdp * rh2p)
  ehulpp = jnp.where(rob2 <= 0, 0.0, ehulpp)


  rhulp3 = jnp.where(rob3 == 0, 0, distance / rob3)
  rh2pp = rhulp3 ** popi
  ehulppp = jnp.exp(pdo*rh2pp)
  ehulppp = jnp.where(rob3 <= 0, 0.0, ehulppp)

  #print(ehulp , ehulpp , ehulppp)
  bor = ehulp + ehulpp + ehulppp


  return bor # if < cutoff, will be ignored

def find_limits(type1, type2, force_field, cutoff):
  vect_bo_function = jax.jit(jax.vmap(calculate_bo_single,
                          in_axes=(0,None,None,None,None,
                                   None,None,None,
                                   None,None,None)),backend='cpu')
  rob1 = force_field.rob1[type1,type2] # select max, typical values (0.1, 2)
  bop2 = force_field.bop2[type1,type2] # select min, typical values (1,10)
  bop1 = force_field.bop1[type1,type2] # select max, typical values (-0.2, -0.01)

  rob2 = force_field.rob2[type1,type2]
  ptp = force_field.ptp[type1,type2]
  pdp = force_field.pdp[type1,type2]

  rob3 = force_field.rob3[type1,type2]
  popi = force_field.popi[type1,type2]
  pdo = force_field.pdo[type1,type2]

  distance = jnp.linspace(0.0, 10, 1000)

  res = vect_bo_function(distance,
                  rob1, rob2, rob3,
                  ptp, pdp, popi, pdo, bop1, bop2,
                  cutoff)

  ind = jnp.sum(res > cutoff)

  return distance[ind]

def find_all_cutoffs(force_field,cutoff,atom_indices):
  lenn = len(atom_indices)
  cutoff_dict = dict()
  for i in range(lenn):
    type_i = atom_indices[i]
    for j in range(i,lenn):
      type_j = atom_indices[j]
      dist = find_limits(type_i,type_j, force_field, cutoff)
      cutoff_dict[(type_i,type_j)] = dist
      cutoff_dict[(type_j,type_i)] = dist

      if dist > CLOSE_NEIGH_CUTOFF and type_i != -1 and type_j!=-1: #-1 TYPE is for the filler atom
        dist = round(dist,2)
        print(f"[WARNING] between type {type_i} and type {type_j}" +
              "the bond length could be greater than {CLOSE_NEIGH_CUTOFF} A! ({dist} A)")
        cutoff_dict[(type_i,type_j)] = CLOSE_NEIGH_CUTOFF
        cutoff_dict[(type_j,type_i)] = CLOSE_NEIGH_CUTOFF
  return cutoff_dict

def parse_and_save_force_field(old_ff_file, new_ff_file,force_field):
  output = ""
  f = open(old_ff_file, 'r')
  line = f.readline()
  output = output + line
  header = line.strip()

  line = f.readline()
  output = output + line
  num_params = int(line.strip().split()[0])
  global_params = jnp.zeros(shape=(num_params,1), dtype=jnp.float64)
  ff = force_field
  for i in range(num_params):
    line = f.readline()
    line = list(line)
    #-------------------------------------------------------------
    if i == 0:
      line[:10] = "{:10.4f}".format(ff.over_coord1[0])  #overcoord1
    if i == 1:
      line[:10] = "{:10.4f}".format(ff.over_coord2[0]) #overcoord2
    #-------------------------------------------------------------

    #-------------------------------------------------------------
    if i == 3:
      line[:10] = "{:10.4f}".format(ff.trip_stab4[0])  #trip_stab4
    if i == 4:
      line[:10] = "{:10.4f}".format(ff.trip_stab5[0]) #trip_stab5
    if i == 7:
      line[:10] = "{:10.4f}".format(ff.trip_stab8[0])  #trip_stab8
    if i == 10:
      line[:10] = "{:10.4f}".format(ff.trip_stab11[0]) #trip_stab11
    #-------------------------------------------------------------
    #valency related parameters
    if i == 2:
      line[:10] = "{:10.4f}".format(ff.val_par3[0])  #val_par3
    if i == 14:
      line[:10] = "{:10.4f}".format(ff.val_par15[0]) #val_par15
    if i == 15:
      line[:10] = "{:10.4f}".format(ff.par_16[0])  #par_16
    if i == 16:
      line[:10] = "{:10.4f}".format(ff.val_par17[0]) #val_par17
    if i == 17:
      line[:10] = "{:10.4f}".format(ff.val_par18[0])  #val_par18
    if i == 19:
      line[:10] = "{:10.4f}".format(ff.val_par20[0]) #val_par20
    if i == 20:
      line[:10] = "{:10.4f}".format(ff.val_par21[0])  #val_par21
    if i == 30:
      line[:10] = "{:10.4f}".format(ff.val_par31[0]) #val_par31
    if i == 33:
      line[:10] = "{:10.4f}".format(ff.val_par34[0])  #val_par34
    if i == 38:
      line[:10] = "{:10.4f}".format(ff.val_par39[0]) #val_par39
    #-------------------------------------------------------------

    #-------------------------------------------------------------
    #over-under coord.
    if i == 5:
      line[:10] = "{:10.4f}".format(ff.par_6[0])  #par_6
    if i == 6:
      line[:10] = "{:10.4f}".format(ff.par_7[0]) #par_7
    if i == 8:
      line[:10] = "{:10.4f}".format(ff.par_9[0])  #par_9
    if i == 9:
      line[:10] = "{:10.4f}".format(ff.par_10[0]) #par_10
    if i == 31:
      line[:10] = "{:10.4f}".format(ff.par_32[0])  #par_32
    if i == 32:
      line[:10] = "{:10.4f}".format(ff.par_33[0]) #par_33

    #-------------------------------------------------------------
    #torsion
    if i == 23:
      line[:10] = "{:10.4f}".format(ff.par_24[0])  #par_24
    if i == 24:
      line[:10] = "{:10.4f}".format(ff.par_25[0]) #par_25
    if i == 25:
      line[:10] = "{:10.4f}".format(ff.par_26[0])  #par_26
    if i == 27:
      line[:10] = "{:10.4f}".format(ff.par_28[0]) #par_28

    #-------------------------------------------------------------
    # vdw
    if i == 28:
      line[:10] = "{:10.4f}".format(ff.vdw_shiedling[0]) #vdw_shiedling
    output = output + ''.join(line)

  line = f.readline()
  output = output + line

  num_atom_types = int(line.strip().split()[0])
  # skip 3 lines of comment
  output = output + f.readline()
  output = output + f.readline()
  output = output + f.readline()

  atom_names = []
  line_ctr = 0
  for i in range(num_atom_types):
    # first line
    line = f.readline()
    line = list(line)
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.rat[i]) #rat - rob1
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.rvdw[i]) #rvdw
    line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.eps[i]) #eps
    line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.gamma[i]) #gamma
    line[3 + 9 * 6:3 + 9 * 7] = "{:9.4f}".format(ff.rapt[i]) #rapt - rob2
    line[3 + 9 * 7:3 + 9 * 8] = "{:9.4f}".format(ff.stlp[i]) #stlp

    output = output + ''.join(line)

    # second line
    line = f.readline()
    line = list(line)
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.alf[i]) #alf
    line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vop[i]) #vop
    line[3 + 9 * 2:3 + 9 * 3] = "{:9.4f}".format(ff.valf[i]) #valf
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.valp1[i]) #valp1
    line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.electronegativity[i])
    line[3 + 9 * 6:3 + 9 * 7] = "{:9.4f}".format(ff.idempotential[i])

    output = output + ''.join(line)
    # third line
    line = f.readline()
    line = list(line)
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.vnq[i]) #vnq - rob3
    line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vlp1[i]) #vlp1
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.bo131[i]) #bo131
    line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.bo132[i]) #bo132
    line[3 + 9 * 5:3 + 9 * 6] = "{:9.4f}".format(ff.bo133[i]) #bo133

    output = output + ''.join(line)

    # fourth line
    line = f.readline()
    line = list(line)
    line[3 + 9 * 0:3 + 9 * 1] = "{:9.4f}".format(ff.vovun[i])
    line[3 + 9 * 1:3 + 9 * 2] = "{:9.4f}".format(ff.vval1[i])
    line[3 + 9 * 3:3 + 9 * 4] = "{:9.4f}".format(ff.vval3[i])
    line[3 + 9 * 4:3 + 9 * 5] = "{:9.4f}".format(ff.vval4[i])

    output = output + ''.join(line)



  line = f.readline()  # num_bonds
  output = output + line

  line = line.strip()
  num_bonds = int(line.split()[0])
  output = output + f.readline() # skip next line (comment)
  for _ in range(num_bonds):
    # first line
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i = int(tmp[0]) - 1 # index starts at 0
    j = int(tmp[1]) - 1

    line[6 + 9 * 0:6 + 9 * 1] = "{:9.4f}".format(ff.de1[i,j])
    line[6 + 9 * 1:6 + 9 * 2] = "{:9.4f}".format(ff.de2[i,j])
    line[6 + 9 * 2:6 + 9 * 3] = "{:9.4f}".format(ff.de3[i,j])
    line[6 + 9 * 3:6 + 9 * 4] = "{:9.4f}".format(ff.psi[i,j])
    line[6 + 9 * 4:6 + 9 * 5] = "{:9.4f}".format(ff.pdo[i,j])
    line[6 + 9 * 5:6 + 9 * 6] = "{:9.4f}".format(ff.v13cor[i,j])
    line[6 + 9 * 6:6 + 9 * 7] = "{:9.4f}".format(ff.popi[i,j])
    line[6 + 9 * 7:6 + 9 * 8] = "{:9.4f}".format(ff.vover[i,j])
    #print(''.join(line))
    output = output + ''.join(line)
    # second line
    line = f.readline()
    line = list(line)

    line[6 + 9 * 0:6 + 9 * 1] = "{:9.4f}".format(ff.psp[i,j])
    line[6 + 9 * 1:6 + 9 * 2] = "{:9.4f}".format(ff.pdp[i,j])
    line[6 + 9 * 2:6 + 9 * 3] = "{:9.4f}".format(ff.ptp[i,j])
    line[6 + 9 * 4:6 + 9 * 5] = "{:9.4f}".format(ff.bop1[i,j])
    line[6 + 9 * 5:6 + 9 * 6] = "{:9.4f}".format(ff.bop2[i,j])
    line[6 + 9 * 6:6 + 9 * 7] = "{:9.4f}".format(ff.ovc[i,j])
    #print(''.join(line))
    output = output + ''.join(line)
  line = f.readline()  # num_off_diag
  output = output + line

  line = line.strip()
  num_off_diag = int(line.split()[0])

  for _ in range(num_off_diag):
    # first line
    # first line
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i = int(tmp[0]) - 1 # index starts at 0
    j = int(tmp[1]) - 1

    line[6 + 9 * 0:6 + 9 * 1] = "{:9.4f}".format(ff.p2co_off[i,j])
    line[6 + 9 * 1:6 + 9 * 2] = "{:9.4f}".format(ff.p1co_off[i,j])  # was /2
    line[6 + 9 * 2:6 + 9 * 3] = "{:9.4f}".format(ff.p3co_off[i,j])
    line[6 + 9 * 3:6 + 9 * 4] = "{:9.4f}".format(ff.rob1_off[i,j])
    line[6 + 9 * 4:6 + 9 * 5] = "{:9.4f}".format(ff.rob2_off[i,j])
    line[6 + 9 * 5:6 + 9 * 6] = "{:9.4f}".format(ff.rob3_off[i,j])

    output = output + ''.join(line)

  #valency angle parameters
  line = f.readline()  # num_val_params
  output = output + line

  line = line.strip()
  num_val_params = int(line.split()[0])

  for _ in range(num_val_params):
    # first line
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i = int(tmp[0]) - 1 # index starts at 0
    j = int(tmp[1]) - 1
    k = int(tmp[2]) - 1

    line[9 + 9 * 0:9 + 9 * 1] = "{:9.4f}".format(ff.th0[i,j,k])
    line[9 + 9 * 1:9 + 9 * 2] = "{:9.4f}".format(ff.vka[i,j,k])
    line[9 + 9 * 2:9 + 9 * 3] = "{:9.4f}".format(ff.vka3[i,j,k])
    line[9 + 9 * 3:9 + 9 * 4] = "{:9.4f}".format(ff.vka8[i,j,k])
    line[9 + 9 * 4:9 + 9 * 5] = "{:9.4f}".format(ff.vkac[i,j,k])
    line[9 + 9 * 5:9 + 9 * 6] = "{:9.4f}".format(ff.vkap[i,j,k])
    line[9 + 9 * 6:9 + 9 * 7] = "{:9.4f}".format(ff.vval2[i,j,k])

    output = output + ''.join(line)

  #torsion parameters
  line = f.readline()  # num_tors_params
  output = output + line

  line = line.strip()
  num_tors_params = int(line.split()[0])

  for _ in range(num_tors_params):
    # first line
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i1 = int(tmp[0]) - 1 # index starts at 0
    i2 = int(tmp[1]) - 1
    i3 = int(tmp[2]) - 1
    i4 = int(tmp[3]) - 1

    if i1 != -1 and i4 != -1:
      line[12 + 9 * 0:12 + 9 * 1] = "{:9.4f}".format(ff.v1[i1,i2,i3,i4])
      line[12 + 9 * 1:12 + 9 * 2] = "{:9.4f}".format(ff.v2[i1,i2,i3,i4])
      line[12 + 9 * 2:12 + 9 * 3] = "{:9.4f}".format(ff.v3[i1,i2,i3,i4])
      line[12 + 9 * 3:12 + 9 * 4] = "{:9.4f}".format(ff.v4[i1,i2,i3,i4])
      line[12 + 9 * 4:12 + 9 * 5] = "{:9.4f}".format(ff.vconj[i1,i2,i3,i4])

    if i1 == -1 and i4 == -1:
      sel_ind = force_field.num_atom_types - 1
      line[12 + 9 * 0:12 + 9 * 1] = "{:9.4f}".format(ff.v1[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 1:12 + 9 * 2] = "{:9.4f}".format(ff.v2[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 2:12 + 9 * 3] = "{:9.4f}".format(ff.v3[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 3:12 + 9 * 4] = "{:9.4f}".format(ff.v4[sel_ind,
                                                           i2,
                                                           i3,
                                                           sel_ind])
      line[12 + 9 * 4:12 + 9 * 5] = "{:9.4f}".format(ff.vconj[sel_ind,
                                                              i2,
                                                              i3,
                                                              sel_ind])
    output = output + ''.join(line)

  # hbond parameters
  #torsion parameters
  line = f.readline()  # num_tors_params
  output = output + line

  line = line.strip()
  num_hbond_params = int(line.split()[0])

  for i in range(num_hbond_params):
    line = f.readline()
    tmp = line.strip().split()
    line = list(line)
    i1 = int(tmp[0]) - 1
    i2 = int(tmp[1]) - 1
    i3 = int(tmp[2]) -1
    line[9 + 9 * 0:9 + 9 * 1] = "{:9.4f}".format(ff.rhb[i1,i2,i3])
    line[9 + 9 * 1:9 + 9 * 2] = "{:9.4f}".format(ff.dehb[i1,i2,i3])
    line[9 + 9 * 2:9 + 9 * 3] = "{:9.4f}".format(ff.vhb1[i1,i2,i3])
    line[9 + 9 * 3:9 + 9 * 4] = "{:9.4f}".format(ff.vhb2[i1,i2,i3])
    output = output + ''.join(line)



  # need to append some extra lines because of 0 values
  for line in f:
    output = output + line

  file_new = open(new_ff_file,"w")
  file_new.write(output)
  file_new.close()

  f.close()


def init_params_for_filler_atom_type(FF_field_dict):
  #TODO: make sure that index -1 doesnt belong to a real atom!!!
  FF_field_dict['rat'][-1] = 1
  FF_field_dict['rapt'][-1] = 1
  FF_field_dict['vnq'][-1] = 1

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

def read_force_field(force_field_file, cutoff2, dtype=jnp.float32):
  # to store all arguments together before creating the class
  FF_field_dict = {f.name:None for f in fields(ForceField) if f.init}

  f = open(force_field_file, 'r')
  header = f.readline().strip()

  num_params = int(f.readline().strip().split()[0])
  global_params = onp.zeros(shape=(num_params,1), dtype=onp.float64)
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
  FF_field_dict['low_tap_rad'] = global_params[11]
  FF_field_dict['up_tap_rad'] = global_params[12]
  FF_field_dict['vdw_shiedling'] = global_params[28]
  FF_field_dict['cutoff'] = global_params[29] * 0.01
  FF_field_dict['cutoff2'] = cutoff2
  FF_field_dict['over_coord1'] = global_params[0]
  FF_field_dict['over_coord2'] = global_params[1]
  FF_field_dict['trip_stab4'] = global_params[3]
  FF_field_dict['trip_stab5'] = global_params[4]
  FF_field_dict['trip_stab8'] = global_params[7]
  FF_field_dict['trip_stab11'] = global_params[10]

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

  # over under
  FF_field_dict['par_6'] = global_params[5]
  FF_field_dict['par_7'] = global_params[6]
  FF_field_dict['par_9'] = global_params[8]
  FF_field_dict['par_10'] = global_params[9]
  FF_field_dict['par_32'] = global_params[31]
  FF_field_dict['par_33'] = global_params[32]

  # torsion par_24,par_25, par_26,par_28
  FF_field_dict['par_24'] = global_params[23]
  FF_field_dict['par_25'] = global_params[24]
  FF_field_dict['par_26'] = global_params[25]
  FF_field_dict['par_28'] = global_params[27]


  real_num_atom_types = int(f.readline().strip().split()[0])
  num_atom_types = real_num_atom_types + 1 # 1 extra to store dummy atoms
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


    FF_field_dict['name_to_index'] = name_to_index
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

    # third line
    line = f.readline().strip()
    split_line = line.split()
    FF_field_dict['vnq'][i] = float(split_line[0])
    FF_field_dict['vlp1'][i] = float(split_line[1])
    FF_field_dict['bo131'][i] = float(split_line[3])
    FF_field_dict['bo132'][i] = float(split_line[4])
    FF_field_dict['bo133'][i] = float(split_line[5])

    # fourth line
    line = f.readline().strip()
    split_line = line.split()
    FF_field_dict['vovun'][i] = float(split_line[0]) #over-under coord
    FF_field_dict['vval1'][i] = float(split_line[1])
    FF_field_dict['vval3'][i] = float(split_line[3])
    FF_field_dict['vval4'][i] = float(split_line[4])

    # This part is moved to the related part in energy calculation
    #if FF_field_dict['amas'][i] < 21.0:
    #    FF_field_dict['vval3'][i] = FF_field_dict['valf'][i]

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

    if (rpi > 0
        and FF_field_dict['rapt'][nodm1] > 0
        and FF_field_dict['rapt'][nodm2] > 0):
      FF_field_dict['rob2_off_mask'][nodm1,nodm2] = 1

    if (rpi2 > 0
        and FF_field_dict['vnq'][nodm1] > 0
        and FF_field_dict['vnq'][nodm2] > 0):
      FF_field_dict['rob3_off_mask'][nodm1,nodm2] = 1

    if (rodmh > 0):
      FF_field_dict['p1co_off_mask'][nodm1,nodm2] = 1

    if (deodmh > 0):
      FF_field_dict['p2co_off_mask'][nodm1,nodm2] = 1

    if (godmh > 0):
      FF_field_dict['p3co_off_mask'][nodm1,nodm2] = 1

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


    # TODO: handle 0 indices in the param. file later
    if (ind1 > -1 and ind4 > -1):
      FF_field_dict['v1'][ind1,ind2,ind3,ind4] = v1
      FF_field_dict['v2'][ind1,ind2,ind3,ind4] = v2
      FF_field_dict['v3'][ind1,ind2,ind3,ind4] = v3
      FF_field_dict['v4'][ind1,ind2,ind3,ind4] = v4
      FF_field_dict['vconj'][ind1,ind2,ind3,ind4] = vconj
      FF_field_dict['body4_params_mask'][ind1,ind2,ind3,ind4] = 1
      FF_field_dict['body4_params_mask'][ind4,ind3,ind2,ind1] = 1

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
      lines_with_negative_vals.append([ind1,ind2,ind3,ind4,v1,v2,v3,v4,vconj])
    else:
      print(f"Invalid torsion parameter section, line:{tors+1}")
      return None

  # the lines with negative values affect mutliple types, so they need to
  # be processed at the end
  # if line dedicated to [ind1, ind2, ind3, ind4] is not available
  # then [-1, line2, line3, -1] will be used instead
  for vals in lines_with_negative_vals:
    ind1,ind2,ind3,ind4,v1,v2,v3,v4,vconj = vals
    # Last index is reserved for this part
    sel_ind = FF_field_dict['num_atom_types'] - 1

    for i in range(real_num_atom_types):
      for j in range(real_num_atom_types):
        if (i,ind2,ind3,j) not in torsion_param_sets:
          body_4_indices_src[0].append(sel_ind)
          body_4_indices_src[1].append(ind2)
          body_4_indices_src[2].append(ind3)
          body_4_indices_src[3].append(sel_ind)

          body_4_indices_dst[0].append(i)
          body_4_indices_dst[1].append(ind2)
          body_4_indices_dst[2].append(ind3)
          body_4_indices_dst[3].append(j)

          body_4_indices_src[0].append(sel_ind)
          body_4_indices_src[1].append(ind2)
          body_4_indices_src[2].append(ind3)
          body_4_indices_src[3].append(sel_ind)

          body_4_indices_dst[0].append(j)
          body_4_indices_dst[1].append(ind3)
          body_4_indices_dst[2].append(ind2)
          body_4_indices_dst[3].append(i)

          FF_field_dict['v1'][sel_ind,ind2,ind3,sel_ind] = v1
          FF_field_dict['v2'][sel_ind,ind2,ind3,sel_ind] = v2
          FF_field_dict['v3'][sel_ind,ind2,ind3,sel_ind] = v3
          FF_field_dict['v4'][sel_ind,ind2,ind3,sel_ind] = v4
          FF_field_dict['vconj'][sel_ind,ind2,ind3,sel_ind] = vconj
          FF_field_dict['body4_params_mask'][i,ind2,ind3,j] = 1
          FF_field_dict['body4_params_mask'][j,ind3,ind2,i] = 1

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
    ind3 = int(split_line[2]) -1

    rhb = float(split_line[3])
    dehb = float(split_line[4])
    vhb1 = float(split_line[5])
    vhb2 = float(split_line[6])

    FF_field_dict['rhb'][ind1,ind2,ind3] = rhb
    FF_field_dict['dehb'][ind1,ind2,ind3] = dehb
    FF_field_dict['vhb1'][ind1,ind2,ind3] = vhb1
    FF_field_dict['vhb2'][ind1,ind2,ind3] = vhb2
    FF_field_dict['hb_params_mask'][ind1,ind2,ind3] = 1

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

  FF_fields = ForceField.__dataclass_fields__
  for k in FF_field_dict:
    is_static = k in FF_fields and FF_fields[k].metadata.get('static', False)
    if (type(FF_field_dict[k]) == onp.ndarray
        or type(FF_field_dict[k]) == float):
      FF_field_dict[k] = jnp.array(FF_field_dict[k],dtype=dtype)

  force_field = ForceField.init_from_arg_dict(FF_field_dict)

  return force_field
