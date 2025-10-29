import jax
jax.config.update("jax_enable_x64", True)
import numpy as onp
import jax.numpy as jnp
from ase.io import read
from jax_md import space
from jax_md.reaxff.reaxff_interactions import reaxff_inter_list
from jax_md.reaxff.reaxff_interactions import calculate_all_angles_and_distances
from jax_md.reaxff.reaxff_helper import read_force_field
from jax_md.reaxff.reaxff_forcefield import ForceField
from absl.testing import parameterized
import jax_md.reaxff.reaxff_energy as reaxff_energy
from jax_md.test_util import JAXMDTestCase
import os
import json

ATOL = 1e-5 # per atom
RTOL = 1e-5


TEST_DATA = []
def read_test_data(test_folder):
  for root, sub_dirs, _ in os.walk(test_folder):
    for dir in sub_dirs:
      test_name = dir
      ffield_path = f"{root}/{dir}/ffield"
      geo_path = f"{root}/{dir}/geo.pdb"
      results_path = f"{root}/{dir}/results.json"
      with open(results_path, 'r') as f:
        results = json.load(f)
      item = {"name":test_name, "ffield_path":ffield_path,
              "geo_path":geo_path, "dtype":jnp.float64,
              "results":results,
              "preds":dict()}
      TEST_DATA.append(item)
# The file paths for force fields and geometries will be provided here
# as well as the target data to match.

read_test_data("tests/data/reax_data")

def read_and_process_FF_file(filename, cutoff2 = 0.001, dtype=jnp.float64):
  force_field = read_force_field(filename,cutoff2 = cutoff2, dtype=dtype)
  force_field = ForceField.fill_off_diag(force_field)
  force_field = ForceField.fill_symm(force_field)

  return force_field

class ReaxFFEnergyTest(JAXMDTestCase):
  @classmethod
  def setUpClass(cls):
    cls.values = [0 for _ in TEST_DATA]

    cls.ffields = [read_and_process_FF_file(test["ffield_path"],
                                             dtype=test["dtype"])
                    for test in TEST_DATA]
    cls.geos = [read(test["geo_path"]) for test in TEST_DATA]
    cls.names = [test["name"] for test in TEST_DATA]
    cls.dtypes = [test["dtype"] for test in TEST_DATA]
    cls.nbr_lists = []
    cls.angles_and_dists = []
    cls.species = []
    cls.atomic_nums = []

    for i in range(len(cls.geos)):
      geo = cls.geos[i]
      ffield = cls.ffields[i]
      dtype = cls.dtypes[i]
      positions = (geo.positions - onp.min(geo.positions,axis=0))
      R = jnp.array(positions, dtype=cls.dtypes[i])
      types = geo.get_chemical_symbols()
      types_int = [ffield.name_to_index[t] for t in types]
      species = jnp.array(types_int)
      atomic_nums = jnp.array(geo.get_atomic_numbers())

      if (onp.all(geo.pbc)):
        box_size = jnp.array(geo.cell.lengths(),dtype=dtype)
      else:
        diff = jnp.max(R,axis=0) - jnp.min(R,axis=0)
        box_size = jnp.max(diff) + 20

      displacement, shift = space.periodic(box_size)

      reaxff_inter_fn, energy_fn = reaxff_inter_list(displacement,
                                                      box_size,
                                                      species,
                                                      atomic_nums,
                                                      ffield,
                                                      tol=1e-14)

      metric = space.metric(displacement)
      map_metric = space.map_neighbor(metric)
      map_disp = space.map_neighbor(displacement)
      nbr_lists = reaxff_inter_fn.allocate(R)

      angles_and_dists = calculate_all_angles_and_distances(R,
                                                        nbr_lists,
                                                        map_metric,
                                                        map_disp)
      cls.species.append(species)
      cls.atomic_nums.append(atomic_nums)
      cls.nbr_lists.append(nbr_lists)
      cls.angles_and_dists.append(angles_and_dists)

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_coulomb(self, i, name):
    N = len(self.species[i])
    atom_mask = jnp.ones(N, dtype=jnp.bool_)
    far_nbr_inds = self.nbr_lists[i].far_nbrs.idx
    far_neigh_types = self.species[i][far_nbr_inds]
    far_nbr_dists = self.angles_and_dists[i][1]
    far_nbr_dists = far_nbr_dists * (far_nbr_inds != N)

    tapered_dists = reaxff_energy.taper(far_nbr_dists, 0.0, 10.0)
    tapered_dists = jnp.where((far_nbr_dists > 10.0) | (far_nbr_dists < 0.001),
                              0.0,
                              tapered_dists)

    # shared accross charge calc and coulomb
    gamma = jnp.power(self.ffields[i].gamma.reshape(-1, 1), 3/2)
    gamma_mat = gamma * gamma.transpose()
    gamma_mat = gamma_mat[far_neigh_types, self.species[i].reshape(-1, 1)]
    hulp1_mat = far_nbr_dists ** 3 + (1/gamma_mat)
    hulp2_mat = jnp.power(hulp1_mat, 1.0/3.0)

    charges = reaxff_energy.calculate_eem_charges(self.species[i],
                                    atom_mask,
                                    far_nbr_inds,
                                    hulp2_mat,
                                    tapered_dists,
                                    self.ffields[i].idempotential,
                                    self.ffields[i].electronegativity,
                                    None,
                                    0.0,
                                    False,
                                    1e-14)

    cou_pot = reaxff_energy.calculate_coulomb_pot(far_nbr_inds,
                                    atom_mask,
                                    hulp2_mat,
                                    tapered_dists,
                                    charges[:-1])

    charge_pot = reaxff_energy.calculate_charge_energy(self.species[i],
                                         charges[:-1],
                                         self.ffields[i].idempotential,
                                         self.ffields[i].electronegativity)
    TEST_DATA[i]["preds"]["E_coul"] = cou_pot
    TEST_DATA[i]["preds"]["E_charge"] = charge_pot
    self.assertAllClose(cou_pot, TEST_DATA[i]["results"]["E_coul"],
                        atol=ATOL*N,rtol=RTOL)
    self.assertAllClose(charge_pot, TEST_DATA[i]["results"]["E_charge"],
                        atol=ATOL*N,rtol=RTOL)

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_vdw(self, i, name):
    N = len(self.species[i])
    atom_mask = jnp.ones(N, dtype=jnp.bool_)
    far_nbr_inds = self.nbr_lists[i].far_nbrs.idx
    far_nbr_dists = self.angles_and_dists[i][1]
    far_nbr_dists = far_nbr_dists * (far_nbr_inds != N)
    far_nbr_mask = ((far_nbr_inds != N)
                    & (atom_mask.reshape(-1,1) & atom_mask[far_nbr_inds]))
    tapered_dists = reaxff_energy.taper(far_nbr_dists, 0.0, 10.0)
    tapered_dists = jnp.where((far_nbr_dists > 10.0) | (far_nbr_dists < 0.001),
                              0.0,
                              tapered_dists)

    vdw_pot = reaxff_energy.calculate_vdw_pot(self.species[i],
                                far_nbr_mask,
                                far_nbr_inds,
                                far_nbr_dists,
                                tapered_dists,
                                self.ffields[i])
    TEST_DATA[i]["preds"]["E_vdw"] = vdw_pot
    self.assertAllClose(vdw_pot, TEST_DATA[i]["results"]["E_vdw"],
                        atol=ATOL*N,rtol=RTOL)

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_2_body(self,i, name):
    N = len(self.species[i])
    atom_inds = jnp.arange(N).reshape(-1,1)
    close_nbr_inds = self.nbr_lists[i].close_nbrs.idx[atom_inds,
                                                  self.nbr_lists[i].filter2.idx]
    close_nbr_inds = jnp.where(self.nbr_lists[i].filter2.idx != -1,
                                   close_nbr_inds,
                                   N)
    #close_nbr_inds = self.nbr_lists[i].close_nbrs.idx
    close_nbr_dists = self.angles_and_dists[i][0]
    atomic_num1 = self.atomic_nums[i].reshape(-1, 1)
    atomic_num2 = self.atomic_nums[i][close_nbr_inds]
    # O: 8, C:6
    triple_bond1 = jnp.logical_and(atomic_num1 == 8, atomic_num2 == 6)
    triple_bond2 = jnp.logical_and(atomic_num1 == 6, atomic_num2 == 8)
    triple_bond = jnp.logical_or(triple_bond1, triple_bond2)

    [cov_pot,
     bo,
     bopi,
     bopi2,
     abo] = reaxff_energy.calculate_covbon_pot(close_nbr_inds,
                                                           close_nbr_dists,
                                                           close_nbr_inds != N,
                                                           self.species[i],
                                                           triple_bond,
                                                           self.ffields[i])
    TEST_DATA[i]["preds"]["E_bonded"] = cov_pot
    self.assertAllClose(cov_pot, TEST_DATA[i]["results"]["E_bonded"],
                        atol=ATOL*N,rtol=RTOL)

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_lone_pair(self,i, name):
    N = len(self.species[i])
    atom_mask = jnp.ones(N, dtype=jnp.bool_)
    atom_inds = jnp.arange(N).reshape(-1,1)
    close_nbr_inds = self.nbr_lists[i].close_nbrs.idx[atom_inds,
                                                  self.nbr_lists[i].filter2.idx]
    close_nbr_inds = jnp.where(self.nbr_lists[i].filter2.idx != -1,
                                   close_nbr_inds,
                                   N)
    #close_nbr_inds = self.nbr_lists[i].close_nbrs.idx
    close_nbr_dists = self.angles_and_dists[i][0]
    atomic_num1 = self.atomic_nums[i].reshape(-1, 1)
    atomic_num2 = self.atomic_nums[i][close_nbr_inds]
    # O: 8, C:6
    triple_bond1 = jnp.logical_and(atomic_num1 == 8, atomic_num2 == 6)
    triple_bond2 = jnp.logical_and(atomic_num1 == 6, atomic_num2 == 8)
    triple_bond = jnp.logical_or(triple_bond1, triple_bond2)

    [cov_pot,
     bo,
     bopi,
     bopi2,
     abo] = reaxff_energy.calculate_covbon_pot(close_nbr_inds,
                                              close_nbr_dists,
                                              close_nbr_inds != N,
                                              self.species[i],
                                              triple_bond,
                                              self.ffields[i])

    [lone_pot, vlp] = reaxff_energy.calculate_lonpar_pot(self.species[i],
                                           atom_mask,
                                           abo,
                                           self.ffields[i])
    TEST_DATA[i]["preds"]["E_lone_pair"] = lone_pot
    self.assertAllClose(lone_pot, TEST_DATA[i]["results"]["E_lone_pair"],
                        atol=ATOL*N,rtol=RTOL)

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_3_body(self, i, name):
    N = len(self.species[i])
    atom_mask = jnp.ones(N, dtype=jnp.bool_)
    atom_inds = jnp.arange(N).reshape(-1,1)

    close_nbr_inds = self.nbr_lists[i].close_nbrs.idx[atom_inds,
                                                  self.nbr_lists[i].filter2.idx]
    close_nbr_inds = jnp.where(self.nbr_lists[i].filter2.idx != -1,
                                   close_nbr_inds,
                                   N)

    #close_nbr_inds = self.nbr_lists[i].close_nbrs.idx
    close_nbr_dists = self.angles_and_dists[i][0]
    atomic_num1 = self.atomic_nums[i].reshape(-1, 1)
    atomic_num2 = self.atomic_nums[i][close_nbr_inds]
    # O: 8, C:6
    triple_bond1 = jnp.logical_and(atomic_num1 == 8, atomic_num2 == 6)
    triple_bond2 = jnp.logical_and(atomic_num1 == 6, atomic_num2 == 8)
    triple_bond = jnp.logical_or(triple_bond1, triple_bond2)

    [cov_pot,
     bo,
     bopi,
     bopi2,
     abo] = reaxff_energy.calculate_covbon_pot(close_nbr_inds,
                                              close_nbr_dists,
                                              close_nbr_inds != N,
                                              self.species[i],
                                              triple_bond,
                                              self.ffields[i])

    [lone_pot, vlp] = reaxff_energy.calculate_lonpar_pot(self.species[i],
                                           atom_mask,
                                           abo,
                                           self.ffields[i])
    body_3_inds = self.nbr_lists[i].filter3.idx
    body_3_angles = self.angles_and_dists[i][2]
    [val_pot,
     total_penalty,
     total_conj] = reaxff_energy.calculate_valency_pot(self.species[i],
                                         body_3_inds,
                                         body_3_angles,
                                         body_3_inds[:,1] != body_3_inds[:,2],
                                         close_nbr_inds,
                                         vlp,
                                         bo,bopi, bopi2, abo,
                                         self.ffields[i])
    TEST_DATA[i]["preds"]["E_val"] = val_pot+total_penalty
    TEST_DATA[i]["preds"]["E_coa"] = total_conj
    self.assertAllClose(val_pot+total_penalty, TEST_DATA[i]["results"]["E_val"],
                        atol=ATOL*N,rtol=RTOL)

    self.assertAllClose(total_conj, TEST_DATA[i]["results"]["E_coa"],
                        atol=ATOL*N,rtol=RTOL)

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_4_body(self, i, name):
    N = len(self.geos[i])
    atom_inds = jnp.arange(N).reshape(-1,1)

    close_nbr_inds = self.nbr_lists[i].close_nbrs.idx[atom_inds,
                                                  self.nbr_lists[i].filter2.idx]
    close_nbr_inds = jnp.where(self.nbr_lists[i].filter2.idx != -1,
                                   close_nbr_inds,
                                   N)

    #close_nbr_inds = self.nbr_lists[i].close_nbrs.idx
    close_nbr_dists = self.angles_and_dists[i][0]
    atomic_num1 = self.atomic_nums[i].reshape(-1, 1)
    atomic_num2 = self.atomic_nums[i][close_nbr_inds]
    # O: 8, C:6
    triple_bond1 = jnp.logical_and(atomic_num1 == 8, atomic_num2 == 6)
    triple_bond2 = jnp.logical_and(atomic_num1 == 6, atomic_num2 == 8)
    triple_bond = jnp.logical_or(triple_bond1, triple_bond2)

    [cov_pot,
     bo,
     bopi,
     bopi2,
     abo] = reaxff_energy.calculate_covbon_pot(close_nbr_inds,
                                              close_nbr_dists,
                                              close_nbr_inds != N,
                                              self.species[i],
                                              triple_bond,
                                              self.ffields[i])

    body_4_inds = self.nbr_lists[i].filter4.idx
    body_4_angles =self.angles_and_dists[i][3]
    [torsion_pot,
     tor_conj] = reaxff_energy.calculate_torsion_pot(self.species[i],
                                       body_4_inds,
                                       body_4_angles,
                                       body_4_inds[:,1] != body_4_inds[:,2],
                                       close_nbr_inds,
                                       bo,bopi,abo,
                                       self.ffields[i])
    TEST_DATA[i]["preds"]["E_tors"] = torsion_pot
    TEST_DATA[i]["preds"]["E_conj"] = tor_conj
    self.assertAllClose(torsion_pot, TEST_DATA[i]["results"]["E_tors"],
                        atol=ATOL*N,rtol=RTOL)

    self.assertAllClose(tor_conj, TEST_DATA[i]["results"]["E_conj"],
                        atol=ATOL*N,rtol=RTOL)

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_hbond(self, i, name):
    N = len(self.species[i])
    atom_inds = jnp.arange(N).reshape(-1,1)

    close_nbr_inds = self.nbr_lists[i].close_nbrs.idx[atom_inds,
                                                  self.nbr_lists[i].filter2.idx]
    close_nbr_inds = jnp.where(self.nbr_lists[i].filter2.idx != -1,
                                   close_nbr_inds,
                                   N)

    #close_nbr_inds = self.nbr_lists[i].close_nbrs.idx
    close_nbr_dists = self.angles_and_dists[i][0]
    atomic_num1 = self.atomic_nums[i].reshape(-1, 1)
    atomic_num2 = self.atomic_nums[i][close_nbr_inds]
    # O: 8, C:6
    triple_bond1 = jnp.logical_and(atomic_num1 == 8, atomic_num2 == 6)
    triple_bond2 = jnp.logical_and(atomic_num1 == 6, atomic_num2 == 8)
    triple_bond = jnp.logical_or(triple_bond1, triple_bond2)

    [cov_pot,
     bo,
     bopi,
     bopi2,
     abo] = reaxff_energy.calculate_covbon_pot(close_nbr_inds,
                                              close_nbr_dists,
                                              close_nbr_inds != N,
                                              self.species[i],
                                              triple_bond,
                                              self.ffields[i])
    if self.nbr_lists[i].filter_hb != None:
      hb_inds = self.nbr_lists[i].filter_hb.idx
    else:
      hb_inds = None
    hb_pot = 0.0
    if hb_inds != None:
      hb_ang_dist = self.angles_and_dists[i][4]
      hb_mask = (hb_inds[:,1] != -1) & (hb_inds[:,2] != -1)
      far_nbr_inds = self.nbr_lists[i].far_nbrs.idx

      hb_pot = reaxff_energy.calculate_hb_pot(self.species[i],
                               hb_inds,
                               hb_ang_dist,
                               hb_mask,
                               close_nbr_inds,
                               far_nbr_inds,
                               bo,
                               self.ffields[i])
    TEST_DATA[i]["preds"]["E_hb"] = hb_pot
    self.assertAllClose(hb_pot, TEST_DATA[i]["results"]["E_hb"],
                        atol=ATOL*N,rtol=RTOL)

  @parameterized.parameters(
       [(i, TEST_DATA[i]['name'])  for i in range(len(TEST_DATA))])
  def test_overcoord(self, i, name):
    N = len(self.species[i])
    atom_mask = jnp.ones(N, dtype=jnp.bool_)
    atom_inds = jnp.arange(N).reshape(-1,1)

    close_nbr_inds = self.nbr_lists[i].close_nbrs.idx[atom_inds,
                                                  self.nbr_lists[i].filter2.idx]
    close_nbr_inds = jnp.where(self.nbr_lists[i].filter2.idx != -1,
                                   close_nbr_inds,
                                   N)

    #close_nbr_inds = self.nbr_lists[i].close_nbrs.idx
    close_nbr_dists = self.angles_and_dists[i][0]
    atomic_num1 = self.atomic_nums[i].reshape(-1, 1)
    atomic_num2 = self.atomic_nums[i][close_nbr_inds]
    # O: 8, C:6
    triple_bond1 = jnp.logical_and(atomic_num1 == 8, atomic_num2 == 6)
    triple_bond2 = jnp.logical_and(atomic_num1 == 6, atomic_num2 == 8)
    triple_bond = jnp.logical_or(triple_bond1, triple_bond2)

    [cov_pot,
     bo,
     bopi,
     bopi2,
     abo] = reaxff_energy.calculate_covbon_pot(close_nbr_inds,
                                              close_nbr_dists,
                                              close_nbr_inds != N,
                                              self.species[i],
                                              triple_bond,
                                              self.ffields[i])

    [lone_pot, vlp] = reaxff_energy.calculate_lonpar_pot(self.species[i],
                                           atom_mask,
                                           abo,
                                           self.ffields[i])

    overunder_pot = reaxff_energy.calculate_ovcor_pot(self.species[i],
                                        self.atomic_nums[i],
                                        atom_mask,
                                        close_nbr_inds,
                                        close_nbr_dists,
                                        close_nbr_inds != N,
                                        bo, bopi, bopi2, abo, vlp,
                                        self.ffields[i])
    TEST_DATA[i]["preds"]["E_atom"] = overunder_pot
    self.assertAllClose(overunder_pot, TEST_DATA[i]["results"]["E_atom"],
                        atol=ATOL*N,rtol=RTOL)




