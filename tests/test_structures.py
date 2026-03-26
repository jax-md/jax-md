"""Shared test structure definitions for UMA comparison tests."""

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule


def make_structures():
  """Build 60+ diverse structures for comparison testing."""
  structures = {}
  rng = np.random.default_rng(2024)

  # FCC metals
  for el, a in [('Al',4.05),('Cu',3.615),('Ag',4.085),('Au',4.078),
                ('Ni',3.524),('Pt',3.924),('Pd',3.890),('Rh',3.803),
                ('Ir',3.839),('Pb',4.951)]:
    atoms = bulk(el, 'fcc', a=a, cubic=True)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{el}_fcc'] = (atoms, 'omat')

  # BCC metals
  for el, a in [('Fe',2.87),('W',3.165),('Mo',3.147),('Cr',2.91),
                ('V',3.03),('Nb',3.30),('Ta',3.30)]:
    atoms = bulk(el, 'bcc', a=a, cubic=True)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{el}_bcc'] = (atoms, 'omat')

  # Diamond structures
  for el, a in [('Si',5.43),('Ge',5.66),('C',3.567)]:
    atoms = bulk(el, 'diamond', a=a)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{el}_diamond'] = (atoms, 'omat')

  # Rocksalt
  def rocksalt(el1, el2, a):
    return Atoms(f'{el1}{el2}'*4,
      scaled_positions=[(0,0,0),(0.5,0.5,0.5),(0.5,0,0),(0,0.5,0.5),
                        (0,0.5,0),(0.5,0,0.5),(0,0,0.5),(0.5,0.5,0)],
      cell=[a,a,a], pbc=True)

  for el1, el2, a in [('Na','Cl',5.64),('K','Cl',6.29),('Li','F',4.03),
                       ('Mg','O',4.21),('Ca','O',4.81),('Ba','O',5.52),
                       ('Na','F',4.62),('K','Br',6.60)]:
    atoms = rocksalt(el1, el2, a)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{el1}{el2}_rs'] = (atoms, 'omat')

  # Perovskites
  def perovskite(A, B, a):
    return Atoms(f'{A}{B}OOO',
      scaled_positions=[(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
      cell=[a,a,a], pbc=True)

  for A, B, a in [('Sr','Ti',3.905),('Ba','Ti',4.01),('Ca','Ti',3.84)]:
    atoms = perovskite(A, B, a)
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    structures[f'{A}{B}O3'] = (atoms, 'omat')

  # Supercells
  for el, a in [('Cu',3.615),('Fe',2.87),('Al',4.05)]:
    cryst = 'fcc' if el in ('Cu','Al') else 'bcc'
    atoms = bulk(el, cryst, a=a, cubic=True).repeat((2,2,2))
    atoms.positions += rng.normal(scale=0.015, size=atoms.positions.shape)
    structures[f'{el}_2x2x2'] = (atoms, 'omat')

  # Molecules
  for mol in ['H2O','NH3','CH4','CO2','H2','N2','O2','C2H6','C2H4','C2H2',
              'CH3OH','H2CO','HCN','HF','H2S','PH3','SiH4','NF3','CF4','SF6']:
    try:
      atoms = molecule(mol)
      atoms.center(vacuum=10.0)
      atoms.pbc = False
      atoms.info['charge'] = 0
      atoms.info['spin'] = 1
      atoms.positions += rng.normal(scale=0.03, size=atoms.positions.shape)
      structures[f'{mol}_mol'] = (atoms, 'omol')
    except Exception:
      pass

  # Strained metals
  for el, a0 in [('Cu',3.615),('Al',4.05),('Pt',3.924)]:
    for strain in [-0.03, 0.03]:
      a = a0 * (1 + strain)
      tag = 'comp' if strain < 0 else 'tens'
      atoms = bulk(el, 'fcc', a=a, cubic=True)
      structures[f'{el}_{tag}'] = (atoms, 'omat')

  # Spin states
  o2 = molecule('O2')
  o2.center(vacuum=10.0)
  o2.pbc = False
  o2.info['charge'] = 0
  o2.info['spin'] = 3
  structures['O2_triplet'] = (o2.copy(), 'omol')
  o2.info['spin'] = 1
  structures['O2_singlet'] = (o2.copy(), 'omol')

  # Charged
  oh = Atoms('OH', positions=[[0,0,0],[0,0,0.97]])
  oh.center(vacuum=10.0)
  oh.pbc = False
  oh.info['charge'] = -1
  oh.info['spin'] = 1
  structures['OH_anion'] = (oh, 'omol')

  return structures
