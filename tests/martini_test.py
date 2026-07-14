"""
martini_test.py
=================
Reference test suite comparing hardcoded GROMACS reference values and JaxMD
energies and forces.

Layout expected for each test system
-------------------------------------
<system_name>/
  martini_md.mdp      # mdp file; epsilon_r is read from here automatically
  conf.gro          # starting coordinates
  system.top        # topology (parsed by your GromacsTopFile equivalent)
  defines.txt         # optional preprocessor defines (one per line)

Usage
-----
  python martini_test.py

Note: Helper class GromacsGroFile is used for loading Gromacs GRO files and is
adapted from OpenMM grofile.py (MIT License)

Portions copyright (c) 2012-2016 Stanford University and the Authors.
Authors: Lee-Ping Wang, Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import tarfile
import time
from pathlib import Path
from typing import NamedTuple

import jax

from absl.testing import absltest

from jax_md.mm_forcefields.martini.topology import create_topology
from jax_md.mm_forcefields.martini.energy import build_energies, energy_fn
from jax_md.mm_forcefields.martini.top_file_parser import GromacsTopFile

import jax.numpy as jnp
from jax_md import quantity
from jax_md.mm_forcefields import neighbor
from jax_md.space import periodic
from jax_md.test_util import JAXMDTestCase
from jax_md.util import Array

jax.config.update('jax_enable_x64', True)

from re import match
import numpy as np

_DATA_ROOT = Path(__file__).parent / 'data'
_MARTINI_DATA_DIR = _DATA_ROOT / 'martini_test'
_MARTINI_DATA_TARBALL = _DATA_ROOT / 'martini_data.tar.gz'


def _ensure_martini_data_extracted() -> None:
  if _MARTINI_DATA_DIR.exists():
    return
  if not _MARTINI_DATA_TARBALL.exists():
    raise FileNotFoundError(
      f'Missing martini test data and tarball: {_MARTINI_DATA_TARBALL}'
    )
  with tarfile.open(_MARTINI_DATA_TARBALL, 'r:gz') as tf:
    tf.extractall(_DATA_ROOT)


_ensure_martini_data_extracted()


def _isint(word):
  """ONLY matches integers! If you have a decimal point? None shall pass!

  @param[in] word String (for instance, '123', '153.0', '2.', '-354')
  @return answer Boolean which specifies whether the string is an integer (only +/- sign followed by digits)

  """
  return match('^[-+]?[0-9]+$', word)


def _isfloat(word):
  """Matches ANY number; it can be a decimal, scientific notation, what have you
  CAUTION - this will also match an integer.

  @param[in] word String (for instance, '123', '153.0', '2.', '-354')
  @return answer Boolean which specifies whether the string is any number

  """
  return match(r'^[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?$', word)


def _is_gro_coord(line):
  """Determines whether a line contains GROMACS data or not

  @param[in] line The line to be tested

  """
  # data lines are fixed field
  fields = []
  fields.append(line[16:20].strip())  # atom number
  fields.append(line[21:28].strip())  # x coord
  fields.append(line[29:36].strip())  # y coord
  fields.append(line[37:44].strip())  # z coord
  if all([f != '' for f in fields]):  # check for empty fields
    return all(
      [
        _isint(fields[0]),
        _isfloat(fields[1]),
        _isfloat(fields[2]),
        _isfloat(fields[3]),
      ]
    )
  else:
    return 0


def _is_gro_box(line):
  """Determines whether a line contains a GROMACS box vector or not

  @param[in] line The line to be tested

  """
  sline = line.split()
  if len(sline) == 9 and all([_isfloat(i) for i in sline]):
    return 1
  elif len(sline) == 3 and all([_isfloat(i) for i in sline]):
    return 1
  else:
    return 0


class GromacsGroFile:
  """GromacsGroFile parses a Gromacs .gro file and constructs a set of atom positions from it.

  A .gro file also contains some topological information, such as elements and residue names,
  but not enough to construct a full Topology object.  This information is recorded and stored
  in the object's public fields."""

  def __init__(self, file):
    """Load a .gro file.

    The atom positions can be retrieved by calling getPositions().

    Parameters
    ----------
    file : string
        the name of the file to load
    """
    xyzs = []
    xyz = []
    box = []
    ln = 0
    frame = 0
    with open(file) as grofile:
      for line in grofile:
        if ln == 0:
          pass
        elif ln == 1:
          na = int(line.strip())
        elif _is_gro_coord(line):
          firstDecimalPos = line.index('.', 20)
          secondDecimalPos = line.index('.', firstDecimalPos + 1)
          digits = secondDecimalPos - firstDecimalPos
          pos = [
            float(line[20 + i * digits : 20 + (i + 1) * digits])
            for i in range(3)
          ]
          xyz.append([pos[0], pos[1], pos[2]])
        elif _is_gro_box(line) and ln == na + 2:
          box = [float(i) for i in line.split()]
          xyzs.append(xyz)
          xyz = []
          ln = -1
          frame += 1
        else:
          raise Exception('Unexpected line in .gro file: ' + line)
        ln += 1

    ## The atom positions read from the file.  If the file contains multiple frames, these are the positions in the first frame.
    self._positions = jnp.array(xyzs)
    self.box = np.array(box)

  def getNumFrames(self):
    """Get the number of frames stored in the file."""
    return len(self._positions)

  def getPositions(self, frame=0):
    """Get the atomic positions.

    Parameters
    ----------
    frame : int=0
        the index of the frame for which to get positions

    Returns
    -------
    positions : jnp.ndarray of shape (num_atoms, 3)
    """
    return self._positions[frame]


ENERGY_ATOL = 1e-4  # kJ/mol  – absolute tolerance for total energy
ENERGY_RTOL = 1e-5  # relative tolerance for total energy
FORCE_ATOL = 1e-4  # kJ/(mol·nm) – absolute tolerance per force component
FORCE_RTOL = 1e-5  # relative tolerance per force component

# Tolerance for per-term energy comparison (looser – terms can be large & opposite)
TERM_ATOL = 1e-2
TERM_RTOL = 1e-4

# Absolute deviation allowed when checking constraint / vsite positions (nm)
CONSTRAINT_POS_TOL = 2.5e-3


class SystemSpec(NamedTuple):
  name: str
  test_dir: str  # relative to this file


SYSTEMS: list[SystemSpec] = [
  SystemSpec('2f4k', 'data/martini_test/2f4k'),
  SystemSpec('1ubq', 'data/martini_test/1ubq'),
]

VSITE_SYSTEMS: list[SystemSpec] = [
  SystemSpec('vsites', 'data/martini_test/vsites'),
]

# Each entry is keyed by SystemSpec.name.
# Fields:
#   energy_total  – float, kJ/mol (the "Potential" column from gmx energy)
#   energy_terms  – dict[str, float], per-term energies (kJ/mol)
#   forces    – list of [fx, fy, fz] per atom (kJ mol⁻¹ nm⁻¹);
#           set to [] to skip force comparison for that system
GMX_REFERENCE: dict[str, dict] = {
  '1ubq': {
    'energy_total': 914.479709,
    'energy_terms': {
      'Bond': 453.132328,
      'G96Angle': 102.051221,
      'Restr. Angles': 1110.932141,
      'Proper Dih.': 33.757235,
      'Improper Dih.': 0.001362,
      'LJ (SR)': -584.821466,
      'Coulomb (SR)': -200.573112,
      'Potential': 914.479709,
      'RB-Fourier': 0.0,
      'Periodic Dih.': 33.757235,
    },
    'forces': [
      [-3.80860e03, 2.66059e03, -3.37056e03],
      [1.52163e02, -1.81838e02, 9.24090e00],
      [-1.29075e01, 3.55807e02, 2.90039e02],
      [1.83806e01, -5.51677e01, -3.54594e00],
      [-2.92907e03, 3.93281e02, 1.08519e03],
      [-2.67910e00, -1.85394e01, 6.86044e00],
      [5.18251e02, -1.35466e02, -4.15083e02],
      [-1.62183e02, 3.67218e02, -2.11549e01],
      [1.86803e01, -5.09774e01, 9.04545e00],
      [4.92593e01, -4.83303e01, 3.85476e00],
      [-1.59160e03, -7.89430e02, 1.87446e03],
      [3.50304e01, 4.41919e00, 2.86639e01],
      [-3.56273e02, 9.43457e01, 1.30670e02],
      [1.19397e02, -1.70509e02, -7.42449e01],
      [-1.06095e02, 7.57359e01, 3.19871e01],
      [5.76862e02, -1.59194e02, 2.06037e02],
      [4.36246e02, -6.40006e02, 9.18480e02],
      [-1.16360e04, -5.75982e03, -8.97363e03],
      [5.73504e03, 3.01323e03, 4.38503e03],
      [5.22437e03, 2.48547e03, 5.11100e03],
      [2.90744e02, 9.79370e01, -2.81261e02],
      [-4.17819e01, 7.95996e01, -1.49719e01],
      [-1.20331e02, 6.91300e02, -1.78559e03],
      [-8.08290e01, 2.03935e02, 3.56387e02],
      [-5.45405e01, 2.09784e02, -4.06012e02],
      [-1.21984e02, -1.32274e02, -1.95832e01],
      [-7.01081e01, -4.17897e01, -3.31828e00],
      [1.52103e03, 5.21377e02, -2.03011e03],
      [-1.41112e03, -6.85410e02, -2.69606e02],
      [-1.63441e02, -9.59935e01, -1.63494e01],
      [-3.23890e01, -7.28370e01, -6.57567e01],
      [3.10700e03, -5.03652e02, -9.34601e02],
      [4.61428e01, -5.14148e01, 1.00069e02],
      [-7.12194e01, 6.04672e02, -5.33637e02],
      [-7.49334e01, -5.67544e01, 2.37882e02],
      [3.78881e03, -2.46097e03, 3.43267e03],
      [-1.98651e02, -7.93580e01, 3.58149e02],
      [-2.07484e03, 8.29595e02, -1.52305e03],
      [-4.22625e01, -4.91234e01, -1.35312e01],
      [6.47189e01, 8.23393e01, 3.99476e02],
      [-8.27316e02, -1.68617e02, -4.23465e02],
      [-1.76217e02, -6.15636e02, 6.58330e02],
      [1.31188e02, -1.56094e01, -3.83866e02],
      [1.57127e02, 8.29784e02, -2.56440e02],
      [2.69028e03, -4.12849e02, 1.32037e03],
      [-1.73295e02, -4.67378e02, 1.36200e02],
      [-7.49476e00, 3.41354e01, 4.11689e01],
      [-3.73251e02, 1.63957e02, -9.02320e02],
      [1.50994e02, 2.07262e02, -7.07877e01],
      [-6.62304e01, 1.44278e02, 3.21539e02],
      [-1.40115e02, 3.23364e01, -4.92824e01],
      [4.44071e02, 1.15975e01, -3.07245e02],
      [7.76978e00, -1.96749e02, -1.77233e02],
      [-1.00646e02, 2.03144e01, 1.75646e02],
      [-3.71994e00, 2.16586e01, 2.77879e01],
      [-2.25428e02, 6.93387e01, -9.31255e01],
      [1.62022e02, -2.99578e02, -1.62052e02],
      [-2.47667e02, 3.64019e01, 2.89748e01],
      [2.87887e02, -4.78344e01, 3.44590e02],
      [7.31127e00, 7.01237e00, -9.56583e01],
      [-1.32728e02, -4.34638e02, -3.40886e02],
      [5.85220e01, 2.78888e02, 3.03654e02],
      [9.02777e01, -4.31531e02, -2.14983e01],
      [-4.48255e02, 1.15741e02, -2.05403e02],
      [1.71329e01, 4.13794e01, 5.61423e00],
      [-1.56629e02, -1.21794e03, -5.18394e02],
      [7.67156e01, -1.70618e02, -1.04159e02],
      [5.45424e02, 1.06357e03, 4.90045e02],
      [5.77608e00, -3.30926e02, -1.48074e02],
      [-1.21279e02, -2.52608e02, -5.49497e02],
      [5.04736e01, -1.34229e02, 3.10067e02],
      [-1.26168e02, 3.93444e01, -2.74848e02],
      [9.94207e02, 1.27799e01, -5.93670e02],
      [1.02094e03, 7.16047e02, 8.73177e02],
      [-4.30944e01, 4.43309e02, 3.12763e02],
      [-7.49902e02, 5.75664e02, 1.71036e02],
      [-1.75050e00, -4.03963e01, 5.51807e00],
      [-6.29929e02, -8.39652e01, -2.93257e02],
      [1.27115e03, -1.16924e02, 4.86043e02],
      [3.41802e01, 3.54716e01, 5.65488e02],
      [2.09374e02, -5.29078e02, -2.80242e02],
      [-3.87236e02, 1.66811e01, -3.48465e02],
      [4.56443e02, 8.05336e01, 3.25724e02],
      [5.24559e02, -1.73455e03, -1.39848e03],
      [-4.26423e02, 2.51809e02, 2.49032e02],
      [-1.24306e02, 5.19246e02, -4.53877e02],
      [-2.56148e02, 7.39642e02, 1.64823e02],
      [-2.40004e02, -3.06892e03, -2.82295e02],
      [2.22591e02, 1.12187e02, -8.50753e01],
      [-9.87306e01, 1.62778e01, 1.19122e01],
      [7.75590e01, 2.63750e02, -8.46506e01],
      [2.45543e01, -2.07768e01, -9.51526e01],
      [-5.00179e03, -8.20296e03, 4.16623e03],
      [-1.17994e02, 9.31301e01, 3.16317e01],
      [-5.86275e02, 2.36413e03, -6.97711e03],
      [8.58765e01, -1.05501e02, -3.03272e02],
      [-1.00771e00, 1.56608e01, 4.65844e00],
      [2.59748e00, -2.32623e01, 9.09357e00],
      [3.33465e02, 1.83291e03, -8.12839e02],
      [-2.71177e02, -9.81563e02, 5.97588e02],
      [-1.56712e02, -4.91827e02, 8.22202e02],
      [1.02692e03, -2.81359e03, 6.07133e03],
      [4.19783e02, 8.55719e01, 2.12334e02],
      [-2.83742e02, -1.27979e02, -2.42336e02],
      [-2.11510e01, 1.05571e02, 4.79137e02],
      [-1.05296e02, -4.68873e01, 3.30075e01],
      [1.58338e02, -2.86868e02, 1.45200e01],
      [-7.00884e01, -8.28708e00, -1.35316e02],
      [-7.39161e01, -1.77916e02, 3.64210e02],
      [-6.88980e00, 3.93980e01, -1.18314e02],
      [-2.00413e01, 4.58270e01, 6.42652e02],
      [2.44664e02, -3.00811e02, 9.95735e01],
      [8.92855e02, -1.27884e03, -1.20696e03],
      [-1.66719e03, 2.77037e03, 1.90402e03],
      [1.38193e02, 2.96571e02, 1.79094e02],
      [-8.82211e01, -2.09726e02, 1.07395e02],
      [1.72982e03, -9.48045e02, -1.69628e03],
      [1.62789e02, 2.20668e01, -1.48360e02],
      [2.77319e02, 3.90113e02, 5.29751e02],
      [2.40704e01, 1.48559e01, -2.19567e02],
      [2.05485e02, -4.00103e02, -4.76471e02],
      [1.89353e02, -2.36915e02, -1.19807e02],
      [-6.35979e02, 4.74414e01, -5.05601e02],
      [-1.09809e03, -9.66396e02, 6.53516e02],
      [-4.87169e02, 1.15006e02, 7.65301e02],
      [3.17969e02, 3.48819e02, 1.22192e02],
      [2.55167e01, -5.71480e01, -3.53485e01],
      [-2.42095e01, 9.51820e00, 3.40342e00],
      [-4.06164e01, -5.60670e00, -1.67872e01],
      [2.27968e02, 2.37387e02, 1.00248e02],
      [-2.12271e02, -9.32497e01, -6.33531e01],
      [2.88501e03, -3.90201e02, -9.86039e02],
      [-2.12094e01, 3.96547e01, 7.93358e00],
      [-6.23551e03, -5.71130e02, 2.16653e02],
      [2.11327e02, 9.57349e01, 6.76821e01],
      [2.96975e03, -1.01201e03, 6.79791e01],
      [2.04498e02, -4.31769e02, 1.37503e02],
      [-2.03072e02, 9.31624e00, -1.60792e02],
      [-1.76242e04, -4.85398e02, 3.37164e03],
      [7.61499e03, 1.18461e03, -1.23905e03],
      [1.02973e04, -4.65518e02, -2.35521e03],
      [2.13120e02, 1.66560e03, 1.24787e03],
      [-3.86474e02, 3.83591e02, 6.58272e00],
      [-1.15167e02, 1.30470e02, 5.23084e01],
      [2.23750e01, 1.63289e01, 1.58551e02],
      [5.90210e01, -1.49047e02, 8.69221e00],
      [5.41157e03, 7.80042e03, -4.19532e03],
      [-3.84043e02, 3.21914e02, 2.07287e02],
      [3.65435e01, -4.38749e01, -4.37370e01],
      [5.27547e01, 7.20328e00, -7.17160e01],
      [-3.40612e02, 5.86663e01, 2.02439e02],
      [3.03676e02, -1.74899e02, -1.92802e02],
      [2.11617e02, 2.93912e03, 3.45982e02],
      [-1.63775e01, -2.01005e01, 9.56115e-01],
      [-5.94392e01, -1.45956e02, 7.99098e01],
      [1.10107e02, -1.75539e01, -1.10966e02],
      [-5.35675e02, 1.74836e03, 1.82261e03],
      [-5.24526e01, 6.32803e00, 1.99591e02],
      [-8.18594e01, -2.70655e02, -1.29950e02],
      [1.15664e02, -9.87929e-01, 7.88652e01],
      [3.40561e01, 5.44145e01, -2.87663e01],
      [-3.26330e02, -1.26160e02, -9.46377e01],
      [-3.17279e02, 1.95504e02, 1.63042e02],
      [2.60530e02, -1.98270e02, -1.47582e02],
      [3.13620e02, 4.54404e02, 1.04873e01],
      [-9.38582e01, -3.19797e02, -8.30674e01],
    ],
  },
  '2f4k': {
    'energy_total': -65.361552,
    'energy_terms': {
      'Bond': 260.576687,
      'G96Angle': 69.911357,
      'Restr. Angles': 350.899762,
      'Proper Dih.': 28.241199,
      'Improper Dih.': 2e-05,
      'LJ (SR)': -686.646802,
      'Coulomb (SR)': -88.343774,
      'Potential': -65.361552,
      'RB-Fourier': 0.0,
      'Periodic Dih.': 28.241199,
    },
    'forces': [
      [4.16138e01, -5.75608e01, -1.34696e02],
      [2.29577e01, 1.00374e02, 2.11069e02],
      [-7.93615e02, 2.29984e02, -2.63164e02],
      [7.97920e00, -2.92122e02, -4.18687e02],
      [-9.25577e01, 4.78145e02, 3.35256e00],
      [1.40815e01, -1.65177e02, -1.54107e01],
      [-9.39251e02, -3.82502e01, -4.57368e02],
      [5.64459e02, -3.38938e02, 1.49794e02],
      [1.06420e03, -4.21789e02, 4.62808e02],
      [3.29277e02, 3.25375e02, 1.12294e02],
      [2.77151e02, 2.90740e02, 2.45526e02],
      [-3.09943e02, 8.09493e01, -1.26875e02],
      [2.94649e01, 1.86873e01, 1.40563e01],
      [5.18348e00, -1.47242e01, 2.60506e01],
      [5.15171e02, 2.55811e02, -6.28155e02],
      [1.35074e02, 2.93134e02, 4.37976e01],
      [-1.07842e02, -2.42984e02, 2.15711e01],
      [-1.72356e01, -5.18633e02, 5.76971e02],
      [2.00835e01, 5.99201e01, -1.39935e01],
      [-1.27622e02, 2.55644e02, -4.15758e02],
      [-8.44559e01, -1.39551e01, -7.72231e00],
      [8.23811e02, 9.45944e00, 6.22847e02],
      [-3.69288e02, 1.71002e02, -2.14348e02],
      [2.00704e01, 1.48134e01, 3.74037e01],
      [3.42274e01, -3.52214e01, -7.36518e00],
      [-1.75719e03, 8.77623e02, 1.87074e03],
      [2.11671e03, -1.83957e03, -3.64041e03],
      [-7.39033e01, 1.09383e02, 9.33974e01],
      [-1.22275e03, 7.79598e02, 1.81231e03],
      [5.46094e01, -2.59736e01, 2.01390e01],
      [1.50190e02, 9.45794e00, 3.03555e01],
      [-2.01805e02, -1.78624e02, 2.42434e02],
      [1.45572e02, 5.21500e01, -1.19135e02],
      [-3.27456e02, 9.78087e01, -2.52502e02],
      [-7.62567e-01, -2.48867e02, -5.91800e01],
      [1.75703e01, -2.33474e01, 2.19362e02],
      [-6.47043e01, 4.77888e01, 1.30867e02],
      [3.97983e01, 7.98140e01, -1.49896e02],
      [3.78747e02, 8.61968e01, -2.26323e02],
      [1.33938e01, -8.48586e00, 2.69743e01],
      [8.83509e01, -4.28280e01, -2.47538e01],
      [3.98042e02, 2.24275e02, 1.64563e02],
      [-1.51516e02, -1.36859e02, 4.30119e01],
      [-4.43316e02, 3.70676e02, -2.41425e02],
      [-9.10900e01, -3.09524e02, 9.22657e01],
      [1.41899e00, -4.52745e02, 1.10160e01],
      [1.09075e02, -1.15322e01, 7.84206e01],
      [9.63238e-01, 2.45916e02, -7.86174e02],
      [-1.78114e02, 7.69635e01, 1.04462e03],
      [-3.77333e02, 5.70978e02, 1.16419e03],
      [-2.20513e02, 2.44860e02, -3.07828e02],
      [1.25348e02, 7.15373e01, -6.73005e02],
      [-2.74242e01, -4.55071e01, 2.87660e02],
      [2.19585e01, -1.95863e01, -4.97021e01],
      [0.00000e00, 0.00000e00, 0.00000e00],
      [-3.98129e01, 9.77539e-01, 4.21640e01],
      [5.74845e00, -1.07847e01, -2.82065e01],
      [-2.34620e02, -5.13521e02, 4.26297e02],
      [-4.21844e01, -2.52373e02, 2.39519e02],
      [9.18027e01, -2.07427e02, -1.37882e02],
      [3.92544e02, -3.55008e02, -7.31804e02],
      [-6.00326e02, -1.16768e03, -8.44576e02],
      [2.11937e02, 8.20989e02, 3.17787e02],
      [8.47174e02, 1.49084e03, -3.65489e02],
      [-2.18180e02, -1.65730e02, 5.48760e02],
      [2.66418e01, -5.62407e00, -4.14801e01],
      [2.31274e00, -7.10050e-01, -2.45536e01],
      [-2.27743e01, -4.53979e02, 3.94596e02],
      [-1.19272e02, -1.95793e02, -1.29031e02],
      [-4.30783e02, -6.99965e02, 8.45313e02],
      [-2.33290e02, -1.20497e01, 6.22650e01],
      [-1.93907e02, 9.08507e02, 1.44214e02],
      [3.24096e02, -2.64822e02, 6.06227e01],
      [-3.09712e02, 8.75728e01, -2.96428e01],
      [2.91485e02, 2.23624e02, -6.20271e02],
      [-1.51734e02, -1.46023e02, 2.71241e02],
      [6.34098e02, -1.84360e02, 3.02383e02],
      [-2.17177e02, -1.08238e02, -3.45757e02],
      [-5.45474e01, -6.24172e01, 6.70612e01],
      [3.50691e01, 2.46300e02, -4.92993e01],
      [2.13718e02, -1.85239e02, -8.50802e02],
      [-7.56137e01, -1.08527e02, -6.01016e01],
      [2.57499e02, 2.73428e02, -1.56725e02],
      [1.10193e01, -3.57541e00, 7.88578e01],
      [1.69990e01, 1.67350e01, -1.38665e01],
      [-5.06793e00, -1.13931e01, 2.41504e00],
    ],
  },
}

# Vsite reference positions keyed by SystemSpec.name.
# Each value is a list of [x, y, z] rows (nm) for every atom in the system.
# Only the vsite atom indices (see test_vsites) are compared.
VSITE_REFERENCE: dict[str, list[list[float]]] = {
  'vsites': [
    [0.0, 0.0, 0.0],
    [0.24, 0.0, 0.0],
    [0.12, 0.208, 0.0],
    [0.072, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.108, 0.062, 0.0],
    [0.087, 0.05, 0.0],
    [0.18, 0.104, 0.05],
    [0.12, 0.069, 0.0],
    [0.13, 0.087, 0.0],
  ],
}


def parse_mdp(mdp_path: Path) -> dict[str, str]:
  """Parse a GROMACS .mdp file into a key/value mapping.

  Args:
    mdp_path: Path to the .mdp file to parse.

  Returns:
    Dictionary containing the parsed parameter names and values.
  """
  params: dict[str, str] = {}
  for line in mdp_path.read_text().splitlines():
    line = line.split(';')[0].strip()
    if '=' not in line:
      continue
    key, _, value = line.partition('=')
    params[key.strip().lower().replace('-', '_')] = value.strip()
  return params


def read_epsilon_r(test_dir: Path) -> float:
  """Read the dielectric constant from the MARTINI .mdp file.

  Args:
    test_dir: Directory containing the test system files.

  Returns:
    The epsilon_r value from the .mdp file, or 15.0 when absent.
  """
  mdp_candidates = list(test_dir.glob('*.mdp'))
  if not mdp_candidates:
    return 15.0
  mdp_path = next(
    (p for p in mdp_candidates if p.name == 'martini_md.mdp'), mdp_candidates[0]
  )
  params = parse_mdp(mdp_path)
  return float(params.get('epsilon_r', 15.0))


TEST_DIR = Path(__file__).parent.resolve()


def run_jaxmd(
  test_dir: Path, epsilon_r: float
) -> tuple[Array, dict[str, float], Array]:
  """Run JAX-MD for a system and return the energy, terms, and forces.

  Args:
    test_dir: Directory containing the topology and coordinates for the system.
    epsilon_r: Relative dielectric constant used by the topology.

  Returns:
    A tuple of the total energy in kJ/mol, per-term energies, and forces.
  """
  topology_file = GromacsTopFile(test_dir / 'system.top', epsilon_r=epsilon_r)

  gro_file = GromacsGroFile(test_dir / 'conf.gro')
  positions = gro_file.getPositions(0)  # (N, 3) in nm
  displacement_fn, shift_fn = periodic(gro_file.box)

  n_fn = neighbor.create_neighbor_list(
    displacement_fn,
    gro_file.box,
    1.1,
  )

  martini_topology = create_topology(
    topology_file, nonbonded_cutoff=1.1, epsilon_r=epsilon_r
  )

  positions = martini_topology.apply_vsites(
    positions, displacement_fn, shift_fn
  )
  nlist = n_fn.allocate(positions)
  martini_energy_fn = energy_fn(
    martini_topology, displacement_fn, shift_fn, include_vsites=True
  )
  total_energy = martini_energy_fn(positions, box=gro_file.box, neighbor=nlist)

  force_fn = quantity.force(martini_energy_fn)
  forces = force_fn(positions, box=gro_file.box, neighbor=nlist)

  vsite_mask = martini_topology.masses > 0
  forces = jnp.where(vsite_mask[:, None], forces, 0.0)

  # Per-term energies (for diagnostic output)
  bonded_terms, lj_energy, coulomb = build_energies(martini_topology)
  energy_terms: dict[str, float] = {}
  for name in bonded_terms:
    val = bonded_terms[name](positions, displacement_fn)
    if val != 0:
      energy_terms[name] = val

  val_lj = lj_energy(positions, nlist, displacement_fn)
  if val_lj != 0:
    energy_terms['LJ (SR)'] = val_lj

  energy_terms['Coulomb (SR)'] = (
    coulomb.energy(
      positions * 10.0,
      martini_topology.charges,
      gro_file.box * 10.0,
      martini_topology.excl_mask,
      None,
      nlist,
      None,
    )
    * 4.184
  )

  print('JaxMD energy terms: ', energy_terms)
  return total_energy, energy_terms, jnp.array(forces)


def _check_forces(
  gmx_forces: Array,
  jmd_forces: Array,
  system_name: str,
) -> list[str]:
  """Compare force arrays and report mismatches.

  Args:
    gmx_forces: Reference forces from GROMACS.
    jmd_forces: Forces produced by JAX-MD.
    system_name: Name of the system being tested.

  Returns:
    A list of human-readable failure messages.
  """
  messages: list[str] = []

  if gmx_forces.shape != jmd_forces.shape:
    messages.append(
      f'Force array shape mismatch: '
      f'reference {gmx_forces.shape} vs JaxMD {jmd_forces.shape}'
    )
    return messages

  close = jnp.isclose(jmd_forces, gmx_forces, rtol=FORCE_RTOL, atol=FORCE_ATOL)
  if close.all():
    print(
      f'JaxMD forces match reference.\n  JMD: {jmd_forces}\n  Ref: {gmx_forces}'
    )
    return messages

  errors = ~close
  n_diff = int(errors.sum())
  total = gmx_forces.size

  excess = (
    jnp.abs(jmd_forces - gmx_forces)
    - FORCE_ATOL
    - FORCE_RTOL * jnp.abs(gmx_forces)
  )
  max_idx = jnp.unravel_index(jnp.argmax(excess), excess.shape)
  atom_idx = max_idx[0]

  g = gmx_forces[atom_idx]
  j = jmd_forces[atom_idx]
  d = jnp.abs(g - j)

  messages.append(
    f'[{system_name}] Forces differ at {n_diff}/{total} components.\n'
    f'  Worst atom index : {atom_idx}\n'
    f'  GROMACS   : {g[0]:+.6f}  {g[1]:+.6f}  {g[2]:+.6f}\n'
    f'  JaxMD   : {j[0]:+.6f}  {j[1]:+.6f}  {j[2]:+.6f}\n'
    f'  |Delta|   : {d[0]:.6f}  {d[1]:.6f}  {d[2]:.6f}'
  )
  return messages


def _check_energy(
  gmx_energy: float,
  jmd_energy: float,
  system_name: str,
) -> list[str]:
  """Compare total energies and report mismatches.

  Args:
    gmx_energy: Reference total energy from GROMACS.
    jmd_energy: Total energy produced by JAX-MD.
    system_name: Name of the system being tested.

  Returns:
    A list of human-readable failure messages.
  """
  messages: list[str] = []
  abs_err = abs(gmx_energy - jmd_energy)
  rel_err = abs_err / (abs(gmx_energy) + 1e-30)
  if rel_err > ENERGY_RTOL and abs_err > ENERGY_ATOL:
    messages.append(
      f'[{system_name}] Total energy mismatch.\n'
      f'  GROMACS.  : {gmx_energy:16.4f} kJ/mol\n'
      f'  JaxMD   : {jmd_energy:16.4f} kJ/mol\n'
      f'  |Delta|   : {abs_err:16.4f} kJ/mol  (relative: {rel_err:.2e})'
    )
  print(f'GROMACS energy   : {gmx_energy:.4f} kJ/mol')
  print(f'JaxMD energy   : {jmd_energy:.4f} kJ/mol')
  return messages


def _check_energy_terms(
  gmx_terms: dict[str, float],
  jmd_terms: dict[str, float],
  system_name: str,
) -> list[str]:
  """Compare per-term energies and report mismatches.

  Args:
    gmx_terms: Reference energy terms from GROMACS.
    jmd_terms: Energy terms produced by JAX-MD.
    system_name: Name of the system being tested.

  Returns:
    A list of human-readable failure messages.
  """
  messages: list[str] = []
  shared_keys = set(gmx_terms) & set(jmd_terms)

  for key in sorted(shared_keys):
    g_val = gmx_terms[key]
    j_val = jmd_terms[key]
    abs_err = abs(g_val - j_val)
    rel_err = abs_err / (abs(g_val) + 1e-30)
    if rel_err > TERM_RTOL and abs_err > TERM_ATOL:
      messages.append(
        f"[{system_name}] Energy term '{key}' mismatch.\n"
        f'  GROMACS.  : {g_val:16.4f}\n'
        f'  JaxMD   : {j_val:16.4f}\n'
        f'  |Delta|   : {abs_err:16.4f}  (relative: {rel_err:.2e})'
      )
    print(f"GROMACS   '{key}': {g_val:.8f} kJ/mol")
    print(f"JaxMD   '{key}': {j_val:.8f} kJ/mol")

  gmx_only = set(gmx_terms) - set(jmd_terms)
  jmd_only = set(jmd_terms) - set(gmx_terms)
  if gmx_only:
    print(
      f'[{system_name}] Energy terms only in GROMACS (not compared): '
      f'{sorted(gmx_only)}'
    )
  if jmd_only:
    print(
      f'[{system_name}] Energy terms only in JaxMD (not compared): {sorted(jmd_only)}'
    )
  return messages


class GromacsReferenceTest(JAXMDTestCase):
  """Compare JAX-MD energies and forces against hardcoded GROMACS references."""

  def test_gromacs_vs_jaxmd(self) -> None:
    """Run the registered systems and compare their energies and forces."""
    for spec in SYSTEMS:
      with self.subTest(system=spec.name):
        print(f'\n=== Testing system: {spec.name} ===')
        test_dir = TEST_DIR / spec.test_dir

        self.assertTrue(
          test_dir.is_dir(), f'Test directory not found: {test_dir}'
        )

        ref = GMX_REFERENCE.get(spec.name)
        self.assertIsNotNone(
          ref,
          f"No GROMACS reference data found for system '{spec.name}'.",
        )
        assert ref is not None

        gmx_energy: float = ref['energy_total']
        gmx_terms: dict[str, float] = ref['energy_terms']
        gmx_forces_list: list[list[float]] = ref['forces']

        epsilon_r = read_epsilon_r(test_dir)

        t0 = time.perf_counter()
        jmd_energy, jmd_terms, jmd_forces = run_jaxmd(test_dir, epsilon_r)
        jmd_time = time.perf_counter() - t0

        print(f'JAX-MD runtime for {spec.name}: {jmd_time:.3f}s')

        failures: list[str] = []
        failures += _check_energy(gmx_energy, jmd_energy.item(), spec.name)
        failures += _check_energy_terms(gmx_terms, jmd_terms, spec.name)

        if gmx_forces_list:
          gmx_forces = jnp.array(gmx_forces_list, dtype=float)
          failures += _check_forces(gmx_forces, jmd_forces, spec.name)
        else:
          print(
            f'[{spec.name}] No reference forces provided – skipping force comparison.'
          )

        if failures:
          self.fail('\n\n'.join(failures))

  def test_vsites(self) -> None:
    """Verify that virtual-site coordinates match the hardcoded reference."""
    for spec in VSITE_SYSTEMS:
      with self.subTest(system=spec.name):
        print(f'\n=== Testing vsite system: {spec.name} ===')
        test_dir = TEST_DIR / spec.test_dir

        self.assertTrue(
          test_dir.is_dir(), f'Test directory not found: {test_dir}'
        )

        ref_pos_list = VSITE_REFERENCE.get(spec.name)
        self.assertIsNotNone(
          ref_pos_list,
          f"No vsite reference positions found for system '{spec.name}'.",
        )
        ref_pos = jnp.array(ref_pos_list)

        epsilon_r = read_epsilon_r(test_dir)

        topology_file = GromacsTopFile(
          test_dir / 'system.top', epsilon_r=epsilon_r
        )

        gro_file = GromacsGroFile(test_dir / 'conf.gro')
        displacement_fn, shift_fn = periodic(gro_file.box)

        martini_topology = create_topology(
          topology_file, nonbonded_cutoff=1.1, epsilon_r=epsilon_r
        )

        vsite_indices = jnp.array([3, 4, 5, 6, 7, 8, 9, 10])
        input_pos = ref_pos.at[vsite_indices].set(jnp.zeros((3,)))

        result = martini_topology.apply_vsites(
          input_pos, displacement_fn, shift_fn
        )
        print('Result positions:\n', result)

        failures: list[str] = []
        for idx in vsite_indices:
          if not jnp.allclose(
            result[idx], ref_pos[idx], atol=CONSTRAINT_POS_TOL
          ):
            failures.append(
              f'Vsite {idx}: got {result[idx]}, expected {ref_pos[idx]}'
            )

        if failures:
          self.fail('\n\n'.join(failures))


if __name__ == '__main__':
  absltest.main()
