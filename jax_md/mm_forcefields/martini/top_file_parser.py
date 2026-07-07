"""
Gromacs topology file parser. Adapted from OpenMM's gromacstopfile.py

Portions copyright (c) 2012-2025 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Jason Swails

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

from collections import OrderedDict
from copy import deepcopy
import os
import re
import shutil

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Defaults:
  """Fields specified in [ defaults ]. Martini v3 only provides 2 fields."""

  nb_func_type: int
  comb_rule: int


@dataclass
class Molecule:
  """Molecule."""

  name: str
  count: int


@dataclass
class Atom:
  """Atom in a residue."""

  idx: int
  type: str
  residue_number: int
  residue_name: str
  atom_name: str
  charge_group_num: int | None = None
  q: float | None = None
  mass: float | None = None


@dataclass
class AtomType:
  """Atom type definition."""

  name: str
  bonded_type: str | None
  at_num: float | None
  mass: float
  charge: float
  ptype: str
  v: float
  w: float


@dataclass
class BondedTerm:
  """Base class for bonded interaction terms."""

  atoms: List[int]
  func_type: str


@dataclass
class BondParams:
  """BondType parameters."""

  b0: float
  k: float


@dataclass
class Bond(BondedTerm):
  """Bond."""

  params: BondParams | None = None


@dataclass
class AngleParams:
  """AngleType parameters."""

  k: float
  theta0: float


@dataclass
class Angle(BondedTerm):
  """Angle."""

  params: AngleParams | None = None


@dataclass
class PeriodicTorsionParams:
  """Periodic torsion parameters."""

  phi0: float
  k: float
  n: int


@dataclass
class PeriodicTorsion(BondedTerm):
  """Periodic torsion."""

  params: PeriodicTorsionParams | None = None


@dataclass
class HarmonicTorsionParams:
  """Harmonic torsion parameters."""

  phi0: float
  k: float


@dataclass
class HarmonicTorsion(BondedTerm):
  """Harmonic torsion."""

  params: HarmonicTorsionParams | None = None


@dataclass
class RBTorsionParams:
  """Ryckaert-Bellemans torsion parameters."""

  c_array: List[float]


@dataclass
class RBTorsion(BondedTerm):
  """Ryckaert-Bellemans torsion."""

  params: RBTorsionParams | None = None


@dataclass
class CBTParams:
  """Combined bending-torsion parameters."""

  k: float
  a_array: List[float]


@dataclass
class CBT(BondedTerm):
  """Combined bending-torsion potential."""

  params: CBTParams | None = None


@dataclass
class Exclusion:
  """Non-bonded exclusion between atoms."""

  atoms: List[int]


@dataclass
class PairParams:
  """Pair interaction parameters."""

  v: float
  w: float


@dataclass
class Pair:
  """Explicit pair interaction."""

  atoms: List[int]
  params: PairParams | None = None


@dataclass
class Constraint:
  """Rigid geometric constraint between atoms."""

  atoms: List[int]
  distance: float


@dataclass
class CmapParams:
  """CMAP correction map parameters."""

  x_size: float
  y_size: float
  grid: List[float]


@dataclass
class Cmap:
  """CMAP correction term."""

  atoms: List[int]
  params: CmapParams | None = None


@dataclass
class NonbondedType:
  """Non-bonded interaction type parameters."""

  v: float
  w: float


@dataclass
class LinearVirtualSite:
  """Virtual site constructed as a linear combination of atom positions."""

  atom_weights: Dict[int, float]


@dataclass
class TwoAtomFdVirtualSite:
  """Virtual site defined along the vector between two atoms."""

  index: int
  atom_i: int
  atom_j: int
  d: float


@dataclass
class ThreeAtomFdVirtualSite:
  """Virtual site defined in the plane of three atoms with a fixed distance."""

  index: int
  atom_i: int
  atom_j: int
  atom_k: int
  a: float
  d: float


@dataclass
class ThreeAtomOutVirtualSite:
  """Virtual site defined out of the plane of three atoms."""

  index: int
  atom_i: int
  atom_j: int
  atom_k: int
  a: float
  b: float
  c: float


@dataclass
class NAtomCOMVirtualSite:
  """Virtual site placed at the center of mass of N atoms."""

  index: int
  atoms: List[int]


class GromacsTopFile:
  """Parse a Gromacs Martini top file and constructs a Topology. Adapted from OpenMM (MIT License)"""

  class _MoleculeType:
    """Inner class to store information about a molecule type."""

    def __init__(self, name, nrexcl):
      self.name = name
      self.nrexcl = nrexcl
      self.atoms = []
      self.bonds = []
      self.angles = []
      self.dihedrals = []
      self.exclusions = []
      self.pairs = []
      self.constraints = []
      self.cmaps = []
      self.linear_vsites = {}
      self.two_fd_vsites = []
      self.three_fd_vsites = []
      self.three_out_vsites = []
      self.n_com_vsites = []
      self.has_virtual_sites = False
      self.has_nbfix_terms = False

    def findExclusionsFromBonds(self, genpairs):
      """Find exclusions between atoms separated by up to nrexcl bonds if genpairs is false,
      or up to 2 bonds if genpairs is true.
      """
      bondedTo = [set() for i in range(len(self.atoms))]
      for bond in self.bonds:
        i = bond.atoms[0]
        j = bond.atoms[1]
        bondedTo[i].add(j)
        bondedTo[j].add(i)

      # Identify all neighbors of each atom with each separation.

      bondedWithSeparation = [bondedTo]
      maxBonds = self.nrexcl
      if genpairs:
        maxBonds = min(maxBonds, 2)
      for i in range(maxBonds - 1):
        lastBonds = bondedWithSeparation[-1]
        newBonds = deepcopy(lastBonds)
        for atom in range(len(self.atoms)):
          for a1 in lastBonds[atom]:
            for a2 in bondedTo[a1]:
              newBonds[atom].add(a2)
        bondedWithSeparation.append(newBonds)

      # Build the list of pairs.

      pairs = []
      for atom in range(len(self.atoms)):
        for otherAtom in bondedWithSeparation[-1][atom]:
          if otherAtom > atom:
            pairs.append((atom, otherAtom))
      return pairs

  def __init__(
    self,
    file,
    includeDir=None,
    defines=None,
    epsilon_r=15.0,
  ):
    """Load a top file.

    Parameters
    ----------
    file : str
      the name of the file to load
    includeDir : string=None
      A directory in which to look for other files included from the
      top file. If not specified, we will attempt to locate a gromacs
      installation on your system. When gromacs is installed in
      /usr/local, this will resolve to /usr/local/gromacs/share/gromacs/top
    defines : dict={}
      preprocessor definitions that should be predefined when parsing the file
    TODO: Pass in defines
    """
    self.epsilon_r = epsilon_r
    if includeDir is None:
      includeDir = _default_gromacs_include_dir()
    self._includeDirs = (os.path.dirname(file), includeDir)
    # Most of the gromacs water itp files for different forcefields,
    # unless the preprocessor #define FLEXIBLE is given, don't define
    # bonds between the water hydrogen and oxygens, but only give the
    # constraint distances and exclusions.
    self._defines = OrderedDict()
    # self._defines["FLEXIBLE"] = True
    self._genpairs = True
    if defines is not None:
      for define, value in defines.items():
        self._defines[define] = value

    # Parse the file.

    self._currentCategory = None
    self._ifStack = []
    self._elseStack = []
    self._moleculeTypes = {}
    self._molecules = []
    self._currentMoleculeType = None
    self._atomTypes = {}
    self._bondTypes = {}
    self._angleTypes = {}
    self._dihedralTypes = {}
    self._pairTypes = {}
    self._cmapTypes = {}
    self._nonbondTypes = {}
    self._processFile(file)

  def _processFile(self, file):
    append = ''
    with open(file) as f:
      for line in f:
        if line.strip().endswith('\\'):
          trimmed = line[: line.rfind('\\')]
          append = f'{append} {trimmed}'
        else:
          self._processLine(append + ' ' + line, file)
          append = ''

  def _processLine(self, line, file):
    """Process one line from a file."""
    if ';' in line:
      line = line[: line.index(';')]
    stripped = line.strip()
    ignore = not all(self._ifStack)
    if stripped.startswith('*') or len(stripped) == 0:
      # A comment or empty line.
      return

    elif stripped.startswith('[') and not ignore:
      # The start of a category.
      if not stripped.endswith(']'):
        raise ValueError('Illegal line in .top file: ' + line)
      self._currentCategory = stripped[1:-1].strip()

    elif stripped.startswith('#'):
      # A preprocessor command.
      fields = stripped.split()
      command = fields[0]
      if len(self._ifStack) != len(self._elseStack):
        raise RuntimeError('#if/#else stack out of sync')

      if command == '#include' and not ignore:
        # Locate the file to include
        name = stripped[len(command) :].strip(' \t"<>')
        searchDirs = self._includeDirs + (os.path.dirname(file),)
        for dir in searchDirs:
          file = os.path.join(dir, name)
          if os.path.isfile(file):
            # We found the file, so process it.
            self._processFile(file)
            break
        else:
          raise ValueError('Could not locate #include file: ' + name)
      elif command == '#define' and not ignore:
        # Add a value to our list of defines.
        if len(fields) < 2:
          raise ValueError('Illegal line in .top file: ' + line)
        name = fields[1]
        valueStart = stripped.find(name, len(command)) + len(name) + 1
        value = line[valueStart:].strip()
        value = value or '1'  # Default define is 1
        self._defines[name] = value
      elif command == '#ifdef':
        # See whether this block should be ignored.
        if len(fields) < 2:
          raise ValueError('Illegal line in .top file: ' + line)
        name = fields[1]
        self._ifStack.append(name in self._defines)
        self._elseStack.append(False)
      elif command == '#undef':
        # Un-define a variable
        if len(fields) < 2:
          raise ValueError('Illegal line in .top file: ' + line)
        if fields[1] in self._defines:
          self._defines.pop(fields[1])
      elif command == '#ifndef':
        # See whether this block should be ignored.
        if len(fields) < 2:
          raise ValueError('Illegal line in .top file: ' + line)
        name = fields[1]
        self._ifStack.append(name not in self._defines)
        self._elseStack.append(False)
      elif command == '#endif':
        # Pop an entry off the if stack.
        if len(self._ifStack) == 0:
          raise ValueError('Unexpected line in .top file: ' + line)
        del self._ifStack[-1]
        del self._elseStack[-1]
      elif command == '#else':
        # Reverse the last entry on the if stack
        if len(self._ifStack) == 0:
          raise ValueError('Unexpected line in .top file: ' + line)
        if self._elseStack[-1]:
          raise ValueError(
            'Unexpected line in .top file: #else has already been used ' + line
          )
        self._ifStack[-1] = not self._ifStack[-1]
        self._elseStack[-1] = True

    elif not ignore:
      # Gromacs occasionally uses #define's to introduce specific
      # parameters for individual terms (for instance, this is how
      # ff99SB-ILDN is implemented). So make sure we do the appropriate
      # pre-processor replacements necessary
      line = _replace_defines(line, self._defines)
      # A line of data for the current category
      if self._currentCategory is None:
        raise ValueError('Unexpected line in .top file: ' + line)
      if self._currentCategory == 'defaults':
        self._processDefaults(line)
      elif self._currentCategory == 'moleculetype':
        self._processMoleculeType(line)
      elif self._currentCategory == 'molecules':
        self._processMolecule(line)
      elif self._currentCategory == 'atoms':
        self._processAtom(line)
      elif self._currentCategory == 'bonds':
        self._processBond(line)
      elif self._currentCategory == 'angles':
        self._processAngle(line)
      elif self._currentCategory == 'dihedrals':
        self._processDihedral(line)
      elif self._currentCategory == 'exclusions':
        self._processExclusion(line)
      elif self._currentCategory == 'pairs':
        self._processPair(line)
      elif self._currentCategory == 'constraints':
        self._processConstraint(line)
      elif self._currentCategory == 'cmap':
        self._processCmap(line)
      elif self._currentCategory == 'atomtypes':
        self._processAtomType(line)
      elif self._currentCategory == 'bondtypes':
        self._processBondType(line)
      elif self._currentCategory == 'angletypes':
        self._processAngleType(line)
      elif self._currentCategory == 'dihedraltypes':
        self._processDihedralType(line)
      elif self._currentCategory == 'pairtypes':
        self._processPairType(line)
      elif self._currentCategory == 'cmaptypes':
        self._processCmapType(line)
      elif self._currentCategory == 'nonbond_params':
        self._processNonbondType(line)
      elif self._currentCategory == 'virtual_sites2':
        self._processVirtualSites2(line)
      elif self._currentCategory == 'virtual_sites3':
        self._processVirtualSites3(line)
      elif self._currentCategory == 'virtual_sitesn':
        self._processVirtualSitesn(line)
      elif self._currentCategory.startswith('virtual_sites'):
        if self._currentMoleculeType is None:
          raise ValueError(
            f'Found {self._currentCategory} before [ moleculetype ]'
          )
        raise ValueError('Virtual sites not yet supported by Gromacs parsers')

  def _processDefaults(self, line):
    """Process the [ defaults ] line."""
    fields = line.split()
    if len(fields) > 2:
      raise ValueError(
        'Too many fields in [ defaults ] line for Martini force-field: ' + line
      )
    if fields[0] != '1':
      raise ValueError('Unsupported nonbonded type: ' + fields[0])
    if fields[1] not in ('1', '2', '3'):
      raise ValueError('Unsupported combination rule: ' + fields[1])
    self._defaults = Defaults(
      nb_func_type=int(fields[0]), comb_rule=int(fields[1])
    )

  def _processMoleculeType(self, line):
    """Process a line in the [ moleculetypes ] category."""
    fields = line.split()
    if len(fields) < 1:
      raise ValueError('Too few fields in [ moleculetypes ] line: ' + line)
    type = GromacsTopFile._MoleculeType(fields[0], int(fields[1]))
    self._moleculeTypes[fields[0]] = type
    self._currentMoleculeType = type

  def _processMolecule(self, line):
    """Process a line in the [ molecules ] category."""
    fields = line.split()
    if len(fields) < 2:
      raise ValueError('Too few fields in [ molecules ] line: ' + line)
    self._molecules.append(Molecule(name=fields[0], count=int(fields[1])))

  def _processAtom(self, line):
    """Process a line in the [ atoms ] category."""
    if self._currentMoleculeType is None:
      raise ValueError('Found [ atoms ] section before [ moleculetype ]')
    fields = line.split()
    if len(fields) < 5:
      raise ValueError('Too few fields in [ atoms ] line: ' + line)
    atom = Atom(
      idx=int(fields[0]),
      type=fields[1],
      residue_number=int(fields[2]),
      residue_name=fields[3],
      atom_name=fields[4],
      charge_group_num=int(fields[5]) if len(fields) > 5 else None,
      q=float(fields[6]) if len(fields) > 6 else None,
      mass=float(fields[7]) if len(fields) > 7 else None,
    )
    self._currentMoleculeType.atoms.append(atom)

  def _processBond(self, line):
    """Process a line in the [ bonds ] category."""
    if self._currentMoleculeType is None:
      raise ValueError('Found [ bonds ] section before [ moleculetype ]')
    fields = line.split()
    if len(fields) < 3:
      raise ValueError('Too few fields in [ bonds ] line: ' + line)
    func_type = fields[2]
    if func_type not in ('1', '6'):
      raise ValueError('Unsupported function type in [ bonds ] line: ' + line)
    bond = Bond(
      atoms=[int(fields[i]) - 1 for i in range(2)],
      func_type=func_type,
      params=(
        BondParams(b0=float(fields[3]), k=float(fields[4]))
        if len(fields) > 4
        else None
      ),
    )
    self._currentMoleculeType.bonds.append(bond)

  def _processAngle(self, line):
    """Process a line in the [ angles ] category."""
    if self._currentMoleculeType is None:
      raise ValueError('Found [ angles ] section before [ moleculetype ]')
    fields = line.split()
    if len(fields) < 4:
      raise ValueError('Too few fields in [ angles ] line: ' + line)
    func_type = fields[3]
    if func_type not in ('1', '2', '10'):
      raise ValueError('Unsupported function type in [ angles ] line: ' + line)
    self._currentMoleculeType.angles.append(
      Angle(
        atoms=[int(fields[i]) - 1 for i in range(3)],
        func_type=func_type,
        params=(
          AngleParams(theta0=float(fields[4]), k=float(fields[5]))
          if len(fields) > 5
          else None
        ),
      )
    )

  def _processDihedral(self, line):
    """Process a line in the [ dihedrals ] category."""
    if self._currentMoleculeType is None:
      raise ValueError('Found [ dihedrals ] section before [ moleculetype ]')
    fields = line.split()
    if len(fields) < 5:
      raise ValueError('Too few fields in [ dihedrals ] line: ' + line)
    func_type = fields[4]
    if func_type not in ('1', '2', '3', '4', '5', '9', '11'):
      raise ValueError(
        'Unsupported function type in [ dihedrals ] line: ' + line
      )

    atoms = [int(fields[i]) - 1 for i in range(4)]
    if func_type in ('1', '4', '9'):
      self._currentMoleculeType.dihedrals.append(
        PeriodicTorsion(
          atoms=atoms,
          func_type=func_type,
          params=(
            PeriodicTorsionParams(
              phi0=float(fields[5]),
              k=float(fields[6]),
              n=int(float(fields[7])),
            )
            if len(fields) > 7
            else None
          ),
        )
      )
    elif func_type == '2':
      self._currentMoleculeType.dihedrals.append(
        HarmonicTorsion(
          atoms=atoms,
          func_type=func_type,
          params=(
            HarmonicTorsionParams(
              phi0=float(fields[5]),
              k=float(fields[6]),
            )
            if len(fields) > 6
            else None
          ),
        )
      )
    elif func_type in ('3', '5'):
      self._currentMoleculeType.dihedrals.append(
        RBTorsion(
          atoms=atoms,
          func_type=func_type,
          params=(
            RBTorsionParams(c_array=[float(x) for x in fields[5:]])
            if len(fields) > 8
            else None
          ),
        )
      )
    elif func_type == '11':
      self._currentMoleculeType.dihedrals.append(
        CBT(
          atoms=atoms,
          func_type=func_type,
          params=(
            CBTParams(
              k=float(fields[5]), a_array=[float(x) for x in fields[6:11]]
            )
            if len(fields) > 10
            else None
          ),
        )
      )

  def _processExclusion(self, line):
    """Process a line in the [ exclusions ] category."""
    if self._currentMoleculeType is None:
      raise ValueError('Found [ exclusions ] section before [ moleculetype ]')
    fields = line.split()
    if len(fields) < 2:
      raise ValueError('Too few fields in [ exclusions ] line: ' + line)
    self._currentMoleculeType.exclusions.append(
      Exclusion(atoms=[int(x) - 1 for x in fields])
    )

  def _processPair(self, line):
    """Process a line in the [ pairs ] category."""
    if self._currentMoleculeType is None:
      raise ValueError('Found [ pairs ] section before [ moleculetype ]')
    fields = line.split()
    if len(fields) < 3:
      raise ValueError('Too few fields in [ pairs ] line: ' + line)
    if fields[2] != '1':
      raise ValueError('Unsupported function type in [ pairs ] line: ' + line)
    self._currentMoleculeType.pairs.append(
      Pair(
        atoms=[int(x) - 1 for x in fields[:2]],
        params=(
          PairParams(v=float(fields[3]), w=float(fields[4]))
          if len(fields) > 4
          else None
        ),
      )
    )

  def _processConstraint(self, line):
    """Process a line in the [ constraints ] category."""
    if self._currentMoleculeType is None:
      raise ValueError('Found [ constraints ] section before [ moleculetype ]')
    fields = line.split()
    if len(fields) < 4:
      raise ValueError('Too few fields in [ constraints ] line: ' + line)
    if fields[2] != '1':
      raise ValueError(
        'Unsupported function type in [ constraints ] line: ' + line
      )
    self._currentMoleculeType.constraints.append(
      Constraint(
        atoms=[int(x) - 1 for x in fields[:2]], distance=float(fields[3])
      )
    )

  def _processCmap(self, line):
    """Process a line in the [ cmaps ] category."""
    if self._currentMoleculeType is None:
      raise ValueError('Found [ cmap ] section before [ moleculetype ]')
    fields = line.split()
    if len(fields) < 6:
      raise ValueError('Too few fields in [ cmap ] line: ' + line)
    self._currentMoleculeType.cmaps.append(
      Cmap(
        atoms=[int(x) - 1 for x in fields[:5]],
        params=(
          CmapParams(
            x_size=int(fields[6]),
            y_size=int(fields[7]),
            grid=fields[8:],
          )
          if (
            len(fields) >= 8
            and len(fields) >= 8 + int(fields[6]) * int(fields[7])
          )
          else None
        ),
      )
    )

  def _processAtomType(self, line):
    """Process a line in the [ atomtypes ] category."""
    fields = line.split()
    if len(fields) < 6:
      raise ValueError('Too few fields in [ atomtypes ] line: ' + line)
    if len(fields[3]) == 1:
      # Bonded type and atomic number are both missing.
      fields.insert(1, None)
      fields.insert(1, None)
    elif len(fields[4]) == 1 and fields[4].isalpha():
      if fields[1][0].isalpha():
        # Atomic number is missing.
        fields.insert(2, None)
      else:
        # Bonded type is missing.
        fields.insert(1, None)
    self._atomTypes[fields[0]] = AtomType(
      name=fields[0],
      bonded_type=fields[1] if fields[2] is not None else None,
      at_num=float(fields[2]) if fields[2] is not None else None,
      mass=float(fields[3]),
      charge=float(fields[4]),
      ptype=fields[5],
      v=float(fields[6]),
      w=float(fields[7]),
    )

  def _processBondType(self, line):
    """Process a line in the [ bondtypes ] category."""
    fields = line.split()
    if len(fields) < 5:
      raise ValueError('Too few fields in [ bondtypes ] line: ' + line)
    if fields[2] != '1':
      raise ValueError(
        'Unsupported function type in [ bondtypes ] line: ' + line
      )
    self._bondTypes[tuple(fields[:3])] = BondParams(
      b0=float(fields[3]),
      k=float(fields[4]),
    )

  def _processAngleType(self, line):
    """Process a line in the [ angletypes ] category."""
    fields = line.split()
    if len(fields) < 6:
      raise ValueError('Too few fields in [ angletypes ] line: ' + line)
    if fields[3] not in ('1', '2', '10'):
      raise ValueError(
        'Unsupported function type in [ angletypes ] line: ' + line
      )
    self._angleTypes[tuple(fields[:3])] = AngleParams(
      theta0=float(fields[4]),
      k=float(fields[5]),
    )

  def _processDihedralType(self, line):
    """Process a line in the [ dihedraltypes ] category."""
    fields = line.split()
    if len(fields[2]) == 1 and fields[2].isdigit():
      # The third field contains the function type, meaning only two atom types are specified.
      # Interpret them as the two inner ones.
      fields = ['X', fields[0], fields[1], 'X'] + fields[2:]
    if len(fields) < 7:
      raise ValueError('Too few fields in [ dihedraltypes ] line: ' + line)
    func_type = fields[4]
    if func_type not in ('1', '2', '3', '4', '5', '9'):
      raise ValueError(
        'Unsupported function type in [ dihedraltypes ] line: ' + line
      )
    key = tuple(fields[:5])
    if func_type in ('1', '4', '9'):
      dihedral_params = PeriodicTorsionParams(
        phi0=float(fields[5]), k=float(fields[6]), n=int(float(fields[7]))
      )
      if func_type == '9' and key in self._dihedralTypes:
        # There are multiple dihedrals defined for these atom types.
        self._dihedralTypes[key].append(dihedral_params)
      else:
        self._dihedralTypes[key] = [dihedral_params]
    elif func_type == '2':
      self._dihedralTypes[key] = [
        HarmonicTorsionParams(phi0=float(fields[5]), k=float(fields[6]))
      ]
    elif func_type in ('3', '5'):
      self._dihedralTypes[key] = [
        RBTorsionParams(c_array=[float(x) for x in fields[5:]])
      ]

  def _processPairType(self, line):
    """Process a line in the [ pairtypes ] category."""
    fields = line.split()
    if len(fields) < 5:
      raise ValueError('Too few fields in [ pairtypes] line: ' + line)
    if fields[2] != '1':
      raise ValueError(
        'Unsupported function type in [ pairtypes ] line: ' + line
      )
    self._pairTypes[tuple(fields[:2])] = PairParams(
      v=float(fields[3]), w=float(fields[4])
    )

  def _processCmapType(self, line):
    """Process a line in the [ cmaptypes ] category."""
    fields = line.split()
    if len(fields) < 8 or len(fields) < 8 + int(fields[6]) * int(fields[7]):
      raise ValueError('Too few fields in [ cmaptypes ] line: ' + line)
    if fields[5] != '1':
      raise ValueError(
        'Unsupported function type in [ cmaptypes ] line: ' + line
      )
    self._cmapTypes[tuple(fields[:5])] = CmapParams(
      x_size=int(fields[6]),
      y_size=int(fields[7]),
      grid=fields[8:],
    )

  def _processNonbondType(self, line):
    """Process a line in the [ nonbond_params ] category."""
    fields = line.split()
    if len(fields) < 5:
      raise ValueError('Too few fields in [ nonbond_params ] line: ' + line)
    if fields[2] != '1':
      raise ValueError(
        'Unsupported function type in [ nonbond_params ] line: ' + line
      )
    self._nonbondTypes[tuple(sorted(fields[:2]))] = NonbondedType(
      v=float(fields[3]), w=float(fields[4])
    )

  def _processVirtualSites2(self, line):
    """Process a line in the [ virtual_sites2 ] category."""
    fields = line.split()
    if len(fields) < 5:
      raise ValueError('Too few fields in [ virtual_sites2 ] line: ' + line)
    index = int(fields[0])
    atom_i = int(fields[1])
    atom_j = int(fields[2])
    func_type = int(fields[3])
    if func_type not in (1, 2):
      raise ValueError(
        'Unsupported function type in [ virtual_sites2 ] line: ' + line
      )
    if func_type == 1:
      a = float(fields[4])
      self._currentMoleculeType.linear_vsites[index] = LinearVirtualSite(
        atom_weights={atom_i: 1 - a, atom_j: a}
      )
    elif func_type == 2:
      self._currentMoleculeType.two_fd_vsites.append(
        TwoAtomFdVirtualSite(index, atom_i, atom_j, d=float(fields[4]))
      )

  def _processVirtualSites3(self, line):
    """Process a line in the [ virtual_sites3 ] category."""
    fields = line.split()
    if len(fields) < 7:
      raise ValueError('Too few fields in [ virtual_sites3 ] line: ' + line)
    func_type = int(fields[4])
    if func_type not in (1, 2, 4):
      raise ValueError(
        'Unsupported function type in [ virtual_sites3 ] line: ' + line
      )
    index = int(fields[0])
    atom_i = int(fields[1])
    atom_j = int(fields[2])
    atom_k = int(fields[3])
    if func_type == 1:
      a, b = float(fields[5]), float(fields[6])
      self._currentMoleculeType.linear_vsites[index] = LinearVirtualSite(
        atom_weights={atom_i: 1 - a - b, atom_j: a, atom_k: b}
      )
    elif func_type == 2:
      self._currentMoleculeType.three_fd_vsites.append(
        ThreeAtomFdVirtualSite(
          index,
          atom_i,
          atom_j,
          atom_k,
          a=float(fields[5]),
          d=float(fields[6]),
        )
      )
    elif func_type == 4:
      if len(fields) < 8:
        raise ValueError('Too few fields in [ virtual_sites3 ] line: ' + line)
      self._currentMoleculeType.three_out_vsites.append(
        ThreeAtomOutVirtualSite(
          index,
          atom_i,
          atom_j,
          atom_k,
          a=float(fields[5]),
          b=float(fields[6]),
          c=float(fields[7]),
        )
      )

  def _processVirtualSitesn(self, line):
    """Process a line in the [ virtual_sitesn ] category."""
    fields = line.split()
    if len(fields) < 3:
      raise ValueError('Too few fields in [ virtual_sitesn ] line: ' + line)
    func_type = int(fields[1])
    if func_type not in (1, 2):
      raise ValueError(
        'Unsupported function type in [ virtual_sitesn ] line: ' + line
      )
    index = int(fields[0])
    atoms = [int(field) for field in fields[2:]]

    if func_type == 1:
      w = 1.0 / len(atoms)
      self._currentMoleculeType.linear_vsites[index] = LinearVirtualSite(
        atom_weights={atom: w for atom in atoms}
      )
    elif func_type == 2:
      w = 1  # Dummy that will be replaced with weights based on atom masses
      self._currentMoleculeType.n_com_vsites.append(
        NAtomCOMVirtualSite(index, atoms=atoms)
      )


def _find_all_instances_in_string(string, substr):
  """Find indices of all instances of substr in string"""
  indices = []
  idx = string.find(substr, 0)
  while idx > -1:
    indices.append(idx)
    idx = string.find(substr, idx + 1)
  return indices


def _replace_defines(line, defines):
  """Replaces defined tokens in a given line"""
  novarcharre = re.compile(r'\W')
  if not defines:
    return line
  for define in reversed(defines):
    value = defines[define]
    indices = _find_all_instances_in_string(line, define)
    if not indices:
      continue
    # Check to see if it's inside of quotes
    inside = ''
    idx = 0
    n_to_skip = 0
    new_line = []
    for i, char in enumerate(line):
      if n_to_skip:
        n_to_skip -= 1
        continue
      if char in ('\'"'):
        if not inside:
          inside = char
        else:
          if inside == char:
            inside = ''
      if idx < len(indices) and i == indices[idx]:
        if inside:
          new_line.append(char)
          idx += 1
          continue
        if i == 0 or novarcharre.match(line[i - 1]):
          endidx = indices[idx] + len(define)
          if endidx >= len(line) or novarcharre.match(line[endidx]):
            new_line.extend(list(value))
            n_to_skip = len(define) - 1
            idx += 1
            continue
        idx += 1
      new_line.append(char)
    line = ''.join(new_line)

  return line


def _default_gromacs_include_dir():
  """Find the location where gromacs #include files are referenced from, by
  searching for (1) gromacs environment variables, (2) for the gromacs binary
  'pdb2gmx' or 'gmx' in the PATH, or (3) just using the default gromacs
  install location, /usr/local/gromacs/share/gromacs/top"""
  if 'GMXDATA' in os.environ:
    return os.path.join(os.environ['GMXDATA'], 'top')
  if 'GMXBIN' in os.environ:
    return os.path.abspath(
      os.path.join(os.environ['GMXBIN'], '..', 'share', 'gromacs', 'top')
    )

  pdb2gmx_path = shutil.which('pdb2gmx')
  if pdb2gmx_path is not None:
    return os.path.abspath(
      os.path.join(
        os.path.dirname(pdb2gmx_path), '..', 'share', 'gromacs', 'top'
      )
    )
  else:
    gmx_path = shutil.which('gmx')
    if gmx_path is not None:
      return os.path.abspath(
        os.path.join(os.path.dirname(gmx_path), '..', 'share', 'gromacs', 'top')
      )

  return '/usr/local/gromacs/share/gromacs/top'
