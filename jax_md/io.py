# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as np

def _xyz_vec2str(x):
  """Convert a vector to string.
  """
  return "\t".join([str(i) for i in x])

def write_xyz(filename, *args):
  """Write arrays to xyz file format.

  Args:
    filename: Output filename.
    args: Arguments can be 1D or 2D arrays where length (along axis=0) is equal to number of atoms.

  Examples:
    write_xyz('minimize.xyz', R)
    write_xyz('minimize.xyz', species, R)
    write_xyz('minimize.xyz', species, R, velocities, forces)
  """
  vars = [arg[:, np.newaxis] if len(arg.shape)==1 else arg for arg in args]
  vars = np.hstack(vars)

  with open(filename, "w+") as f:
    N = len(vars)
    str_ = f"{N}" + "\n\n"
    f.write(str_)
    for j in range(N):
      str_ = f"{j+1}\t" + _xyz_vec2str(vars[j, :]) + "\n"
      f.write(str_)
