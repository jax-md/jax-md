import os

import jax
import pymatgen as mg
import pymatgen.core
from absl.testing import absltest, parameterized

from jax_md.a2c.crystallizer_utils import (
  get_subcells_to_crystallize,
  get_subcells_to_crystallize_parallel,
)

jax.config.update('jax_enable_x64', True)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
AMORPHOUS_SI64_PATH = os.path.join(DATA_DIR, 'amorphous_si64.cif')


class A2CTest(parameterized.TestCase):
  """Test the A2C workflow."""

  def test_subcells_to_crystallize_parallel(self):
    """Test the parallel version of the subcells to crystallize function."""
    amorphous_structure = mg.core.Structure.from_file(AMORPHOUS_SI64_PATH)

    # Run both versions of the crystallize function, timing each, and compare the results
    d_frac = 0.2
    nmin = 1
    nmax = 12
    n_workers = os.cpu_count() - 1

    orig = get_subcells_to_crystallize(
      amorphous_structure, d_frac=d_frac, nmin=nmin, nmax=nmax
    )
    orig = sorted(orig, key=lambda x: (tuple(x[0]), tuple(x[1]), tuple(x[2])))

    improved_subcell_generation_results = get_subcells_to_crystallize_parallel(
      amorphous_structure, d_frac=d_frac, nmin=nmin, nmax=nmax, n_workers=1
    )
    improved_subcell_generation_results = sorted(
      improved_subcell_generation_results,
      key=lambda x: (tuple(x[0]), tuple(x[1]), tuple(x[2])),
    )

    if n_workers > 1:
      parallel_and_improved_crystallize_results = (
        get_subcells_to_crystallize_parallel(
          amorphous_structure,
          d_frac=d_frac,
          nmin=nmin,
          nmax=nmax,
          n_workers=n_workers,
        )
      )
      parallel_and_improved_crystallize_results = sorted(
        parallel_and_improved_crystallize_results,
        key=lambda x: (tuple(x[0]), tuple(x[1]), tuple(x[2])),
      )

    # Compare results to confirm consistency between original and improved subcell generation functions
    self.assertEqual(
      len(orig),
      len(improved_subcell_generation_results),
      f'Original function and improved subcell generation function returned different number of subcells: {len(orig)} != {len(improved_subcell_generation_results)}',
    )
    if n_workers > 1:
      self.assertEqual(
        len(orig),
        len(parallel_and_improved_crystallize_results),
        f'Original function and parallel & improved crystallize function returned different number of subcells: {len(orig)} != {len(parallel_and_improved_crystallize_results)}',
      )
    else:
      print('Parallel function not tested because n_workers <= 1')


if __name__ == '__main__':
  absltest.main()
