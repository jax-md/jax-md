import os
import time

import jax
import pymatgen as mg
from absl.testing import absltest, parameterized

from jax_md.a2c.crystallizer_utils import (
  get_subcells_to_crystallize,
  get_subcells_to_crystallize_parallel,
)

jax.config.update('jax_enable_x64', True)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
AMORPHOUS_SI64_PATH = os.path.join(DATA_DIR, 'amorphous_si64.cif')


class A2CTest(parameterized.TestCase):
    """Test the A2C workflow.
    """


    def test_subcells_to_crystallize_parallel(self):
        """Test the parallel version of the subcells to crystallize function.
        """
        amorphous_structure = mg.core.Structure.from_file(AMORPHOUS_SI64_PATH)

        # Run both versions of the crystallize function, timing each, and compare the results
        d_frac = 0.08
        nmin = 1
        nmax = 12
        n_workers = os.cpu_count()-1

        start_time = time.time()
        orig = get_subcells_to_crystallize(amorphous_structure, d_frac=d_frac, nmin=nmin, nmax=nmax)
        orig_time = time.time() - start_time
        orig = sorted(orig, key=lambda x: (tuple(x[0]), tuple(x[1]), tuple(x[2])))

        start_time = time.time()
        improved_subcell_generation_results = get_subcells_to_crystallize_parallel(amorphous_structure, d_frac=d_frac, nmin=nmin, nmax=nmax, n_workers=1)
        improved_subcell_generation_time = time.time() - start_time
        improved_subcell_generation_results = sorted(improved_subcell_generation_results, key=lambda x: (tuple(x[0]), tuple(x[1]), tuple(x[2])))

        if n_workers > 1:
            start_time = time.time()
            parallel_and_improved_crystallize_results = get_subcells_to_crystallize_parallel(amorphous_structure, d_frac=d_frac, nmin=nmin, nmax=nmax, n_workers=n_workers)
            parallel_and_improved_crystallize_results_time = time.time() - start_time
            parallel_and_improved_crystallize_results = sorted(parallel_and_improved_crystallize_results, key=lambda x: (tuple(x[0]), tuple(x[1]), tuple(x[2])))

        print(f"Original crystallize function time   ({len(orig)} subcells):\t{orig_time:.2f}s")
        print(f"Improved subcell generation time     ({len(improved_subcell_generation_results)} subcells):\t{improved_subcell_generation_time:.2f}s")
        if n_workers > 1:
            print(f"Parallel & improved function time    ({len(parallel_and_improved_crystallize_results)} subcells):\t{parallel_and_improved_crystallize_results_time:.2f}s")
        print(f"Speedup (original vs. improved):                               \t{orig_time / improved_subcell_generation_time:.2f}x")
        if n_workers > 1:
            print(f"Speedup (original vs. parallel & improved) (# workers: {n_workers}):\t{orig_time / parallel_and_improved_crystallize_results_time:.2f}x\n\tNB: This speedup will increase with # of subcells generated.")

        # Compare results to confirm consistency between original and improved subcell generation functions
        self.assertEqual(
            len(orig),
            len(improved_subcell_generation_results),
            f"Original function and improved subcell generation function returned different number of subcells: {len(orig)} != {len(improved_subcell_generation_results)}",
        )
        if n_workers > 1:
            self.assertEqual(
                len(orig),
                len(parallel_and_improved_crystallize_results),
                f"Original function and parallel & improved crystallize function returned different number of subcells: {len(orig)} != {len(parallel_and_improved_crystallize_results)}",
            )
        else:
            print("Parallel function not tested because n_workers <= 1")

if __name__ == '__main__':
    absltest.main()
