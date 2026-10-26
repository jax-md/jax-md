Molecular Dynamics on TPU
==========================

.. default-role:: code
.. automodule:: jax_md.tpu

Grids
------
.. autoclass:: TPUGrid
.. autofunction:: to_grid
.. autofunction:: from_grid
.. autofunction:: random_grid
.. autofunction:: nearest_valid_grid_size

Grid Operations
----------------
.. autofunction:: compute_displacement
.. autofunction:: accumulate_on_grid
.. autofunction:: shift
.. autofunction:: parallelize
.. autofunction:: unfold_mesh

Energy Functions
-----------------
.. autofunction:: pair_potential
.. autofunction:: soft_sphere
.. autofunction:: lennard_jones

Simulation
-----------
.. autofunction:: nve
.. autofunction:: kinetic_energy
.. autoclass:: NVEState
