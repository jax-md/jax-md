Simulation Routines
===========================

.. default-role:: code
.. automodule:: jax_md.simulate

Deterministic Simulation Environments
--------------------------------------
.. autofunction:: nve
.. autofunction:: nvt_nose_hoover
.. autofunction:: npt_nose_hoover

Stochastic Simulation Environments
--------------------------------------
.. autofunction:: nvt_langevin
.. autofunction:: brownian
.. autofunction:: npt_csvr
.. autofunction:: hybrid_swap_mc

Helper Functions
-----------------
.. autofunction:: velocity_verlet
.. autofunction:: nose_hoover_chain
.. autofunction:: npt_box
.. autofunction:: sinhx_x

Testing Functions
------------------
.. autofunction:: nvt_nose_hoover_invariant
.. autofunction:: npt_nose_hoover_invariant

Data Types
-----------
.. autoclass:: NoseHooverChain
.. autoclass:: NVEState
.. autoclass:: NVTNoseHooverState
.. autoclass:: NPTNoseHooverState
.. autoclass:: NVTLangevinState
.. autoclass:: BrownianState
.. autoclass:: NPTCSVRState
