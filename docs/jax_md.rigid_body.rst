Rigid Body Simulations
=======================

.. default-role:: code
.. automodule:: jax_md.rigid_body

Quaternion Utilities
---------------------

.. autoclass:: Quaternion
.. autofunction:: quaternion_rotate
.. autofunction:: random_quaternion

Rigid Body Simulation
----------------------

.. autoclass:: RigidBody
.. autofunction:: kinetic_energy
.. autofunction:: temperature
.. autofunction:: angular_momentum_to_conjugate_momentum
.. autofunction:: conjugate_momentum_to_angular_momentum

Rigid Collections of Points
----------------------------

.. autoclass:: RigidPointUnion
.. autofunction:: point_union_shape
.. autofunction:: concatenate_shapes
.. autofunction:: point_energy
.. autofunction:: point_energy_neighbor_list
.. autofunction:: transform
.. autofunction:: union_to_points
