Useful Quantities
===========================

.. default-role:: code
.. automodule:: jax_md.quantity

Forces
-------

.. autofunction:: force
.. autofunction:: clipped_force
.. autofunction:: canonicalize_force

Structural Features
--------------------

.. autofunction:: cosine_angles
.. autofunction:: cosine_angle_between_two_vectors
.. autofunction:: pair_correlation
.. autofunction:: pair_correlation_neighbor_list
.. autofunction:: average_pair_correlation_results
.. autofunction:: phop

Physical Properties
--------------------

.. autofunction:: kinetic_energy
.. autofunction:: temperature
.. autofunction:: count_dof
.. autofunction:: pressure
.. autofunction:: stress
.. autofunction:: bulk_modulus
.. autofunction:: gamma_from_stokes_law_3d

Boxes and Volumes
------------------

.. autofunction:: volume
.. autofunction:: box_size_at_number_density
.. autofunction:: box_size_at_volume_fraction
.. autofunction:: box_from_parameters
.. autofunction:: volume_fraction
.. autofunction:: particle_volume
.. autofunction:: nball_unit_volume

Data Types
-----------

.. autoclass:: PHopState
