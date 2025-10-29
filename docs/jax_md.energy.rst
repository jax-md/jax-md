Potential Energy Functions
===========================

.. default-role:: code
.. automodule:: jax_md.energy

Bond Potentials
----------------
.. autofunction:: simple_spring
.. autofunction:: simple_spring_bond

Classical Potentials
---------------------
.. autofunction:: soft_sphere
.. autofunction:: soft_sphere_pair
.. autofunction:: soft_sphere_neighbor_list

.. autofunction:: lennard_jones
.. autofunction:: lennard_jones_pair
.. autofunction:: lennard_jones_neighbor_list

.. autofunction:: morse
.. autofunction:: morse_pair
.. autofunction:: morse_neighbor_list

.. autofunction:: gupta_potential
.. autofunction:: gupta_gold55

.. autofunction:: bks
.. autofunction:: bks_pair
.. autofunction:: bks_neighbor_list

.. autofunction:: bks_silica_pair
.. autofunction:: bks_silica_neighbor_list

.. autofunction:: stillinger_weber
.. autofunction:: stillinger_weber_neighbor_list

.. autofunction:: load_lammps_tersoff_parameters
.. autofunction:: tersoff
.. autofunction:: tersoff_neighbor_list
.. autofunction:: tersoff_from_lammps_parameters_neighbor_list

.. autofunction:: load_lammps_eam_parameters
.. autofunction:: eam
.. autofunction:: eam_from_lammps_parameters
.. autofunction:: eam_neighbor_list
.. autofunction:: eam_from_lammps_parameters_neighbor_list

Neural Network Potentials
--------------------------

.. autofunction:: behler_parrinello
.. autofunction:: behler_parrinello_neighbor_list

.. autofunction:: graph_network
.. autofunction:: graph_network_neighbor_list

Helper Functions
-----------------
.. autofunction:: multiplicative_isotropic_cutoff
.. autofunction:: load_lammps_eam_parameters
