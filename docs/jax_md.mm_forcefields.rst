Molecular Mechanics Force Fields
=================================

.. default-role:: code

Topology and Parameters
------------------------

.. automodule:: jax_md.mm_forcefields.base

.. autoclass:: Topology
.. autoclass:: BondedParameters
.. autoclass:: NonbondedParameters
.. autoclass:: NonbondedOptions

.. autofunction:: combine_lorentz
.. autofunction:: combine_berthelot
.. autofunction:: combine_product
.. autofunction:: compute_angle
.. autofunction:: compute_dihedral
.. autofunction:: hard_cutoff
.. autofunction:: force_switch

Neighbor Lists
---------------

.. automodule:: jax_md.mm_forcefields.neighbor

.. autofunction:: create_neighbor_list
.. autofunction:: make_exclusion_mask
.. autofunction:: make_14_table

Electrostatics
---------------

.. automodule:: jax_md.mm_forcefields.nonbonded.electrostatics

.. autoclass:: CoulombHandler
   :members:
.. autoclass:: CutoffCoulomb
   :members:
.. autoclass:: EwaldCoulomb
   :members:
.. autoclass:: PMECoulomb
   :members:

AMBER
------

.. automodule:: jax_md.mm_forcefields.amber.energy

.. autofunction:: energy
.. autoclass:: FEOptions
.. autofunction:: space_selector

.. autofunction:: harmonic_bond
.. autofunction:: harmonic_angle
.. autofunction:: periodic_torsion
.. autofunction:: harmonic_improper
.. autofunction:: lennard_jones
.. autofunction:: lennard_jones_ab
.. autofunction:: lennard_jones_softcore
.. autofunction:: coulomb_softcore
.. autofunction:: cmap_setup
.. autofunction:: cmap_energy

Constraints
^^^^^^^^^^^^

.. automodule:: jax_md.mm_forcefields.amber.constraints

.. autofunction:: settle
.. autofunction:: ccma
.. autofunction:: prepare_settle_ccma
.. autofunction:: settle_apply_positions
.. autofunction:: settle_apply_velocities
.. autofunction:: ccma_apply_positions
.. autofunction:: ccma_apply_velocities
.. autoclass:: SettleData
.. autoclass:: CCMAData

OPLS-AA
--------

.. automodule:: jax_md.mm_forcefields.oplsaa

.. autofunction:: energy
.. autofunction:: load_charmm_system
.. autofunction:: create_topology
.. autofunction:: validate_topology
.. autofunction:: create_parameters
.. autofunction:: validate_parameters
.. autoclass:: Parameters

Input and Output
-----------------

OpenMM
^^^^^^^

.. automodule:: jax_md.mm_forcefields.io.openmm

.. autofunction:: load_amber_system
.. autofunction:: load_charmm_system
.. autofunction:: load_gromacs_system
.. autofunction:: load_generator_system
.. autofunction:: load_parmed_system
.. autofunction:: convert_openmm_system
.. autofunction:: virtual_site_apply_positions
.. autofunction:: virtual_site_fix_state
.. autoclass:: OpenMMSystem
.. autoclass:: VirtualSiteData

CHARMM
^^^^^^^

.. automodule:: jax_md.mm_forcefields.io.charmm

.. autofunction:: parse_rtf
.. autofunction:: parse_prm
.. autofunction:: parse_pdb_simple
.. autoclass:: AtomType
.. autoclass:: Atom
.. autoclass:: BondType
.. autoclass:: AngleType
.. autoclass:: DihedralType
