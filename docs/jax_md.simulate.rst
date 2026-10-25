Simulation Routines
===========================

.. default-role:: code
.. automodule:: jax_md.simulate

Fixed Center Of Mass
--------------------
Pass ``fix_center_of_mass=True`` to a simulation constructor to remove
center-of-mass (COM) momentum, including from explicitly supplied momenta.
Initialization leaves positions and forces unchanged. Each subsequent
momentum kick obeys the same constraint. This option is intended for
translationally invariant potentials, not an imposed net force.

The momentum projection is
``p_i <- p_i - m_i * sum(p) / sum(m)``. It works for unequal masses and removes
only global translation, not rotation. A system of ``N`` particles in ``d``
dimensions has ``d * (N - 1)`` kinetic degrees of freedom. Multiple arrays
share one global COM constraint. This option supports particle dynamics with
momenta only; Brownian and rigid-body dynamics are not covered.

Periodic runs can use wrapped positions and existing neighbour lists. Wrapping
can change the numerical mean of the stored coordinates even when COM momentum
is zero; it does not represent COM drift. In NPT, the usual box dilation still
acts on positions: zero COM momentum does not prevent their affine scaling with
the box. NPT CSVR requires fractional positions. NPT Nose-Hoover's position
update is unchanged by this option.

``simulate.temperature(state)`` automatically uses the constrained kinetic
degrees of freedom. When using ``quantity.temperature`` directly, pass
``fix_center_of_mass=True`` explicitly. The old Langevin ``center_velocity``
argument remains unused for compatibility; it does not activate this option.
Fixed-COM Langevin requires scalar uniform translational friction.

Both barostats retain the standard NPT volume distribution after integrating
out COM, rather than changing finite-system averages by one particle.
This is a different finite-system convention from i-PI's ``BaroBZP``, which
retains its ``kT`` piston correction and ``-kT * log(V)`` term with ``fixcom``.
For NPT CSVR this means ``3 * N - 2`` particle-plus-piston kinetic degrees of
freedom, a ``2 * kT`` piston correction, and ``-2 * kT * log(V)`` in the
modified Hamiltonian. For the existing MTK-style Nose-Hoover equations,
``alpha = 1 + d / (d * (N - 1))``, the piston force gains ``d * kT``, and the
extended invariant gains ``-kT * log(V)``. These corrections account for the
extra volume weight of the eliminated COM coordinate; see
`Bussi et al., 2009 <https://arxiv.org/abs/0901.0779>`_. In particular, an ideal
gas still has ``p(V)`` proportional to ``V**N * exp(-pressure * V / kT)``.

Kinetic energy and temperature refer to motion relative to COM. The raw
mechanical pressure computed from this kinetic energy excludes COM motion.
To reconstruct the unconstrained ensemble's mean pressure, pass
``kinetic_energy + d * kT / 2`` to ``quantity.pressure``, where ``kT`` is the
target temperature. The volume corrections above do not restore the missing
COM kinetic energy to the particle momenta themselves.

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
.. autofunction:: npt_langevin
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
.. autoclass:: NPTLangevinState
