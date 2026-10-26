Energy Minimization Routines
==============================

.. default-role:: code
.. automodule:: jax_md.minimize

Minimization Routines
----------------------

.. autofunction:: gradient_descent
.. autofunction:: fire_descent
.. autofunction:: fire_descent_box
.. autofunction:: precon_fire_descent
.. autofunction:: precon_fire_descent_box

Preconditioners
----------------

.. autofunction:: exp_preconditioner
.. autofunction:: c1_preconditioner
.. autofunction:: estimate_exp_mu

Data Types
-----------

.. autoclass:: FireDescentState
.. autoclass:: FireBoxDescentState
.. autoclass:: PreconFireDescentState
.. autoclass:: PreconFireBoxDescentState
