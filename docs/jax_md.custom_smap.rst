Multi-Image Structure Maps
===========================

.. default-role:: code
.. automodule:: jax_md.custom_smap

This module provides structure-mapped functions (analogous to :mod:`jax_md.smap`)
that work with :class:`~jax_md.custom_partition.NeighborListMultiImage` to correctly
handle small periodic boxes where :math:`r_\text{cut} > L/2`.

The key difference from standard ``smap`` functions is that these use explicit
lattice shifts stored in the neighbor list rather than relying on the minimum
image convention.

Pair Functions
---------------
.. autofunction:: pair_neighbor_list_multi_image
