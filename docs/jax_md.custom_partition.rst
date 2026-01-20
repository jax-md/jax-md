Multi-Image Neighbor Lists
===========================

.. default-role:: code
.. automodule:: jax_md.custom_partition

This module provides neighbor list construction for small periodic boxes where
the cutoff radius exceeds half the box size (:math:`r_\text{cut} > L/2`). In such
cases, the standard minimum image convention (MIC) fails, and particles may
interact with multiple periodic images of their neighbors.

Neighbor List Construction
---------------------------
.. autofunction:: neighbor_list_multi_image
.. autofunction:: neighbor_list_multi_image_mask

Data Structures
----------------
.. autoclass:: NeighborListMultiImage
   :members:
   :undoc-members:

.. autoclass:: NeighborListMultiImageFns
   :members:

Capacity Estimation
--------------------
.. autofunction:: estimate_max_neighbors
.. autofunction:: estimate_max_neighbors_from_box

Graph Neural Network Support
-----------------------------
.. autofunction:: graph_featurizer
