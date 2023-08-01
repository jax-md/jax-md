Neural Network Primitives
==========================

.. default-role:: code
.. automodule:: jax_md.nn

Behler-Parrinello Networks
---------------------------

JAX MD contains neural network primitives for a common class of 
fixed-feature neural network architectures known as Behler-Parrinello  
Neural Networks (BP-NN) [#behler07]_ [#nongnuch13]_. An energy function 
using this architecture can be found in `energy.py`.

The BP-NN architecture uses a relatively simple,
fully connected neural network to predict the local energy for each atom. Then
the total energy is the sum of local energies due to each atom. Atoms of the
same type use the same NN to predict energy.
Each atomic NN is applied to hand-crafted features called symmetry functions.
There are two kinds of symmetry functions: radial and angular. Radial symmetry
functions represent information about two-body interactions of the central
atom, whereas angular symmetry functions represent information about three-body
interactions. Below we implement radial and angular symmetry functions for
arbitrary number of types of atoms (Note that most applications of BP-NN limit
their systems to 1 to 3 types of atoms). We also present a convenience wrapper
that returns radial and angular symmetry functions with symmetry function
parameters that should work reasonably for most systems (the symmetry functions
are taken from reference [2]). Please see references [1, 2] for details about
how the BP-NN works.

.. rubric:: References

.. [#behler07] Behler, Jörg, and Michele Parrinello. "Generalized neural-network representation of high-dimensional potential-energy surfaces." Physical Review Letters 98.14 (2007): 146401.

.. [#nongnuch13] Artrith, Nongnuch, Björn Hiller, and Jörg Behler. "Neural network potentials for metals and oxides–First applications to copper clusters at zinc oxide." Physica Status Solidi (b) 250.6 (2013): 1191-1203.

.. currentmodule:: jax_md._nn.behler_parrinello
.. autofunction:: radial_symmetry_functions
.. autofunction:: radial_symmetry_functions_neighbor_list

.. autofunction:: angular_symmetry_functions
.. autofunction:: angular_symmetry_functions_neighbor_list

Graph Neural Networks
----------------------
.. currentmodule:: jax_md.nn

JAX MD also contains primitives for constructing graph neural networks. These 
primitives are based on (and are one-to-one with) the excellent Jraph library
(www.github.com/deepmind/jraph). Compared to Jraph, these primitives are adapted 
to work with Dense neighbor lists. However, it is also possible to use Jraph's
primitives directly in combination with Sparse neighbor lists. 

Our implementation here is based off the outstanding GraphNets library by
DeepMind at, www.github.com/deepmind/graph_nets. This implementation was also
heavily influenced by work done by Thomas Keck. We implement a subset of the
functionality from the graph nets library to be compatible with jax-md
states and neighbor lists, end-to-end jit compilation, and easy batching.
Graphs are described by node states, edge states, a global state, and
outgoing / incoming edges.

We provide two components:

  1) A GraphIndependent layer that applies a neural network separately to the
     node states, the edge states, and the globals. This is often used as an
     encoding or decoding step.
     
  2) A GraphNetwork layer that transforms the nodes, edges, and globals using
     neural networks following Battaglia et al. (). Here, we use
     sum-message-aggregation.

.. autoclass:: GraphMapFeatures
.. autoclass:: GraphNetwork
.. autoclass:: GraphsTuple
