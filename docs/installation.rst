Installation
============

JAX MD requires Python 3.10 or newer.

With pip
--------

Install the latest release from `PyPI <https://pypi.org/project/jax-md/>`_:

.. code-block:: sh

   python -m pip install jax-md --upgrade

This installs the CPU build of JAX. To run on a GPU or TPU, install the
matching JAX build by following the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.

The ``amber`` extra installs OpenMM, OpenMM Force Fields and ParmEd, which the
loaders in ``jax_md.mm_forcefields.io.openmm`` need:

.. code-block:: sh

   python -m pip install "jax-md[amber]"

From source
-----------

We recommend `uv <https://docs.astral.sh/uv/>`_ for local development:

.. code-block:: sh

   git clone https://github.com/jax-md/jax-md
   cd jax-md
   uv sync --no-default-groups

Dependency groups add optional tools. ``testing`` and ``docs`` install the test
and documentation dependencies, ``mace`` installs MACE-JAX for
``energy.mace_neighbor_list`` (Python 3.11 or newer, JAX below 0.11), and
``uma`` installs PyTorch and Hugging Face Hub for loading pretrained UMA
checkpoints:

.. code-block:: sh

   uv sync --no-default-groups --group testing --group docs

With pip, install the source checkout in editable mode. The ``--group`` flag
needs pip 25.1 or newer:

.. code-block:: sh

   python -m pip install -e .
   python -m pip install -e . --group testing --group docs

See `CONTRIBUTING.md <https://github.com/jax-md/jax-md/blob/main/CONTRIBUTING.md>`_
for the full development setup and how to run the tests.

64-bit precision
----------------

JAX uses 32-bit floats by default. To enable 64-bit precision, set the flag
before importing ``jax_md``:

.. code-block:: python

   import jax
   jax.config.update('jax_enable_x64', True)

Setting the environment variable ``JAX_ENABLE_X64=1`` has the same effect. The
unit systems in ``jax_md.units`` are built in the active precision, so enable
64-bit precision before creating them.
