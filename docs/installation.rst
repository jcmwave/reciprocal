Installation
============

Install the core package from PyPI:

.. code-block:: console

   python -m pip install reciprocal

Plotting is optional. Install the ``plot`` extra to use
:class:`reciprocal.canvas.Canvas`:

.. code-block:: console

   python -m pip install "reciprocal[plot]"

Development checkout
--------------------

From the repository root, install the package and documentation dependencies,
then build the HTML site:

.. code-block:: console

   python -m pip install -e ".[docs]"
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. The plotting examples are
executed as part of the build, so warnings-as-errors also validates their
imports and API calls.

The project supports Python 3.10 and later. Matplotlib is not imported by the
top-level :mod:`reciprocal` package, so a core-only installation remains
usable without plotting dependencies.
