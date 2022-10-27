.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

.. module:: vqf
   :noindex:
.. mat:module:: matlab
    :noindex:

Installation
============

Python Package
--------------

The VQF Python package can easily be installed from PyPI via pip, e.g.:

.. code-block:: sh

    pip install vqf

(Depending on your Python installation, it might be necessary to use ``pip3`` instead of ``pip`` and/or to add the
``--user`` option.)

C++ Implementation
------------------

In order to use the C++ implementation in your own project, simply add the two files ``vqf.hpp`` and ``vqf.cpp`` to your
project.

The files are located in the directory ``vqf/cpp/`` of the repository. They are also shipped when installing the
Python package. In order to find out the installation path, use the following command:

.. code-block:: sh

    python -c "import vqf; print(vqf.get_cpp_path())"

The basic implementation is also self-contained and consists of the two files ``basicvqf.hpp`` and ``basicvqf.cpp``.
In order to compile the offline version, the files ``offline_vqf.hpp``, ``offline_vqf.cpp``. ``vqf.hpp``, and
``vqf.cpp`` are needed.

Matlab Implementation
---------------------

The Matlab implementation is self-contained in the file ``VQF.m``. In order to use this implementation, add the
directory containing the file to your Matlab path or copy this file to your own project.

The file is located in the directory ``vqf/matlab/`` of the repository. It is also shipped when installing the
Python package. In order to find out the installation path, use the following command:

.. code-block:: sh

    python -c "import vqf; print(vqf.get_matlab_path())"

Development Notes
-----------------

The source code can be found at https://github.com/dlaidig/vqf.

To install the package from a clone of the git repository, use pip with a dot as the package name. With ``[dev]``,
additional dependencies for development will automatically be installed. Editable installs (``-e``) are also possible.
Therefore, use the following command to install the package for development:

.. code-block:: sh

    pip install --user -e ".[dev]"

To build the documentation:

.. code-block:: sh

    python3 setup.py docs -E

To run unit tests and coding style checks (optionally with ``--nomatlab`` and ``--nooctave`` to disable testing the
Matlab implementation):

.. code-block:: sh

    flake8 && pytest
    flake8 && pytest --nomatlab --nooctave

To test `RESUE <https://reuse.software/>`_ compliance:

.. code-block:: sh

    reuse lint

The source distributation and wheels of the Python package for various platforms and Python versions are automatically
built using `cibuildwheel <https://github.com/joerick/cibuildwheel>`_ via GitHub Actions
(see the file ``.github/workflows/build.yml`` in the repository). The resulting files are then uploaded to PyPI via
``twine upload``.
