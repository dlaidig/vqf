.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

.. module:: vqf
   :noindex:
.. mat:module:: matlab
    :noindex:

=======================================================================
VQF: A Versatile Quaternion-based Filter for IMU Orientation Estimation
=======================================================================

Introduction
============

This is the documentation for the implementation of the IMU orientation estimation filter described in the following
publication:

    D. Laidig and T. Seel. "VQF: Highly Accurate IMU Orientation Estimation with Bias Estimation and Magnetic
    Disturbance Rejection." Information Fusion 2023, 91, 187--204.
    `doi:10.1016/j.inffus.2022.10.014 <https://doi.org/10.1016/j.inffus.2022.10.014>`_.
    [Accepted manuscript available at `arXiv:2203.17024 <https://arxiv.org/abs/2203.17024>`_.]


The filter can perform simultaneous 6D (magnetometer-free) and 9D (gyr+acc+mag) sensor fusion and can also be used
without magnetometer data. Different sampling rates for gyroscopes, accelerometers, and magnetometers are
supported as well. While in most cases, the defaults will be reasonable, the algorithm can be influenced via two main
tuning parameters.

The source code can be found at https://github.com/dlaidig/vqf.

Algorithm Variants
------------------

There are three variants of the algorithm:

- The **full version** includes online gyroscope bias estimation and magnetic disturbance rejection.
- The **basic version** does not include those features and only has two tuning parameters. (Note that the full version
  is equivalent when the parameters :cpp:member:`motionBiasEstEnabled <VQFParams::motionBiasEstEnabled>`,
  :cpp:member:`restBiasEstEnabled <VQFParams::restBiasEstEnabled>` and
  :cpp:member:`magDistRejectionEnabled <VQFParams::magDistRejectionEnabled>` are set to false.)
- The **offline version** makes use of acausal signal processing in order to improve the accuracy when the full time
  series of measurement data is available at the time of processing.

A number of implementations in different programming languages is available:

- The main implementation is written in **C++**.
- Cython-based wrappers are provided that allow the fast C++ implementation to be used from **Python**.
- Additionally, there is a (comparatively slow) implementation in **pure Python**.
- A **pure Matlab** version is available as well.

See the following table for an overview of the implementations:

+------------------------+--------------------------+--------------------------+---------------------------+
|                        | Full Version             | Basic Version            | Offline Version           |
|                        |                          |                          |                           |
+========================+==========================+==========================+===========================+
| **C++**                | :cpp:class:`VQF`         | :cpp:class:`BasicVQF`    | :cpp:func:`offlineVQF`    |
+------------------------+--------------------------+--------------------------+---------------------------+
| **Python/C++ (fast)**  | :py:class:`vqf.VQF`      | :py:class:`vqf.BasicVQF` | :py:meth:`vqf.offlineVQF` |
+------------------------+--------------------------+--------------------------+---------------------------+
| **Pure Python (slow)** | :py:class:`vqf.PyVQF`    | --                       | --                        |
+------------------------+--------------------------+--------------------------+---------------------------+
| **Pure Matlab (slow)** | :mat:class:`VQF.m <VQF>` | --                       | --                        |
+------------------------+--------------------------+--------------------------+---------------------------+

VQF is licensed under the terms of the `MIT license <https://spdx.org/licenses/MIT.html>`_.

Documentation
=============

.. toctree::
   :maxdepth: 2

   installation
   getting_started
   faq

.. toctree::
   :maxdepth: 2
   :caption: C++ Reference:

   ref_cpp
   ref_cpp_params
   ref_cpp_state_coeffs
   ref_cpp_basic
   ref_cpp_offline

.. toctree::
   :maxdepth: 2
   :caption: Python Reference:

   ref_cython
   ref_cython_basic
   ref_cython_offline
   ref_python

.. toctree::
   :maxdepth: 2
   :caption: Matlab Reference:

   ref_matlab


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
