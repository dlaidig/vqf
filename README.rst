.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

VQF: A Versatile Quaternion-based Filter for IMU Orientation Estimation
=======================================================================

|tests| |build| |docs| |version| |python| |format| |license| |downloads|

This is the implementation of the IMU orientation estimation filter described in the following publication:

    D. Laidig and T. Seel. "VQF: Highly Accurate IMU Orientation Estimation with Bias Estimation and Magnetic
    Disturbance Rejection." Information Fusion 2023, 91, 187--204.
    `doi:10.1016/j.inffus.2022.10.014 <https://doi.org/10.1016/j.inffus.2022.10.014>`_.
    [Accepted manuscript available at `arXiv:2203.17024 <https://arxiv.org/abs/2203.17024>`_.]

The filter can perform simultaneous 6D (magnetometer-free) and 9D (gyr+acc+mag) sensor fusion and can also be used
without magnetometer data. Different sampling rates for gyroscopes, accelerometers, and magnetometers are
supported as well. While in most cases, the defaults will be reasonable, the algorithm can be influenced via two
tuning parameters.

Documentation
-------------

Detailed documentation can be found at https://vqf.readthedocs.io/.

Installation
------------

The VQF Python package can easily be installed from PyPI via pip, e.g.:

.. code-block:: sh

    pip install vqf

For more information, please refer to the `documentation <https://vqf.readthedocs.io/>`__.

License
-------

VQF is licensed under the terms of the `MIT license <https://spdx.org/licenses/MIT.html>`__.

Contact
-------

Daniel Laidig <laidig at control.tu-berlin.de>


.. |tests| image:: https://img.shields.io/github/workflow/status/dlaidig/vqf/Tests?label=tests
    :target: https://github.com/dlaidig/vqf/actions?query=workflow%3ATests
.. |build| image:: https://img.shields.io/github/workflow/status/dlaidig/vqf/Build
    :target: https://github.com/dlaidig/vqf/actions?query=workflow%3ABuild
.. |docs| image:: https://img.shields.io/readthedocs/vqf
    :target: https://vqf.readthedocs.io/
.. |version| image:: https://img.shields.io/pypi/v/vqf
    :target: https://pypi.org/project/vqf/
.. |python| image:: https://img.shields.io/pypi/pyversions/vqf
    :target: https://pypi.org/project/vqf/
.. |format| image:: https://img.shields.io/pypi/format/vqf
    :target: https://pypi.org/project/vqf/
.. |license| image:: https://img.shields.io/pypi/l/vqf
    :target: https://github.com/dlaidig/vqf_playground/blob/master/LICENSES/MIT.txt
.. |downloads| image:: https://img.shields.io/pypi/dm/vqf
    :target: https://pypi.org/project/vqf/
