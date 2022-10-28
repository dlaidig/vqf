.. SPDX-FileCopyrightText: 2022 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

.. module:: vqf
   :noindex:
.. mat:module:: matlab
    :noindex:

FAQ / Troubleshooting
=====================

The output of VQF does not make sense.
--------------------------------------

Check the following:

- Is the sampling time (the time between consecutive samples in seconds, i.e., the inverse of the sampling frequency)
  specified correctly?
- Is the input data provided in the correct units (rad/s for gyr, m/s² for acc; the units of mag do not matter)?
- Could the input data (e.g., acc and gyr) accidentally be swapped?
- VQF always outputs quaternions in scalar-first format, i.e., [w x y z]. Is it possible that your code expects the data
  to be in [x y z w] format?
- Is it possible that the reference coordinate system does not match the expectations? (see next question)
- Is it possible that the output quaternion needs to be inverted to match the expectations?

The reference frame that I am using does not follow the ENU convention.
-----------------------------------------------------------------------

VQF always outputs quaternions that specify the orientation of the sensor coordinate system :math:`\mathcal{S}` relative
to an east-north-up (ENU) reference frame :math:`\mathcal{E}=\mathcal{E}_\mathrm{ENU}`. In order to obtain output
quaternions that follow a different convention, you can multiply a fixed quaternion to the left side.

For example, in order to obtain quaternions that use the north-east-down (NED) convention, use a rotation of 180° around
[1 1 0], i.e.,
:math:`^{\mathcal{E}_\mathrm{ENU}}_{\mathcal{E}_\mathrm{NED}}\mathbf{q} = \left[0 ~ \frac{1}{\sqrt{2}} ~ \frac{1}{\sqrt{2}} ~ 0\right]^T`:

:math:`^{\mathcal{S}}_{\mathcal{E}_\mathrm{NED}}\mathbf{q} = ^{\mathcal{E}_\mathrm{ENU}}_{\mathcal{E}_\mathrm{NED}}\mathbf{q} \otimes ^{\mathcal{S}}_{\mathcal{E}_\mathrm{ENU}}\mathbf{q} = \left[0 ~ \frac{1}{\sqrt{2}} ~ \frac{1}{\sqrt{2}} ~ 0\right]^T \otimes ^{\mathcal{S}}_{\mathcal{E}_\mathrm{ENU}}\mathbf{q}.`

For a north-west-up reference frame with the x-axis pointing north and the z-axis pointing up, use a rotation of 90°
around [0 0 1]:

:math:`^{\mathcal{S}}_{\mathcal{E}_\mathrm{NWU}}\mathbf{q} = ^{\mathcal{E}_\mathrm{ENU}}_{\mathcal{E}_\mathrm{NWU}}\mathbf{q} \otimes ^{\mathcal{S}}_{\mathcal{E}_\mathrm{ENU}}\mathbf{q} = \left[\frac{1}{\sqrt{2}} ~ 0 ~ 0 ~ \frac{1}{\sqrt{2}}\right]^T \otimes ^{\mathcal{S}}_{\mathcal{E}_\mathrm{ENU}}\mathbf{q}.`

If you are using Matlab's ``ahrsfilter``/``imufilter``, check out :ref:`ref_faq_ahrsfilter`.

The coordinate system of the IMU that I am using does not match what I want.
----------------------------------------------------------------------------

VQF always outputs quaternions that specify the orientation of the sensor coordinate system :math:`\mathcal{S}` relative
to an east-north-up (ENU) reference frame :math:`\mathcal{E}`.

If, instead, you want to obtain the orientation of a body coordinate system :math:`\mathcal{B}` to which the IMU is
rigidly attached, multiply the fixed correction quaternion representing the orientation of the body coordinate system
:math:`\mathcal{B}` relative to the sensor coordinate system :math:`\mathcal{S}` to the right side of the IMU
orientation :math:`^{\mathcal{S}}_{\mathcal{E}}\mathbf{q}` returned by VQF:

:math:`^{\mathcal{B}}_{\mathcal{E}}\mathbf{q} = ^{\mathcal{S}}_{\mathcal{E}}\mathbf{q} \otimes ^{\mathcal{B}}_{\mathcal{S}}\mathbf{q}.`

Another option is to just adjust the input data by swapping and inverting axes. This is not recommended in most cases
and care should be taken in order to make sure that the new input data is still provided in a right-handed coordinate
system.

I am getting a ``ValueError: ndarray is not C-contiguous`` message.
-------------------------------------------------------------------

The Cython functions expect the input arrays to be in C-contiguous memory order. If this is not the case, for example,
due to transposing or array indexing (or just loading the data with a library that returns non-contiguous arrays), you
can easily fix this with
`np.ascontiguousarray <https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html>`_.

(If you process the same data multiple times and care about speed, you will want to only call ``np.ascontiguousarray``
once after loading the data.)

.. _ref_faq_ahrsfilter:

How to obtain quaternions compatible with Matlab's ``ahrsfilter``/``imufilter``?
--------------------------------------------------------------------------------

Use the following code to obtain compatible quaternions with ``'ReferenceFrame', 'NED'`` (the default setting):

.. code-block:: matlab

    f = ahrsfilter('SampleRate', sample_rate, 'ReferenceFrame', 'NED');
    quat_ahrsfilter = f(acc, gyr, mag);

    vqf = VQF(1/sample_rate);
    out = vqf.updateBatch(gyr, acc, mag);
    quat_vqf = quaternion([1/sqrt(2) 0 0 -1/sqrt(2)]) * quaternion(out.quat9D);

If you use ``'ReferenceFrame', 'ENU'``:

.. code-block:: matlab

    f = ahrsfilter('SampleRate', sample_rate, 'ReferenceFrame', 'ENU');
    quat_ahrsfilter = f(acc, gyr, mag);

    vqf = VQF(1/sample_rate);
    out = vqf.updateBatch(gyr, acc, mag);
    quat_vqf = quaternion([0 0 1 0]) * quaternion(out.quat9D);

(The correction quaternions were derived by comparing the outputs since it was not immediately clear how Matlab
defines the reference frames. If you have a clear explanation, let us know.)
