.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

.. module:: vqf
   :noindex:
.. mat:module:: matlab
    :noindex:

Getting Started
===============

The aim of the this section is to provide runnable minimum working examples that show how the orientation
estimation filter can be used and that can serve as a starting point for real use cases.

To create a simple test case that does not require access to real IMU measurements, simple dummy data is used that
consists of a constant gyroscope measurement of 1 °/s in all axes and a constant accelerometer measurement
with norm 9.81 m/s² in direction ``[1 1 1]``. To keep it as simple as possible, magnetometers are not used.
This data is fed to the algorithm---once to the full version and once to the basic version---and the resulting
orientation quaternion is plotted.

It is worth noticing that this example already nicely demonstates the effect of gyroscope bias estimation. After a short
time, the full version estimates and removes the gyroscope bias which leads to a constant quaternion. In contrast, the
quaternion estimated by the basic version always changes (mostly due to rotation around the vertical axis because
magnetometer measurements are not used).

Python
------

.. plot::
   :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from vqf import VQF, BasicVQF, PyVQF

    # generate simple dummy data
    gyr = np.deg2rad(1)*np.ones((6000, 3))  # in rad/s
    acc = 9.81/np.sqrt(3)*np.ones((6000, 3))  # in m/s²
    Ts = 0.01  # sampling time (100 Hz)

    # run orientation estimation
    vqf = VQF(Ts)
    # alternative: vqf = PyVQF(Ts)
    out = vqf.updateBatch(gyr, acc)

    # plot the quaternion
    plt.figure()
    plt.subplot(211)
    plt.plot(out['quat6D'])
    plt.title('quaternion (full version)')
    plt.grid()

    # run the basic version with the same data
    params = dict(
        motionBiasEstEnabled=False,
        restBiasEstEnabled=False,
        magDistRejectionEnabled=False,
    )
    vqf2 = VQF(Ts, **params)
    # alternative: vqf2 = BasicVQF(Ts)
    # alternative: vqf2 = PyVQF(Ts, **params)
    out2 = vqf2.updateBatch(gyr, acc)

    # plot quaternion (notice the difference due to disabled bias estimation)
    plt.subplot(212)
    plt.plot(out2['quat6D'])
    plt.grid()
    plt.title('quaternion (basic version)')
    plt.tight_layout()
    plt.show()

C++
---

The following small C++ program calculates the same values as the Python example and outputs a csv file:

.. code-block:: cpp

    #include "vqf.hpp"

    #include <iostream>

    void run(bool basic) {
        vqf_real_t gyr[3] = {0.01745329, 0.01745329, 0.01745329}; // 1 °/s in rad/s
        vqf_real_t acc[3] = {5.663806, 5.663806, 5.663806}; // 9.81/sqrt(3) m/s²
        vqf_real_t quat[4] = {0, 0, 0, 0}; // output array for quaternion

        VQFParams params;

        if (basic) { // alternative: use the BasicVQF class
            params.restBiasEstEnabled = false;
            params.motionBiasEstEnabled = false;
            params.magDistRejectionEnabled = false;
        }

        VQF vqf(params, 0.01);
        for (size_t i=0; i < 6000; i++) {
            vqf.update(gyr, acc);
            vqf.getQuat6D(quat);
            std::cout << int(basic) << ", " << i << ", "<< quat[0] << ", " << quat[1]
                      << ", "<< quat[2] << ", " << quat[3] << std::endl;
        }
    }

    int main()
    {
        run(false);
        run(true);
        return 0;
    }

Matlab
------

The following small Matlab script calculates the same values as the Python example and shows an equivalent plot:

.. code-block:: matlab

    %% generate simple dummy data
    gyr = deg2rad(1)*ones(5000, 3);  % in rad/s
    acc = 9.81/sqrt(3)*ones(5000, 3); % in m/s²
    Ts = 0.01; % sampling time (100 Hz)

    %% run orientation estimation
    vqf = VQF(Ts);
    out = vqf.updateBatch(gyr, acc);

    %% plot the quaternion
    figure();
    subplot(211);
    plot(out.quat6D);
    title('quaternion (full version)');
    grid();

    %% run the basic version with the same data
    params = struct();
    params.motionBiasEstEnabled = false;
    params.restBiasEstEnabled = false;
    params.magDistRejectionEnabled = false;
    vqf2 = VQF(Ts, params);
    out2 = vqf2.updateBatch(gyr, acc);

    %% plot quaternion (notice the difference due to disabled bias estimation)
    subplot(212);
    plot(out2.quat6D);
    grid();
    title('quaternion (basic version)');
