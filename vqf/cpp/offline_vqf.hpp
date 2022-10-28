// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

#ifndef OFFLINE_VQF_HPP
#define OFFLINE_VQF_HPP

#include "vqf.hpp"

#include <stddef.h>

/**
 * @brief A Versatile Quaternion-based Filter for IMU Orientation Estimation.
 *
 * \rst
 * This function implements the offline variant of orientation estimation filter described in the following publication:
 *
 *
 *     D. Laidig and T. Seel. "VQF: Highly Accurate IMU Orientation Estimation with Bias Estimation and Magnetic
 *     Disturbance Rejection." Information Fusion 2023, 91, 187--204.
 *     `doi:10.1016/j.inffus.2022.10.014 <https://doi.org/10.1016/j.inffus.2022.10.014>`_.
 *     [Accepted manuscript available at `arXiv:2203.17024 <https://arxiv.org/abs/2203.17024>`_.]
 *
 * The filter can perform simultaneous 6D (magnetometer-free) and 9D (gyr+acc+mag) sensor fusion and can also be used
 * without magnetometer data. It performs rest detection, gyroscope bias estimation during rest and motion, and magnetic
 * disturbance detection and rejection.
 *
 * To use this offline implementation, all data must have the same sampling rate and be available in (row-major) data
 * buffers. Similar to VQF::updateBatch(), the output pointer arguments must be null pointers or point to sufficiently
 * large data buffers.
 *
 * This function is the C++ implementation of the offline variant. The offline variant makes use of the fact that the
 * whole data is available and performs acausal signal processing, i.e. future samples are used to calculate the current
 * estimate. Depending on use case and programming language of choice, the following alternatives might be useful:
 *
 * +------------------------+--------------------------+--------------------------+--------------------------------+
 * |                        | Full Version             | Basic Version            | Offline Version                |
 * |                        |                          |                          |                                |
 * +========================+==========================+==========================+================================+
 * | **C++**                | :cpp:class:`VQF`         | :cpp:class:`BasicVQF`    | **offlineVQF (this function)** |
 * +------------------------+--------------------------+--------------------------+--------------------------------+
 * | **Python/C++ (fast)**  | :py:class:`vqf.VQF`      | :py:class:`vqf.BasicVQF` | :py:meth:`vqf.offlineVQF`      |
 * +------------------------+--------------------------+--------------------------+--------------------------------+
 * | **Pure Python (slow)** | :py:class:`vqf.PyVQF`    | --                       | --                             |
 * +------------------------+--------------------------+--------------------------+--------------------------------+
 * | **Pure Matlab (slow)** | :mat:class:`VQF.m <VQF>` | --                       | --                             |
 * +------------------------+--------------------------+--------------------------+--------------------------------+
 * \endrst
 *
 * @param gyr gyroscope measurement in rad/s (N*3 elements, must not be null)
 * @param acc accelerometer measurement in m/s² (N*3 elements, must not be null)
 * @param mag magnetometer measurement in arbitrary units (N*3 elements, can be a null pointer)
 * @param N number of samples
 * @param Ts sampling time in seconds
 * @param params parameters (pass `VQFParams()` to use the default parameters and
 *     see VQFParams for a full list and detailed descriptions)
 * @param out6D output buffer for the 6D quaternion (N*4 elements, can be a null pointer)
 * @param out9D output buffer for the 9D quaternion (N*4 elements, can be a null pointer)
 * @param outDelta output buffer for heading difference angle (N elements, can be a null pointer)
 * @param outBias output buffer for the gyroscope bias estimate (N*3 elements, can be a null pointer)
 * @param outBiasSigma output buffer for the bias estimation uncertainty (N elements, can be a null pointer)
 * @param outRest output buffer for the rest detection state (N elements, can be a null pointer)
 * @param outMagDist output buffer for the magnetic disturbance state (N elements, can be a null pointer)
 */
void offlineVQF(const vqf_real_t gyr[], const vqf_real_t acc[], const vqf_real_t mag[],
                size_t N, vqf_real_t Ts, const VQFParams &params,
                vqf_real_t out6D[], vqf_real_t out9D[], vqf_real_t outDelta[],
                vqf_real_t outBias[], vqf_real_t outBiasSigma[], bool outRest[], bool outMagDist[]);

#endif // OFFLINE_VQF_HPP
