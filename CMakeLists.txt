# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.0)

project(vqf
    VERSION 2.0.1
    DESCRIPTION "A Versatile Quaternion-based Filter for IMU Orientation Estimation"
    HOMEPAGE_URL "https://vqf.readthedocs.io"
    LANGUAGES CXX)

option(VQF_SINGLE_PRECISION "Use single precision floats")
option(VQF_NO_MOTION_BIAS_ESTIMATION "Disable motion bias estimation")

add_subdirectory(vqf/cpp)