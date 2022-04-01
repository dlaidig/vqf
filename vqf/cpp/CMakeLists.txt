# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.0)

project (vqf)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -ansi -std=c++98")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(vqf_dummy
    vqf.cpp
    basicvqf.cpp
    offline_vqf.cpp
    dummy_main.cpp
)

add_executable(vqf_dummy_no_motion_bias_est
    vqf.cpp
    basicvqf.cpp
    dummy_main.cpp
)
target_compile_definitions(vqf_dummy_no_motion_bias_est PUBLIC
    VQF_NO_MOTION_BIAS_ESTIMATION=1
)

add_executable(vqf_dummy_single_precision
    vqf.cpp
    basicvqf.cpp
    offline_vqf.cpp
    dummy_main.cpp
)
target_compile_definitions(vqf_dummy_single_precision PUBLIC
    VQF_SINGLE_PRECISION=1
)

add_executable(vqf_dummy_no_motion_bias_est_single_precision
    vqf.cpp
    basicvqf.cpp
    dummy_main.cpp
)
target_compile_definitions(vqf_dummy_no_motion_bias_est_single_precision PUBLIC
    VQF_SINGLE_PRECISION=1
    VQF_NO_MOTION_BIAS_ESTIMATION=1
)

add_library(vqf SHARED
    vqf.cpp
    basicvqf.cpp
    offline_vqf.cpp
)

set_target_properties(vqf PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
