# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wdouble-promotion -pedantic -ansi -std=c++98")

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

add_library(vqf
    vqf.cpp
    basicvqf.cpp
    offline_vqf.cpp
)

target_compile_definitions(vqf PUBLIC
    VQF_SINGLE_PRECISION=${VQF_SINGLE_PRECISION}
    VQF_NO_MOTION_BIAS_ESTIMATION=${VQF_NO_MOTION_BIAS_ESTIMATION}
)

set_target_properties(vqf PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
