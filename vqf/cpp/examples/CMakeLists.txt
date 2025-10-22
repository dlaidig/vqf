# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wdouble-promotion -pedantic -ansi -std=c++98")
endif()

include_directories(..)
set(VQF_SOURCE_DIR ..)

add_executable(vqf_dummy
    ${VQF_SOURCE_DIR}/vqf.cpp
    ${VQF_SOURCE_DIR}/basicvqf.cpp
    ${VQF_SOURCE_DIR}/offline_vqf.cpp
    dummy_main.cpp
)

add_executable(vqf_dummy_no_motion_bias_est
    ${VQF_SOURCE_DIR}/vqf.cpp
    ${VQF_SOURCE_DIR}/basicvqf.cpp
    dummy_main.cpp
)
target_compile_definitions(vqf_dummy_no_motion_bias_est PUBLIC
    VQF_NO_MOTION_BIAS_ESTIMATION=1
)

add_executable(vqf_dummy_single_precision
    ${VQF_SOURCE_DIR}/vqf.cpp
    ${VQF_SOURCE_DIR}/basicvqf.cpp
    ${VQF_SOURCE_DIR}/offline_vqf.cpp
    dummy_main.cpp
)
target_compile_definitions(vqf_dummy_single_precision PUBLIC
    VQF_SINGLE_PRECISION=1
)

add_executable(vqf_dummy_no_motion_bias_est_single_precision
    ${VQF_SOURCE_DIR}/vqf.cpp
    ${VQF_SOURCE_DIR}/basicvqf.cpp
    dummy_main.cpp
)
target_compile_definitions(vqf_dummy_no_motion_bias_est_single_precision PUBLIC
    VQF_SINGLE_PRECISION=1
    VQF_NO_MOTION_BIAS_ESTIMATION=1
)
