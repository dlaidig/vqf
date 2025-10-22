# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

set(VQF_SOURCES basicvqf.cpp vqf.cpp)

if(NOT VQF_NO_MOTION_BIAS_ESTIMATION)
    list(APPEND VQF_SOURCES offline_vqf.cpp)
endif()

add_library(vqf ${VQF_SOURCES})

target_include_directories(vqf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(VQF_SINGLE_PRECISION)
    target_compile_definitions(vqf PUBLIC VQF_SINGLE_PRECISION)
endif()

if(VQF_NO_MOTION_BIAS_ESTIMATION)
    target_compile_definitions(vqf PUBLIC VQF_NO_MOTION_BIAS_ESTIMATION)
endif()
