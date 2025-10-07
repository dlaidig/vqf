# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(vqf
    vqf.cpp
    basicvqf.cpp
    offline_vqf.cpp
)

if(VQF_SINGLE_PRECISION)
    target_compile_definitions(vqf PUBLIC VQF_SINGLE_PRECISION)
endif()

if(VQF_NO_MOTION_BIAS_ESTIMATION)
    target_compile_definitions(vqf PUBLIC VQF_NO_MOTION_BIAS_ESTIMATION)
endif()

set_target_properties(vqf PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
