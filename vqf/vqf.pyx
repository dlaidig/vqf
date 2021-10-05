# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

# distutils: language = c++
# cython: language_level=3
# cython: embedsignature=True
# distutils: sources = vqf/cpp/vqf.cpp
# distutils: undef_macros = NDEBUG

import numpy as np

from libcpp cimport bool
from libc.string cimport memcpy
cimport numpy as np
cimport cython

ctypedef double vqf_real_t
vqf_real = np.double

cdef extern from 'cpp/vqf.hpp':
    cdef cppclass C_VQF 'VQF':
        C_VQF() except +

        int dummy()


cdef class VQF:
    """Dummy Cython wrapper class

    This is a dummy class for testing. Real code coming soon.
    """
    cdef C_VQF* c_obj

    def __cinit__(self):
        print('VQF Cython constructor called')
        self.c_obj = new C_VQF()

    def __dealloc__(self):
        del self.c_obj

    def dummy(self):
        return self.c_obj.dummy()
