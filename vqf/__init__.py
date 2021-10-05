# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

from .vqf import VQF
from .pyvqf import PyVQF
from .utils import get_cpp_path, get_matlab_path

import sys
print('vqf: This is a dummy repo for testing. Real code coming soon.', file=sys.stderr)

__all__ = ['VQF', 'PyVQF', 'get_cpp_path', 'get_matlab_path']
