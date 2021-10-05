# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

def get_cpp_path():
    """Returns the path to the directory containing the C++ source code."""
    import pkg_resources
    return pkg_resources.resource_filename('vqf', 'cpp/')


def get_matlab_path():
    """Returns the path to the directory containing the Matlab source code."""
    import pkg_resources
    return pkg_resources.resource_filename('vqf', 'matlab/')
