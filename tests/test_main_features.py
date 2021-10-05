# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

from vqf import VQF


def test_dummy():
    vqf = VQF()
    out = vqf.dummy()

    assert out == 42
