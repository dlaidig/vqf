# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import pytest

from vqf import VQF


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_dummy(cls):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls()
    vqf_ref = VQF()

    assert vqf.dummy() == vqf_ref.dummy()
