# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from vqf import VQF

params = [
    dict(motionBiasEstEnabled=False, restBiasEstEnabled=False, magDistRejectionEnabled=False),
    dict(motionBiasEstEnabled=True, restBiasEstEnabled=False, magDistRejectionEnabled=False),
    dict(motionBiasEstEnabled=False, restBiasEstEnabled=True, magDistRejectionEnabled=False),
    dict(motionBiasEstEnabled=False, restBiasEstEnabled=False, magDistRejectionEnabled=True),
    dict(motionBiasEstEnabled=True, restBiasEstEnabled=True, magDistRejectionEnabled=False),
    dict(motionBiasEstEnabled=True, restBiasEstEnabled=False, magDistRejectionEnabled=True),
    dict(motionBiasEstEnabled=False, restBiasEstEnabled=True, magDistRejectionEnabled=True),
    dict(motionBiasEstEnabled=True, restBiasEstEnabled=True, magDistRejectionEnabled=True),
]


@pytest.mark.parametrize('params', params)
def test_rest_detection(imu_data, params):
    """
    Test to verify that the rest detection is roughly doing the right thing if enabled.
    """
    vqf = VQF(1/imu_data.sampling_rate, **params)
    out = vqf.updateBatchFullState(imu_data.gyr, imu_data.acc, imu_data.mag)

    if params['restBiasEstEnabled']:
        np.testing.assert_almost_equal(out['restDetected'][500:2000], 1)  # initial rest
        np.testing.assert_almost_equal(out['restDetected'][3000:11000], 0)  # motion phase
        np.testing.assert_almost_equal(out['restDetected'][13000:], 1)  # rest at end
    else:
        np.testing.assert_almost_equal(out['restDetected'], 0)  # should not detect rest if disabled


@pytest.mark.parametrize('params', params)
def test_mag_dist_detection(imu_data, params):
    """
    Test to verify that the magnetic field disturbance detection will (after some time) detect that the
    field is homogeneous (if enabled).
    """
    vqf = VQF(1/imu_data.sampling_rate, **params)
    out = vqf.updateBatchFullState(imu_data.gyr, imu_data.acc, imu_data.mag)

    if params['magDistRejectionEnabled']:
        np.testing.assert_almost_equal(out['magDistDetected'][:100], 1)  # start with disturbed
        np.testing.assert_almost_equal(out['magDistDetected'][6000:], 0)  # detect that mag is homogeneous
    else:
        np.testing.assert_almost_equal(out['magDistDetected'], 1)  # disabled, initial state should not change
