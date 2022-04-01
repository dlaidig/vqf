# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from vqf import VQF, BasicVQF, offlineVQF


def assertBatchOutEqual(out, ref, basic=False, noMag=False):
    if not basic:
        np.testing.assert_almost_equal(out['bias'], ref['bias'])
        np.testing.assert_almost_equal(out['biasSigma'], ref['biasSigma'])
        np.testing.assert_almost_equal(out['restDetected'], ref['restDetected'])
        if not noMag:
            np.testing.assert_almost_equal(out['magDistDetected'], ref['magDistDetected'])
    np.testing.assert_almost_equal(out['quat6D'], ref['quat6D'])
    if not noMag:
        np.testing.assert_almost_equal(out['quat9D'], ref['quat9D'])
        np.testing.assert_almost_equal(out['delta'], ref['delta'])


@pytest.mark.parametrize('cls', ['PyVQF', 'BasicVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_updateGyr(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)

    for gyr in imu_data.gyr[2650:2750]:
        vqf.updateGyr(gyr)
        vqf_ref.updateGyr(gyr)
        np.testing.assert_almost_equal(vqf.getQuat3D(), vqf_ref.getQuat3D())


@pytest.mark.parametrize('cls', ['PyVQF', 'BasicVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_updateBatch_basic(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    args = dict(motionBiasEstEnabled=False, restBiasEstEnabled=False, magDistRejectionEnabled=False)
    if cls == BasicVQF:
        vqf = cls(1 / imu_data.sampling_rate)
    else:
        vqf = cls(1/imu_data.sampling_rate, **args)
    vqf_ref = VQF(1/imu_data.sampling_rate, **args)

    out = vqf.updateBatch(imu_data.gyr, imu_data.acc, imu_data.mag)
    ref = vqf_ref.updateBatch(imu_data.gyr, imu_data.acc, imu_data.mag)

    assertBatchOutEqual(out, ref, basic=(cls == BasicVQF))


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_updateBatch(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)

    out = vqf.updateBatch(imu_data.gyr, imu_data.acc, imu_data.mag)
    ref = vqf_ref.updateBatch(imu_data.gyr, imu_data.acc, imu_data.mag)

    assertBatchOutEqual(out, ref)


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_updateBatch_startInMotion(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)

    out = vqf.updateBatch(imu_data.gyr[3000:5000], imu_data.acc[3000:5000], imu_data.mag[3000:5000])
    ref = vqf_ref.updateBatch(imu_data.gyr[3000:5000], imu_data.acc[3000:5000], imu_data.mag[3000:5000])

    assertBatchOutEqual(out, ref)


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_updateBatch_6D(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)

    out = vqf.updateBatch(imu_data.gyr, imu_data.acc)
    ref = vqf_ref.updateBatch(imu_data.gyr, imu_data.acc)

    assertBatchOutEqual(out, ref, noMag=True)


@pytest.mark.parametrize('cls', ['VQF', 'PyVQF', 'BasicVQF'], indirect=True)
def test_setState(cls, imu_data):
    vqf = cls(1 / imu_data.sampling_rate)
    ref = vqf.updateBatch(imu_data.gyr[3000:3200], imu_data.acc[3000:3200], imu_data.mag[3000:3200])
    ref_state = vqf.state

    vqf = cls(1 / imu_data.sampling_rate)
    out1 = vqf.updateBatch(imu_data.gyr[3000:3100], imu_data.acc[3000:3100], imu_data.mag[3000:3100])
    state1 = vqf.state

    vqf = cls(1 / imu_data.sampling_rate)
    vqf.state = state1
    out2 = vqf.updateBatch(imu_data.gyr[3100:3200], imu_data.acc[3100:3200], imu_data.mag[3100:3200])
    state2 = vqf.state

    out = {k: np.concatenate([out1[k], out2[k]]) for k in out1}

    assertBatchOutEqual(out, ref, basic=(cls == BasicVQF))
    for k in ref_state:
        np.testing.assert_allclose(state2[k], ref_state[k])


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_getSetBiasEstimate(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)

    vqf.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])
    vqf_ref.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])

    bias, sigma = vqf.getBiasEstimate()
    bias_ref, sigma_ref = vqf_ref.getBiasEstimate()

    np.testing.assert_almost_equal(bias, bias_ref)
    np.testing.assert_almost_equal(sigma, sigma_ref)

    vqf.setBiasEstimate(np.deg2rad([-1.5, 0.8, 1.9]), sigma/2)
    vqf_ref.setBiasEstimate(np.deg2rad([-1.5, 0.8, 1.9]), sigma/2)

    out = vqf.updateBatch(imu_data.gyr[2750:2850], imu_data.acc[2750:2850], imu_data.mag[2750:2850])
    ref = vqf_ref.updateBatch(imu_data.gyr[2750:2850], imu_data.acc[2750:2850], imu_data.mag[2750:2850])

    assertBatchOutEqual(out, ref)


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_getBiasEstimate_sigma_clip(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate, restBiasEstEnabled=False)
    vqf_ref = VQF(1/imu_data.sampling_rate, restBiasEstEnabled=False)

    # rest data, the same axis is vertical all the time
    vqf.updateBatch(imu_data.gyr[:200], imu_data.acc[:200], imu_data.mag[:200])
    vqf_ref.updateBatch(imu_data.gyr[:200], imu_data.acc[:200], imu_data.mag[:200])

    bias, sigma = vqf.getBiasEstimate()
    bias_ref, sigma_ref = vqf_ref.getBiasEstimate()

    np.testing.assert_almost_equal(np.rad2deg(sigma_ref), 0.5)  # sigma output should be clipped to maximum value
    np.testing.assert_almost_equal(np.rad2deg(sigma), np.rad2deg(sigma_ref))


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_getRelativeRestDeviations(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)

    vqf.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])
    out = vqf.getRelativeRestDeviations()

    vqf_ref.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])
    ref = vqf_ref.getRelativeRestDeviations()

    np.testing.assert_allclose(out, ref)


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_getSetMagRef(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    # run C++ implementation on the first samples to obtain a valid mag reference
    vqf_pre = VQF(1/imu_data.sampling_rate)
    out_pre = vqf_pre.updateBatch(imu_data.gyr[:6000], imu_data.acc[:6000], imu_data.mag[:6000])
    assert out_pre['magDistDetected'][199]
    assert not out_pre['magDistDetected'][-1]
    norm = vqf_pre.getMagRefNorm()
    dip = vqf_pre.getMagRefDip()
    assert 40 <= norm <= 50
    assert 65 <= np.rad2deg(dip) <= 75

    # set mag reference and ensure that the magnetic field is quickly regarded as undisturbed
    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)
    vqf.setMagRef(norm, dip)
    vqf_ref.setMagRef(norm, dip)
    out = vqf.updateBatch(imu_data.gyr[:200], imu_data.acc[:200], imu_data.mag[:200])
    ref = vqf_ref.updateBatch(imu_data.gyr[:200], imu_data.acc[:200], imu_data.mag[:200])
    assertBatchOutEqual(out, ref)
    assert not ref['magDistDetected'][199]
    np.testing.assert_almost_equal(vqf.getMagRefNorm(), vqf_ref.getMagRefNorm())
    np.testing.assert_almost_equal(vqf.getMagRefDip(), vqf_ref.getMagRefDip())


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_setMotionBiasEstEnabled(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)
    out = vqf.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])
    ref = vqf_ref.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])
    assertBatchOutEqual(out, ref)

    vqf.setMotionBiasEstEnabled(False)
    vqf_ref.setMotionBiasEstEnabled(False)
    out = vqf.updateBatch(imu_data.gyr[2750:2850], imu_data.acc[2750:2850], imu_data.mag[2750:2850])
    ref = vqf_ref.updateBatch(imu_data.gyr[2750:2850], imu_data.acc[2750:2850], imu_data.mag[2750:2850])
    assertBatchOutEqual(out, ref)

    vqf.setMotionBiasEstEnabled(True)
    vqf_ref.setMotionBiasEstEnabled(True)
    out = vqf.updateBatch(imu_data.gyr[2850:2950], imu_data.acc[2850:2950], imu_data.mag[2850:2950])
    ref = vqf_ref.updateBatch(imu_data.gyr[2850:2950], imu_data.acc[2850:2950], imu_data.mag[2850:2950])
    assertBatchOutEqual(out, ref)


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_setRestBiasEstEnabled(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)
    out = vqf.updateBatch(imu_data.gyr[:1000], imu_data.acc[:1000], imu_data.mag[:1000])
    ref = vqf_ref.updateBatch(imu_data.gyr[:1000], imu_data.acc[:1000], imu_data.mag[:1000])
    assert ref['restDetected'][-1]
    assertBatchOutEqual(out, ref)

    vqf.setRestBiasEstEnabled(False)
    vqf_ref.setRestBiasEstEnabled(False)
    out = vqf.updateBatch(imu_data.gyr[1000:1100], imu_data.acc[1000:1100], imu_data.mag[1000:1100])
    ref = vqf_ref.updateBatch(imu_data.gyr[1000:1100], imu_data.acc[1000:1100], imu_data.mag[1000:1100])
    assertBatchOutEqual(out, ref)

    vqf.setRestBiasEstEnabled(True)
    vqf_ref.setRestBiasEstEnabled(True)
    out = vqf.updateBatch(imu_data.gyr[1100:2000], imu_data.acc[1100:2000], imu_data.mag[1100:2000])
    ref = vqf_ref.updateBatch(imu_data.gyr[1100:2000], imu_data.acc[1100:2000], imu_data.mag[1100:2000])
    assert ref['restDetected'][-1]
    assertBatchOutEqual(out, ref)


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_setMagDistRejectionEnabled(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)
    out = vqf.updateBatch(imu_data.gyr[:6000], imu_data.acc[:6000], imu_data.mag[:6000])
    ref = vqf_ref.updateBatch(imu_data.gyr[:6000], imu_data.acc[:6000], imu_data.mag[:6000])
    assert ref['magDistDetected'][100]
    assert not ref['magDistDetected'][-1]
    assertBatchOutEqual(out, ref)

    vqf.setMagDistRejectionEnabled(False)
    vqf_ref.setMagDistRejectionEnabled(False)
    out = vqf.updateBatch(imu_data.gyr[6000:6100], imu_data.acc[6000:6100], imu_data.mag[6000:6100])
    ref = vqf_ref.updateBatch(imu_data.gyr[6000:6100], imu_data.acc[6000:6100], imu_data.mag[6000:6100])
    assertBatchOutEqual(out, ref)

    vqf.setMagDistRejectionEnabled(True)
    vqf_ref.setMagDistRejectionEnabled(True)
    out = vqf.updateBatch(imu_data.gyr[6100:10000], imu_data.acc[6100:10000], imu_data.mag[6100:10000])
    ref = vqf_ref.updateBatch(imu_data.gyr[6100:10000], imu_data.acc[6100:10000], imu_data.mag[6100:10000])
    assert not ref['magDistDetected'][-1]
    assertBatchOutEqual(out, ref)


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_setTauAcc(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)
    defaultTauAcc = vqf_ref.params['tauAcc']
    out = vqf.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])
    ref = vqf_ref.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])
    assertBatchOutEqual(out, ref)

    # change during initial phase
    assert np.isnan(vqf_ref.state['accLpState'][0])
    vqf.setTauAcc(0.1)
    vqf_ref.setTauAcc(0.1)
    out = vqf.updateBatch(imu_data.gyr[2750:3050], imu_data.acc[2750:3050], imu_data.mag[2750:3050])
    ref = vqf_ref.updateBatch(imu_data.gyr[2750:3050], imu_data.acc[2750:3050], imu_data.mag[2750:3050])
    assertBatchOutEqual(out, ref)

    # change after initial phase
    assert not np.isnan(vqf_ref.state['accLpState'][0])
    vqf.setTauAcc(defaultTauAcc)
    vqf_ref.setTauAcc(defaultTauAcc)
    vqf.setTauAcc(defaultTauAcc/10)
    vqf_ref.setTauAcc(defaultTauAcc/10)
    out = vqf.updateBatch(imu_data.gyr[3050:3150], imu_data.acc[3050:3150], imu_data.mag[3050:3150])
    ref = vqf_ref.updateBatch(imu_data.gyr[3050:3150], imu_data.acc[3050:3150], imu_data.mag[3050:3150])
    assertBatchOutEqual(out, ref)


@pytest.mark.parametrize('cls', ['PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_setTauMag(cls, imu_data):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')

    vqf = cls(1/imu_data.sampling_rate)
    vqf_ref = VQF(1/imu_data.sampling_rate)
    out = vqf.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])
    ref = vqf_ref.updateBatch(imu_data.gyr[2650:2750], imu_data.acc[2650:2750], imu_data.mag[2650:2750])
    assertBatchOutEqual(out, ref)

    defaultTauMag = vqf_ref.params['tauMag']
    vqf.setTauMag(defaultTauMag/10)
    vqf_ref.setTauMag(defaultTauMag/10)
    out = vqf.updateBatch(imu_data.gyr[2750:2850], imu_data.acc[2750:2850], imu_data.mag[2750:2850])
    ref = vqf_ref.updateBatch(imu_data.gyr[2750:2850], imu_data.acc[2750:2850], imu_data.mag[2750:2850])
    assertBatchOutEqual(out, ref)

    vqf.setTauMag(defaultTauMag)
    vqf_ref.setTauMag(defaultTauMag)
    out = vqf.updateBatch(imu_data.gyr[2850:2950], imu_data.acc[2850:2950], imu_data.mag[2850:2950])
    ref = vqf_ref.updateBatch(imu_data.gyr[2850:2950], imu_data.acc[2850:2950], imu_data.mag[2850:2950])
    assertBatchOutEqual(out, ref)


def relQuatAngleInDeg(q1, q2):
    w = q1[..., 0] * q2[..., 0] + q1[..., 1] * q2[..., 1] + q1[..., 2] * q2[..., 2] + q1[..., 3] * q2[..., 3]
    angle = 2 * np.arccos(np.clip(np.abs(w), -1, 1))
    return np.rad2deg(angle)


def test_offlineVQF(imu_data):
    out = offlineVQF(imu_data.gyr, imu_data.acc, imu_data.mag, 1 / imu_data.sampling_rate)

    vqf_ref = VQF(1 / imu_data.sampling_rate)
    ref = vqf_ref.updateBatch(imu_data.gyr, imu_data.acc, imu_data.mag)

    # calculate angle of relative orientation in degrees
    diffAngle6D = relQuatAngleInDeg(ref['quat6D'], out['quat6D'])
    diffAngle9D = relQuatAngleInDeg(ref['quat9D'], out['quat9D'])

    # make sure the mean and maximum deviations are reasonably small
    assert np.mean(diffAngle6D) < 0.5
    assert np.max(diffAngle6D) < 1.0
    assert np.mean(diffAngle9D) < 1.0
    assert np.max(diffAngle9D) < 2.0
