# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import scipy.signal

from vqf import PyVQF

quats = [
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1],
    [2**-0.5, 0, 2**-0.5, 0],
    [2**-0.5, 0, 0, -2**-0.5],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, 0.5],
]

matrices = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[3, 0, 0], [0, 4, 0], [0, 0, 5]],
    [[0.1, 0.2, -0.15], [0.42, -0.01, 0.04], [0.02, -0.23, 0.18]],
    [[100, 200, -150], [420, -10, 40], [20, -230, 180]],
]


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('q1', quats)
@pytest.mark.parametrize('q2', quats)
def test_quatMultiply(cls, q1, q2):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    q1 = np.array(q1, float)
    q2 = np.array(q2, float)
    out = cls.quatMultiply(q1, q2)
    ref = np.array([
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    ])
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('q', quats)
def test_quatConj(cls, q):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    q = np.array(q, float)
    out = cls.quatConj(q)
    ref = np.array([q[0], -q[1], -q[2], -q[3]], float)
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF'], indirect=True)
@pytest.mark.parametrize('q', quats)
def test_quatSetToIdentity(cls, q):
    q = np.array(q, float)
    cls.quatSetToIdentity(q)
    ref = np.array([1, 0, 0, 0], float)
    np.testing.assert_almost_equal(q, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('q', quats)
@pytest.mark.parametrize('delta', np.deg2rad([0, -17, 180, 421]))
def test_quatApplyDelta(cls, q, delta):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    q = np.array(q, float)
    out = cls.quatApplyDelta(q, delta)
    ref = cls.quatMultiply(np.array([np.cos(delta/2), 0, 0, np.sin(delta/2)], float), q)
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('q', quats)
@pytest.mark.parametrize('v', [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 2, -3]])
def test_quatRotate(cls, q, v):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    q = np.array(q, float)
    v = np.array(v, float)
    out = cls.quatRotate(q, v)
    ref = cls.quatMultiply(q, cls.quatMultiply(np.concatenate([[0.0], v]), cls.quatConj(q)))
    np.testing.assert_almost_equal(out, ref[1:])


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF'], indirect=True)
@pytest.mark.parametrize('vec', [[2], [-1, 2], [1, 2, 3], [1, 2, 3, 0], [-1, -2, -3, -4, -5], [0, 0, 0]])
def test_norm(cls, vec):
    vec = np.array(vec, float)
    out = cls.norm(vec)
    ref = np.linalg.norm(vec)
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('vec', [[2], [-1, 2], [1, 2, 3], [1, 2, 3, 0], [-1, -2, -3, -4, -5]])
def test_normalize(cls, vec):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    vec = np.array(vec, float)
    ref = vec/np.linalg.norm(vec)
    out = cls.normalize(vec)
    if out is None:
        out = vec
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('vec', [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]])
def test_normalize_zeroinput(cls, vec):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    vec = np.array(vec, float)
    ref = vec.copy()
    out = cls.normalize(vec)
    if out is None:
        out = vec
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_clip(cls):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    vec = np.array([-1, 0.5, -5, 0.4, -2, -4, 5, 7, 12, 1, 0.2, 2, 3, 5], float)
    ref = np.clip(vec, -1.5, 2.5)
    out = cls.clip(vec, -1.5, 2.5)
    if out is None:
        out = vec
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('tau', [0.1, 1.0, 10.0])
@pytest.mark.parametrize('Ts', [0.01, 0.001])
def test_gainFromTau(cls, tau, Ts):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    ref = 1 - np.exp(-Ts/tau)
    out = cls.gainFromTau(tau, Ts)
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('Ts', [0.01, 0.001])
def test_gainFromTau_zero(cls, Ts):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    out = cls.gainFromTau(0.0, Ts)
    np.testing.assert_almost_equal(out, 1)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('Ts', [0.01, 0.001])
def test_gainFromTau_negative(cls, Ts):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    out = cls.gainFromTau(-1.0, Ts)
    np.testing.assert_almost_equal(out, 0)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('tau', [0.1, 1.0, 10.0])
@pytest.mark.parametrize('Ts', [0.01, 0.001])
def test_filterCoeffs(cls, tau, Ts):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    outB, outA = cls.filterCoeffs(tau, Ts)
    fc = np.sqrt(2) / (2 * np.pi * tau)
    Wn = fc / (1/Ts/2)
    refB, refA = scipy.signal.butter(2, Wn)
    np.testing.assert_almost_equal(outB, refB)
    np.testing.assert_almost_equal(outA, refA[1:])
    assert refA[0] == 1.0


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
@pytest.mark.parametrize('tau', [0.1, 1.0, 10.0])
@pytest.mark.parametrize('Ts', [0.01, 0.001])
@pytest.mark.parametrize('x0', [1.0, 10.0])
def test_filterInitialState(cls, tau, Ts, x0):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    b, a = cls.filterCoeffs(tau, Ts)
    out = cls.filterInitialState(x0, b, a)
    ref = x0*scipy.signal.lfilter_zi(b, np.concatenate([[1.0], a]))
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_filterVec(cls):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    tau = 1.0
    Ts = 0.1
    t = np.arange(0, 10, Ts)
    x = np.column_stack([np.sin(t), np.cos(t)])
    b, a = cls.filterCoeffs(tau, Ts)
    state = np.zeros((2, 2) if hasattr(cls, 'is_matlab') or cls == PyVQF else 2*2)
    out = np.zeros_like(x)
    for i in range(len(t)):
        tmp = cls.filterVec(x[i], tau, Ts, b, a, state)
        if isinstance(tmp, list):
            out[i], state = tmp
        else:
            out[i] = tmp

    ref = scipy.signal.lfilter(b,  np.concatenate([[1.0], a]), x, axis=0)

    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF', 'BasicVQF', 'PyVQF', 'MatlabVQF', 'OctaveVQF'], indirect=True)
def test_filterVec_init(cls):
    if cls is None:
        pytest.skip('--nomatlab and/or --nooctave is set')
    tau = 1.0
    Ts = 0.1
    t = np.arange(0, 2, Ts)
    x = np.column_stack([np.sin(t), np.cos(t)])
    b, a = cls.filterCoeffs(tau, Ts)
    state = np.full((2, 2) if hasattr(cls, 'is_matlab') or cls == PyVQF else 2*2, fill_value=np.nan)
    out = np.zeros_like(x)
    ref = np.zeros_like(x)

    for i in range(len(t)):
        tmp = cls.filterVec(x[i], tau, Ts, b, a, state)
        if isinstance(tmp, list):
            out[i], state = tmp
        else:
            out[i] = tmp
        # for t < tau: calculate mean
        if t[i] <= tau:
            ref[i] = np.mean(x[:i+1], axis=0)

    # calculate initial state based on last mean value
    ind = np.flatnonzero(t >= tau)[0]
    zi = ref[None, ind-1]*scipy.signal.lfilter_zi(b, np.concatenate([[1.0], a]))[:, None]
    ref[ind:] = scipy.signal.lfilter(b,  np.concatenate([[1.0], a]), x[ind:], axis=0, zi=zi)[0]

    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF'], indirect=True)
@pytest.mark.parametrize('mat', matrices)
def test_matrix3SetToScaledIdentity(cls, mat):
    mat = np.array(mat, float)
    cls.matrix3SetToScaledIdentity(3, mat)
    np.testing.assert_almost_equal(mat, 3*np.eye(3))


@pytest.mark.parametrize('cls', ['VQF'], indirect=True)
@pytest.mark.parametrize('mat1', matrices)
@pytest.mark.parametrize('mat2', matrices)
def test_matrix3Multiply(cls, mat1, mat2):
    mat1 = np.array(mat1, float)
    mat2 = np.array(mat2, float)
    out = cls.matrix3Multiply(mat1, mat2)
    ref = mat1 @ mat2
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF'], indirect=True)
@pytest.mark.parametrize('mat1', matrices)
@pytest.mark.parametrize('mat2', matrices)
def test_matrix3MultiplyTpsFirst(cls, mat1, mat2):
    mat1 = np.array(mat1, float)
    mat2 = np.array(mat2, float)
    out = cls.matrix3MultiplyTpsFirst(mat1, mat2)
    ref = mat1.T @ mat2
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF'], indirect=True)
@pytest.mark.parametrize('mat1', matrices)
@pytest.mark.parametrize('mat2', matrices)
def test_matrix3MultiplyTpsSecond(cls, mat1, mat2):
    mat1 = np.array(mat1, float)
    mat2 = np.array(mat2, float)
    out = cls.matrix3MultiplyTpsSecond(mat1, mat2)
    ref = mat1 @ mat2.T
    np.testing.assert_almost_equal(out, ref)


@pytest.mark.parametrize('cls', ['VQF'], indirect=True)
@pytest.mark.parametrize('mat', matrices)
def test_matrix3Inv(cls, mat):
    mat = np.array(mat, float)
    success, out = cls.matrix3Inv(mat)
    ref = np.linalg.inv(mat)
    assert success
    np.testing.assert_almost_equal(out, ref)
