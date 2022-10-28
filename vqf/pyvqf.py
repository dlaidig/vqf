# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import copy
import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np


EPS = np.finfo(float).eps


@dataclass
class PyVQFParams:
    tauAcc: float = 3.0
    tauMag: float = 9.0
    motionBiasEstEnabled: bool = True
    restBiasEstEnabled: bool = True
    magDistRejectionEnabled: bool = True
    biasSigmaInit: float = 0.5
    biasForgettingTime: float = 100.0
    biasClip: float = 2.0
    biasSigmaMotion: float = 0.1
    biasVerticalForgettingFactor: float = 0.0001
    biasSigmaRest: float = 0.03
    restMinT: float = 1.5
    restFilterTau: float = 0.5
    restThGyr: float = 2.0
    restThAcc: float = 0.5
    magCurrentTau: float = 0.05
    magRefTau: float = 20.0
    magNormTh: float = 0.1
    magDipTh: float = 10.0
    magNewTime: float = 20.0
    magNewFirstTime: float = 5.0
    magNewMinGyr: float = 20.0
    magMinUndisturbedTime: float = 0.5
    magMaxRejectionTime: float = 60.0
    magRejectionFactor: float = 2.0


@dataclass
class PyVQFState:
    gyrQuat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], float))
    accQuat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], float))
    delta: float = 0.0

    restDetected: bool = False
    magDistDetected: bool = True

    lastAccLp: np.ndarray = field(default_factory=lambda: np.zeros(3, float))
    accLpState: np.ndarray = field(default_factory=lambda: np.full((2, 3), np.nan, float))

    kMagInit: float = 1.0
    lastMagDisAngle: float = 0.0
    lastMagCorrAngularRate: float = 0.0

    bias: np.ndarray = field(default_factory=lambda: np.zeros(3, float))
    biasP: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan, float))

    motionBiasEstRLpState: np.ndarray = field(default_factory=lambda: np.full((2, 9), np.nan, float))
    motionBiasEstBiasLpState: np.ndarray = field(default_factory=lambda: np.full((2, 2), np.nan, float))

    restLastSquaredDeviations: np.ndarray = field(default_factory=lambda: np.zeros(2, float))
    restT: float = 0.0
    restLastGyrLp: np.ndarray = field(default_factory=lambda: np.zeros(3, float))
    restGyrLpState: np.ndarray = field(default_factory=lambda: np.full((2, 3), np.nan, float))
    restLastAccLp: np.ndarray = field(default_factory=lambda: np.zeros(3, float))
    restAccLpState: np.ndarray = field(default_factory=lambda: np.full((2, 3), np.nan, float))

    magRefNorm: float = 0.0
    magRefDip: float = 0.0
    magUndisturbedT: float = 0.0
    magRejectT: float = -1.0
    magCandidateNorm: float = -1.0
    magCandidateDip: float = 0.0
    magCandidateT: float = 0.0
    magNormDip: np.ndarray = field(default_factory=lambda: np.zeros(2, float))
    magNormDipLpState: np.ndarray = field(default_factory=lambda: np.full((2, 2), np.nan, float))


@dataclass
class PyVQFCoefficients:
    gyrTs: float
    accTs: float
    magTs: float

    accLpB: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, float))
    accLpA: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, float))

    kMag: float = -1.0

    biasP0: float = -1.0
    biasV: float = -1.0
    biasMotionW: float = -1.0
    biasVerticalW: float = -1.0
    biasRestW: float = -1.0

    restGyrLpB: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, float))
    restGyrLpA: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, float))
    restAccLpB: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, float))
    restAccLpA: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, float))

    kMagRef: float = -1.0
    magNormDipLpB: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, float))
    magNormDipLpA: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, float))


class PyVQF:
    """A Versatile Quaternion-based Filter for IMU Orientation Estimation.

    This class implements the orientation estimation filter described in the following publication:

        D. Laidig and T. Seel. "VQF: Highly Accurate IMU Orientation Estimation with Bias Estimation and Magnetic
        Disturbance Rejection." Information Fusion 2023, 91, 187--204.
        `doi:10.1016/j.inffus.2022.10.014 <https://doi.org/10.1016/j.inffus.2022.10.014>`_.
        [Accepted manuscript available at `arXiv:2203.17024 <https://arxiv.org/abs/2203.17024>`_.]

    The filter can perform simultaneous 6D (magnetometer-free) and 9D (gyr+acc+mag) sensor fusion and can also be used
    without magnetometer data. It performs rest detection, gyroscope bias estimation during rest and motion, and
    magnetic disturbance detection and rejection. Different sampling rates for gyroscopes, accelerometers, and
    magnetometers are supported as well. While in most cases, the defaults will be reasonable, the algorithm can be
    influenced via a number of tuning parameters.

    To use this class for online (sample-by-sample) processing,

    1. create a instance of the class and provide the sampling time and, optionally, parameters
    2. for every sample, call one of the update functions to feed the algorithm with IMU data
    3. access the estimation results with :meth:`getQuat6D`, :meth:`getQuat9D` and the other getter methods.

    If the full data is available in numpy arrays, you can use :meth:`updateBatch`.

    This class is a pure Python implementation of the algorithm that only depends on the Python standard library and
    `numpy <https://numpy.org/>`_. Note that the wrapper :class:`vqf.VQF` for the C++ implementation :cpp:class:`VQF`
    is much faster than this pure Python implementation. Depending on use case and programming language of choice,
    the following alternatives might be useful:

    +------------------------+---------------------------+--------------------------+---------------------------+
    |                        | Full Version              | Basic Version            | Offline Version           |
    |                        |                           |                          |                           |
    +========================+===========================+==========================+===========================+
    | **C++**                | :cpp:class:`VQF`          | :cpp:class:`BasicVQF`    | :cpp:func:`offlineVQF`    |
    +------------------------+---------------------------+--------------------------+---------------------------+
    | **Python/C++ (fast)**  | :py:class:`vqf.VQF`       | :py:class:`vqf.BasicVQF` | :py:meth:`vqf.offlineVQF` |
    +------------------------+---------------------------+--------------------------+---------------------------+
    | **Pure Python (slow)** | **vqf.PyVQ (this class)** | --                       | --                        |
    +------------------------+---------------------------+--------------------------+---------------------------+
    | **Pure Matlab (slow)** | :mat:class:`VQF.m <VQF>`  | --                       | --                        |
    +------------------------+---------------------------+--------------------------+---------------------------+

    In the most common case (using the default parameters and all data being sampled with the same frequency, create the
    class like this:

    .. code-block::

         from vqf import PyVQF
         vqf = PyVQF(0.01)  # 0.01 s sampling time, i.e. 100 Hz

    Example code to create an object with magnetic disturbance rejection disabled (use the `**-operator
    <https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists>`_ to pass  parameters from a dict):

    .. code-block::

        from vqf import PyVQF
        vqf = PyVQF(0.01, magDistRejectionEnabled=False)  # 0.01 s sampling time, i.e. 100 Hz

    To use this class as a replacement for the basic version BasicVQF, pass the following parameters:

    .. code-block::

        from vqf import PyVQF
        vqf = PyVQF(0.01, motionBiasEstEnabled=False, restBiasEstEnabled=False, magDistRejectionEnabled=False)

    See :cpp:struct:`VQFParams` for a detailed description of all parameters.
    """
    def __init__(self, gyrTs, accTs=-1.0, magTs=-1.0, **params):
        """

        :param gyrTs: sampling time of the gyroscope measurements in seconds
        :param accTs: sampling time of the accelerometer measurements in seconds
            (the value of `gyrTs` is used if set to -1)
        :param magTs: sampling time of the magnetometer measurements in seconds
            (the value of `gyrTs` is used if set to -1)
        :param (params): optional parameters to override the defaults
            (see :cpp:struct:`VQFParams` for a full list and detailed descriptions)
        """
        accTs = accTs if accTs > 0 else gyrTs
        magTs = magTs if magTs > 0 else gyrTs

        self._params = PyVQFParams(**params)
        self._state = PyVQFState()
        self._coeffs = PyVQFCoefficients(gyrTs=gyrTs, accTs=accTs, magTs=magTs)

        self._setup()

    def updateGyr(self, gyr):
        """Performs gyroscope update step.

        It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
        different sampling rates. Otherwise, simply use :meth:`update`.

        :param gyr: gyroscope measurement in rad/s -- numpy array with shape (3,)
        :return: None
        """
        assert gyr.shape == (3,)

        # rest detection
        if self._params.restBiasEstEnabled or self._params.magDistRejectionEnabled:
            gyrLp = self.filterVec(gyr, self._params.restFilterTau, self._coeffs.gyrTs, self._coeffs.restGyrLpB,
                                   self._coeffs.restGyrLpA, self._state.restGyrLpState)

            deviation = gyr - gyrLp
            squaredDeviation = deviation.dot(deviation)

            biasClip = self._params.biasClip*np.pi/180.0
            if squaredDeviation >= (self._params.restThGyr*np.pi/180.0)**2 or np.max(np.abs(gyrLp)) > biasClip:
                self._state.restT = 0.0
                self._state.restDetected = False
            self._state.restLastGyrLp = gyrLp
            self._state.restLastSquaredDeviations[0] = squaredDeviation

        # remove estimated gyro bias
        gyrNoBias = gyr - self._state.bias

        # gyroscope prediction step
        gyrNorm = math.sqrt(gyrNoBias.dot(gyrNoBias))
        angle = gyrNorm * self._coeffs.gyrTs
        if gyrNorm > EPS:
            c = np.cos(angle/2)
            s = np.sin(angle/2)/gyrNorm
            gyrStepQuat = np.array([c, s*gyrNoBias[0], s*gyrNoBias[1], s*gyrNoBias[2]], float)
            self._state.gyrQuat = self.quatMultiply(self._state.gyrQuat, gyrStepQuat)
            self.normalize(self._state.gyrQuat)

    def updateAcc(self, acc):
        """Performs accelerometer update step.

        It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
        different sampling rates. Otherwise, simply use :meth:`update`.

        Should be called after :meth:`updateGyr` and before :meth:`updateMag`.

        :param acc: accelerometer measurement in m/s² -- numpy array with shape (3,)
        :return: None
        """
        assert acc.shape == (3,)

        # ignore [0 0 0] samples
        if acc[0] == 0.0 and acc[1] == 0.0 and acc[2] == 0.0:
            return

        accTs = self._coeffs.accTs

        # rest detection
        if self._params.restBiasEstEnabled:
            accLp = self.filterVec(acc, self._params.restFilterTau, accTs, self._coeffs.restAccLpB,
                                   self._coeffs.restAccLpA, self._state.restAccLpState)

            deviation = acc - accLp
            squaredDeviation = deviation.dot(deviation)

            if squaredDeviation >= self._params.restThAcc**2:
                self._state.restT = 0.0
                self._state.restDetected = False
            else:
                self._state.restT += accTs
                if self._state.restT >= self._params.restMinT:
                    self._state.restDetected = True

            self._state.restLastAccLp = accLp
            self._state.restLastSquaredDeviations[1] = squaredDeviation

        # filter acc in inertial frame
        accEarth = self.quatRotate(self._state.gyrQuat, acc)
        self._state.lastAccLp = self.filterVec(accEarth, self._params.tauAcc, accTs,
                                               self._coeffs.accLpB, self._coeffs.accLpA, self._state.accLpState)

        # transform to 6D earth frame and normalize
        accEarth = self.quatRotate(self._state.accQuat, self._state.lastAccLp)
        # a = self._state.accQuat
        # b = self.state.lastAccLp
        # accEarth = self.quatRotate(a, b)
        self.normalize(accEarth)

        # inclination correction
        q_w = math.sqrt((accEarth[2]+1)/2)
        if q_w > 1e-6:
            accCorrQuat = np.array([q_w, 0.5*accEarth[1]/q_w, -0.5*accEarth[0]/q_w, 0], float)
        else:
            accCorrQuat = np.array([0, 1, 0, 0], float)
        self._state.accQuat = self.quatMultiply(accCorrQuat, self._state.accQuat)
        self.normalize(self._state.accQuat)

        # calculate correction angular rate to facilitate debugging
        self._state.lastAccCorrAngularRate = math.acos(accEarth[2])/self._coeffs.accTs

        # bias estimation
        if self._params.motionBiasEstEnabled or self._params.restBiasEstEnabled:
            biasClip = self._params.biasClip*np.pi/180.0
            bias = self._state.bias

            # get rotation matrix corresponding to accGyrQuat
            accGyrQuat = self.getQuat6D()
            R = np.array([
                1 - 2*accGyrQuat[2]**2 - 2*accGyrQuat[3]**2,  # r11
                2*(accGyrQuat[2]*accGyrQuat[1] - accGyrQuat[0]*accGyrQuat[3]),  # r12
                2*(accGyrQuat[0]*accGyrQuat[2] + accGyrQuat[3]*accGyrQuat[1]),  # r13
                2*(accGyrQuat[0]*accGyrQuat[3] + accGyrQuat[2]*accGyrQuat[1]),  # r21
                1 - 2*accGyrQuat[1]**2 - 2*accGyrQuat[3]**2,  # r22
                2*(accGyrQuat[2]*accGyrQuat[3] - accGyrQuat[1]*accGyrQuat[0]),  # r23
                2*(accGyrQuat[3]*accGyrQuat[1] - accGyrQuat[0]*accGyrQuat[2]),  # r31
                2*(accGyrQuat[0]*accGyrQuat[1] + accGyrQuat[3]*accGyrQuat[2]),  # r32
                1 - 2*accGyrQuat[1]**2 - 2*accGyrQuat[2]**2,  # r33
            ], float)

            # calculate R*b_hat (only the x and y component, as z is not needed)
            biasLp = np.array([
                R[0]*bias[0] + R[1]*bias[1] + R[2]*bias[2],
                R[3]*bias[0] + R[4]*bias[1] + R[5]*bias[2],
            ], float)

            # low-pass filter R and R*b_hat
            R = self.filterVec(R, self._params.tauAcc, accTs, self._coeffs.accLpB, self._coeffs.accLpA,
                               self._state.motionBiasEstRLpState)
            biasLp = self.filterVec(biasLp, self._params.tauAcc, accTs, self._coeffs.accLpB,
                                    self._coeffs.accLpA, self._state.motionBiasEstBiasLpState)

            # set measurement error and covariance for the respective Kalman filter update
            if self._state.restDetected and self._params.restBiasEstEnabled:
                e = self._state.restLastGyrLp - bias
                R = np.eye(3)
                w = np.full(3, self._coeffs.biasRestW)
            elif self._params.motionBiasEstEnabled:
                e = np.array([
                    -accEarth[1]/accTs + biasLp[0] - R[0]*bias[0] - R[1]*bias[1] - R[2]*bias[2],
                    accEarth[0]/accTs + biasLp[1] - R[3]*bias[0] - R[4]*bias[1] - R[5]*bias[2],
                    - R[6]*bias[0] - R[7]*bias[1] - R[8]*bias[2],
                ], float)
                R.shape = (3, 3)
                w = np.array([self._coeffs.biasMotionW, self._coeffs.biasMotionW, self._coeffs.biasVerticalW], float)
            else:
                w = None
                e = None

            # Kalman filter update
            # step 1: P = P + V (also increase covariance if there is no measurement update!)
            if self._state.biasP[0, 0] < self._coeffs.biasP0:
                self._state.biasP[0, 0] += self._coeffs.biasV
            if self._state.biasP[1, 1] < self._coeffs.biasP0:
                self._state.biasP[1, 1] += self._coeffs.biasV
            if self._state.biasP[2, 2] < self._coeffs.biasP0:
                self._state.biasP[2, 2] += self._coeffs.biasV
            if w is not None:
                # clip disagreement to -2..2 °/s
                # (this also effectively limits the harm done by the first inclination correction step)
                e = np.clip(e, -biasClip, biasClip)

                # step 2: K = P R^T inv(W + R P R^T)
                K = self._state.biasP @ R.T @ np.linalg.inv(np.diag(w) + R @ self._state.biasP @ R.T)

                # step 3: bias = bias + K (y - R bias) = bias + K e
                bias += K @ e

                # step 4: P = P - K R P
                self._state.biasP -= K @ R @ self._state.biasP

                # clip bias estimate to -2..2 °/s
                bias[:] = np.clip(bias, -biasClip, biasClip)

    def updateMag(self, mag):
        """Performs magnetometer update step.

        It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
        different sampling rates. Otherwise, simply use :meth:`update`.

        Should be called after :meth:`updateAcc`.

        :param mag: magnetometer measurement in arbitrary units -- numpy array with shape (3,)
        :return: None
        """
        assert mag.shape == (3,)

        # ignore [0 0 0] samples
        if mag[0] == 0.0 and mag[1] == 0.0 and mag[2] == 0.0:
            return

        magTs = self._coeffs.magTs

        # bring magnetometer measurement into 6D earth frame
        magEarth = self.quatRotate(self.getQuat6D(), mag)

        if self._params.magDistRejectionEnabled:
            magNormDip = self._state.magNormDip
            magNormDip[0] = math.sqrt(magEarth.dot(magEarth))
            magNormDip[1] = -math.asin(magEarth[2]/magNormDip[0])

            if self._params.magCurrentTau > 0:
                magNormDip[:] = self.filterVec(magNormDip, self._params.magCurrentTau, magTs,
                                               self._coeffs.magNormDipLpB, self._coeffs.magNormDipLpA,
                                               self._state.magNormDipLpState)

            # magnetic disturbance detection
            if abs(magNormDip[0] - self._state.magRefNorm) < self._params.magNormTh*self._state.magRefNorm and \
                    abs(magNormDip[1] - self._state.magRefDip) < self._params.magDipTh*np.pi/180.0:
                self._state.magUndisturbedT += magTs
                if self._state.magUndisturbedT >= self._params.magMinUndisturbedTime:
                    self._state.magDistDetected = False
                    self._state.magRefNorm += self._coeffs.kMagRef*(magNormDip[0] - self._state.magRefNorm)
                    self._state.magRefDip += self._coeffs.kMagRef*(magNormDip[1] - self._state.magRefDip)
            else:
                self._state.magUndisturbedT = 0.0
                self._state.magDistDetected = True

            # new magnetic field acceptance
            if abs(magNormDip[0] - self._state.magCandidateNorm) < self._params.magNormTh*self._state.magCandidateNorm \
                    and abs(magNormDip[1] - self._state.magCandidateDip) < self._params.magDipTh*np.pi/180.0:
                gyrNorm = math.sqrt(self._state.restLastGyrLp.dot(self._state.restLastGyrLp))
                if gyrNorm >= self._params.magNewMinGyr*np.pi/180.0:
                    self._state.magCandidateT += magTs

                self._state.magCandidateNorm += self._coeffs.kMagRef*(magNormDip[0] - self._state.magCandidateNorm)
                self._state.magCandidateDip += self._coeffs.kMagRef*(magNormDip[1] - self._state.magCandidateDip)

                if self._state.magDistDetected and (self._state.magCandidateT >= self._params.magNewTime or (
                        self._state.magRefNorm == 0.0 and self._state.magCandidateT >= self._params.magNewFirstTime)):
                    self._state.magRefNorm = self._state.magCandidateNorm
                    self._state.magRefDip = self._state.magCandidateDip
                    self._state.magDistDetected = False
                    self._state.magUndisturbedT = self._params.magMinUndisturbedTime
            else:
                self._state.magCandidateT = 0.0
                self._state.magCandidateNorm = magNormDip[0]
                self._state.magCandidateDip = magNormDip[1]

        # calculate disagreement angle based on current magnetometer measurement
        self._state.lastMagDisAngle = math.atan2(magEarth[0], magEarth[1]) - self._state.delta

        # make sure the disagreement angle is in the range [-pi, pi]
        if self._state.lastMagDisAngle > np.pi:
            self._state.lastMagDisAngle -= 2*np.pi
        elif self._state.lastMagDisAngle < -np.pi:
            self._state.lastMagDisAngle += 2*np.pi

        k = self._coeffs.kMag

        if self._params.magDistRejectionEnabled:
            # magnetic disturbance rejection
            if self._state.magDistDetected:
                if self._state.magRejectT <= self._params.magMaxRejectionTime:
                    self._state.magRejectT += magTs
                    k = 0
                else:
                    k /= self._params.magRejectionFactor
            else:
                self._state.magRejectT = max(self._state.magRejectT - self._params.magRejectionFactor*magTs, 0.0)

        # ensure fast initial convergence
        if self._state.kMagInit != 0.0:
            # make sure that the gain k is at least 1/N, N=1,2,3,... in the first few samples
            if k < self._state.kMagInit:
                k = self._state.kMagInit

            # iterative expression to calculate 1/N
            self._state.kMagInit = self._state.kMagInit/(self._state.kMagInit+1)

            # disable if t > tauMag
            if self._state.kMagInit*self._params.tauMag < self._coeffs.magTs:
                self._state.kMagInit = 0.0

        # first-order filter step
        self._state.delta += k*self._state.lastMagDisAngle
        # calculate correction angular rate to facilitate debugging
        self._state.lastMagCorrAngularRate = k*self._state.lastMagDisAngle/self._coeffs.magTs

        # make sure delta is in the range [-pi, pi]
        if self._state.delta > np.pi:
            self._state.delta -= 2*np.pi
        elif self._state.delta < -np.pi:
            self._state.delta += 2*np.pi

    def update(self, gyr, acc, mag=None):
        """Performs filter update step for one sample.

        :param gyr: gyr gyroscope measurement in rad/s -- numpy array with shape (3,)
        :param acc: acc accelerometer measurement in m/s² -- numpy array with shape (3,)
        :param mag: optional mag magnetometer measurement in arbitrary units -- numpy array with shape (3,)
        :return: None
        """
        self.updateGyr(gyr)
        self.updateAcc(acc)
        if mag is not None:
            self.updateMag(mag)

    def updateBatch(self, gyr, acc, mag=None):
        """Performs batch update for multiple samples at once.

        In order to use this function, all input data must have the same sampling rate and be provided as a
        numpy array. The output is a dictionary containing

        - **quat6D** -- the 6D quaternion -- numpy array with shape (N, 4)
        - **bias** -- gyroscope bias estimate in rad/s -- numpy array with shape (N, 3)
        - **biasSigma** -- uncertainty of gyroscope bias estimate in rad/s -- numpy array with shape (N,)
        - **restDetected** -- rest detection state -- boolean numpy array with shape (N,)

        in all cases and if magnetometer data is provided additionally

        - **quat9D** -- the 9D quaternion -- numpy array with shape (N, 4)
        - **delta** -- heading difference angle between 6D and 9D quaternion in rad -- numpy array with shape (N,)
        - **magDistDetected** -- magnetic disturbance detection state -- boolean numpy array with shape (N,)

        :param gyr: gyroscope measurement in rad/s -- numpy array with shape (N,3)
        :param acc: accelerometer measurement in m/s² -- numpy array with shape (N,3)
        :param mag: optional magnetometer measurement in arbitrary units -- numpy array with shape (N,3)
        :return: dict with entries as described above
        """
        N = gyr.shape[0]
        assert acc.shape == gyr.shape
        assert gyr.shape == (N, 3)

        out6D = np.empty((N, 4))
        outBias = np.empty((N, 3))
        outBiasSigma = np.empty((N,))
        outRest = np.empty(N, dtype=bool)

        if mag is not None:
            assert mag.shape == gyr.shape
            out9D = np.empty((N, 4))
            outDelta = np.empty((N,))
            outMagDist = np.empty(N, dtype=bool)
            for i in range(N):
                self.update(gyr[i], acc[i], mag[i])
                out6D[i] = self.getQuat6D()
                out9D[i] = self.getQuat9D()
                outDelta[i] = self._state.delta
                outBias[i], outBiasSigma[i] = self.getBiasEstimate()
                outRest[i] = self._state.restDetected
                outMagDist[i] = self._state.magDistDetected
            return dict(quat6D=out6D, quat9D=out9D, delta=outDelta, bias=outBias, biasSigma=outBiasSigma,
                        restDetected=outRest, magDistDetected=outMagDist)
        else:
            for i in range(N):
                self.update(gyr[i], acc[i])
                out6D[i] = self.getQuat6D()
                outBias[i], outBiasSigma[i] = self.getBiasEstimate()
                outRest[i] = self._state.restDetected
            return dict(quat6D=out6D, bias=outBias, biasSigma=outBiasSigma, restDetected=outRest)

    def getQuat3D(self):
        r"""Returns the angular velocity strapdown integration quaternion
        :math:`^{\mathcal{S}_i}_{\mathcal{I}_i}\mathbf{q}`.

        :return: quaternion as numpy array with shape (4,)
        """
        return self._state.gyrQuat.copy()

    def getQuat6D(self):
        r"""Returns the 6D (magnetometer-free) orientation quaternion
        :math:`^{\mathcal{S}_i}_{\mathcal{E}_i}\mathbf{q}`.

        :return: quaternion as numpy array with shape (4,)
        """
        return self.quatMultiply(self._state.accQuat, self._state.gyrQuat)

    def getQuat9D(self):
        r"""Returns the 9D (with magnetometers) orientation quaternion
        :math:`^{\mathcal{S}_i}_{\mathcal{E}}\mathbf{q}`.

        :return: quaternion as numpy array with shape (4,)
        """
        return self.quatApplyDelta(self.quatMultiply(self._state.accQuat, self._state.gyrQuat), self._state.delta)

    def getDelta(self):
        r""" Returns the heading difference :math:`\delta` between :math:`\mathcal{E}_i` and :math:`\mathcal{E}`.

        :math:`^{\mathcal{E}_i}_{\mathcal{E}}\mathbf{q} = \begin{bmatrix}\cos\frac{\delta}{2} & 0 & 0 &
        \sin\frac{\delta}{2}\end{bmatrix}^T`.

        :return: delta angle in rad (:cpp:member:`VQFState::delta`)
        """
        return self._state.delta

    def getBiasEstimate(self):
        """Returns the current gyroscope bias estimate and the uncertainty.

        The returned standard deviation sigma represents the estimation uncertainty in the worst direction and is based
        on an upper bound of the largest eigenvalue of the covariance matrix.

        :return: gyroscope bias estimate (rad/s) as (3,) numpy array and standard deviation sigma of the estimation
            uncertainty (rad/s)
        """
        # use largest absolute row sum as upper bound estimate for largest eigenvalue (Gershgorin circle theorem)
        # and clip output to biasSigmaInit
        P = min(np.max(np.sum(np.abs(self._state.biasP), axis=1)), self._coeffs.biasP0)
        sigma = np.sqrt(P)*np.pi/100.0/180.0
        return self._state.bias.copy(), sigma

    def setBiasEstimate(self, bias, sigma):
        """Sets the current gyroscope bias estimate and the uncertainty.

        If a value for the uncertainty sigma is given, the covariance matrix is set to a corresponding scaled identity
        matrix.

        :param bias: gyroscope bias estimate (rad/s)
        :param sigma: standard deviation of the estimation uncertainty (rad/s) - set to -1 (default) in order to not
            change the estimation covariance matrix
        """
        assert bias.shape == (3,)
        self._state.bias[:] = bias
        if sigma > 0:
            self._state.biasP = (sigma*180.0*100.0/np.pi)**2 * np.eye(3)

    def getRestDetected(self):
        """Returns true if rest was detected."""
        return self._state.restDetected

    def getMagDistDetected(self):
        """Returns true if a disturbed magnetic field was detected."""
        return self._state.magDistDetected

    def getRelativeRestDeviations(self):
        """Returns the relative deviations used in rest detection.

         Looking at those values can be useful to understand how rest detection is working and which thresholds are
         suitable. The output array is filled with the last values for gyroscope and accelerometer,
         relative to the threshold. In order for rest to be detected, both values must stay below 1.

        :return: relative rest deviations as (2,) numpy array
        """
        return np.array([
            np.sqrt(self._state.restLastSquaredDeviations[0]) / (self._params.restThGyr * np.pi / 180.0),
            np.sqrt(self._state.restLastSquaredDeviations[1]) / self._params.restThAcc,
        ], float)

    def getMagRefNorm(self):
        """Returns the norm of the currently accepted magnetic field reference."""
        return self._state.magRefNorm

    def getMagRefDip(self):
        """Returns the dip angle of the currently accepted magnetic field reference."""
        return self._state.magRefDip

    def setMagRef(self, norm, dip):
        """Overwrites the current magnetic field reference.

        :param norm: norm of the magnetic field reference
        :param dip: dip angle of the magnetic field reference
        """
        self._state.magRefNorm = norm
        self._state.magRefDip = dip

    def setTauAcc(self, tauAcc):
        r"""Sets the time constant for accelerometer low-pass filtering.

        For more details, see :cpp:member:`VQFParams::tauAcc`.

        :param tauAcc: time constant :math:`\tau_\mathrm{acc}` in seconds
        """
        if self._params.tauAcc == tauAcc:
            return
        self._params.tauAcc = tauAcc
        newB, newA = self.filterCoeffs(self._params.tauAcc, self._coeffs.accTs)

        self.filterAdaptStateForCoeffChange(self._state.lastAccLp, self._coeffs.accLpB, self._coeffs.accLpA,
                                            newB, newA, self._state.accLpState)
        # For R and biasLP, the last value is not saved in the state.
        # Since b0 is small (at reasonable settings), the last output is close to state[0].
        self.filterAdaptStateForCoeffChange(self._state.motionBiasEstRLpState[0].copy(), self._coeffs.accLpB,
                                            self._coeffs.accLpA, newB, newA, self._state.motionBiasEstRLpState)
        self.filterAdaptStateForCoeffChange(self._state.motionBiasEstBiasLpState[0].copy(), self._coeffs.accLpB,
                                            self._coeffs.accLpA, newB, newA, self._state.motionBiasEstBiasLpState)

        self._coeffs.accLpB = newB
        self._coeffs.accLpA = newA

    def setTauMag(self, tauMag):
        r"""Sets the time constant for the magnetometer update.

        For more details, see :cpp:member:`VQFParams::tauMag`.

        :param tauMag: time constant :math:`\tau_\mathrm{mag}` in seconds
        """
        self._params.tauMag = tauMag
        self._coeffs.kMag = self.gainFromTau(self._params.tauMag, self._coeffs.magTs)

    def setMotionBiasEstEnabled(self, enabled):
        """Enables/disabled gyroscope bias estimation during motion."""
        if self._params.motionBiasEstEnabled == enabled:
            return
        self._params.motionBiasEstEnabled = enabled
        self._state.motionBiasEstRLpState = np.full((2, 9), np.nan, float)
        self._state.motionBiasEstBiasLpState = np.full((2, 2), np.nan, float)

    def setRestBiasEstEnabled(self, enabled):
        """Enables/disables rest detection and bias estimation during rest."""
        if self._params.restBiasEstEnabled == enabled:
            return
        self._params.restBiasEstEnabled = enabled
        self._state.restDetected = False
        self._state.restLastSquaredDeviations = np.zeros(3, float)
        self._state.restT = 0.0
        self._state.restLastGyrLp = np.zeros(3, float)
        self._state.restGyrLpState = np.full((2, 3), np.nan, float)
        self._state.restLastAccLp = np.zeros(3, float)
        self._state.restAccLpState = np.full((2, 3), np.nan, float)

    def setMagDistRejectionEnabled(self, enabled):
        """Enables/disables magnetic disturbance detection and rejection."""
        if self._params.magDistRejectionEnabled == enabled:
            return
        self._params.magDistRejectionEnabled = enabled
        self._state.magDistDetected = True
        self._state.magRefNorm = 0.0
        self._state.magRefDip = 0.0
        self._state.magUndisturbedT = 0.0
        self._state.magRejectT = self._params.magMaxRejectionTime
        self._state.magCandidateNorm = -1.0
        self._state.magCandidateDip = 0.0
        self._state.magCandidateT = 0.0
        self._state.magNormDip = np.zeros(2, float)
        self._state.magNormDipLpState = np.full((2, 2), np.nan, float)

    def setRestDetectionThresholds(self, thGyr, thAcc):
        """Sets the current thresholds for rest detection.

        :param thGyr: new value for :cpp:member:`VQFParams::restThGyr`
        :param thAcc: new value for :cpp:member:`VQFParams::restThAcc`
        """
        self._params.restThGyr = thGyr
        self._params.restThAcc = thAcc

    @property
    def params(self):
        """Read-only property to access the current parameters.

        :return: dict with entries corresponding to :cpp:struct:`VQFParams`
        """
        return copy.deepcopy(self._params)

    @property
    def coeffs(self):
        """Read-only property to access the coefficients used by the algorithm.

        :return: dict with entries corresponding to :cpp:struct:`VQFCoefficients`
        """
        return copy.deepcopy(self._coeffs)

    @property
    def state(self):
        """Property to access the current state.

        This property can be written to in order to set a completely arbitrary filter state, which is intended for
        debugging purposes. However, note that the returned dict is a copy of the state and changing elements of this
        dict will not influence the actual state. In order to modify the state, access the state, change some elements
        and then replace the whole state with the modified copy, e.g.

        .. code-block::

                # does not work: v.state['delta'] = 0
                state = vqf.state
                state['delta'] = 0
                vqf.state = state

        :return: dict with entries corresponding to :cpp:struct:`VQFState`
        """
        return dataclasses.asdict(self._state)

    @state.setter
    def state(self, state):
        assert state.keys() == {f.name for f in dataclasses.fields(PyVQFState)}
        for k in state:
            assert isinstance(state[k], type(getattr(self._state, k)))
            if isinstance(state[k], np.ndarray):
                assert state[k].dtype == getattr(self._state, k).dtype
                assert state[k].shape == getattr(self._state, k).shape
        self._state = PyVQFState(**copy.deepcopy(state))

    def resetState(self):
        """Resets the state to the default values at initialization.

        Resetting the state is equivalent to creating a new instance of this class.
        """
        self._state = PyVQFState()
        self._state.biasP = self._coeffs.biasP0*np.eye(3)
        self._state.magRejectT = self._params.magMaxRejectionTime

    @staticmethod
    def quatMultiply(q1, q2):
        r"""Performs quaternion multiplication (:math:`\mathbf{q}_\mathrm{out} = \mathbf{q}_1 \otimes \mathbf{q}_2`).

        :param q1: input quaternion 1 -- numpy array with shape (4,)
        :param q2: input quaternion 2 -- numpy array with shape (4,)
        :return: output quaternion -- numpy array with shape (4,)
        """
        assert q1.shape == (4,)
        assert q2.shape == (4,)
        q10, q11, q12, q13 = q1.tolist()
        q20, q21, q22, q23 = q2.tolist()
        w = q10 * q20 - q11 * q21 - q12 * q22 - q13 * q23
        x = q10 * q21 + q11 * q20 + q12 * q23 - q13 * q22
        y = q10 * q22 - q11 * q23 + q12 * q20 + q13 * q21
        z = q10 * q23 + q11 * q22 - q12 * q21 + q13 * q20
        return np.array([w, x, y, z], float)

    @staticmethod
    def quatConj(q):
        r"""Calculates the quaternion conjugate (:math:`\mathbf{q}_\mathrm{out} = \mathbf{q}^*`).

        :param q: input quaternion -- numpy array with shape (4,)
        :return: output quaternion -- numpy array with shape (4,)
        """
        assert q.shape == (4,)
        return np.array([q[0], -q[1], -q[2], -q[3]], float)

    @staticmethod
    def quatApplyDelta(q, delta):
        r""" Applies a heading rotation by the angle delta (in rad) to a quaternion.

        :math:`\mathbf{q}_\mathrm{out} = \begin{bmatrix}\cos\frac{\delta}{2} & 0 & 0 &
        \sin\frac{\delta}{2}\end{bmatrix} \otimes \mathbf{q}`

        :param q: input quaternion -- numpy array with shape (4,)
        :param delta: heading rotation angle in rad
        :return: output quaternion -- numpy array with shape (4,)
        """
        assert q.shape == (4,)
        c = np.cos(delta/2)
        s = np.sin(delta/2)
        w = c * q[0] - s * q[3]
        x = c * q[1] - s * q[2]
        y = c * q[2] + s * q[1]
        z = c * q[3] + s * q[0]
        return np.array([w, x, y, z], float)

    @staticmethod
    def quatRotate(q, v):
        r"""Rotates a vector with a given quaternion.

        :math:`\begin{bmatrix}0 & \mathbf{v}_\mathrm{out}\end{bmatrix}
        = \mathbf{q} \otimes \begin{bmatrix}0 & \mathbf{v}\end{bmatrix} \otimes \mathbf{q}^*`

        :param q: input quaternion -- numpy array with shape (4,)
        :param v: input vector -- numpy array with shape (3,)
        :return: output vector -- numpy array with shape (3,)
        """
        assert q.shape == (4,)
        assert v.shape == (3,)
        q0, q1, q2, q3 = q.tolist()
        v0, v1, v2 = v.tolist()
        x = (1 - 2*q2*q2 - 2*q3*q3)*v0 + 2*v1*(q2*q1 - q0*q3) + 2*v2*(q0*q2 + q3*q1)
        y = 2*v0*(q0*q3 + q2*q1) + v1*(1 - 2*q1*q1 - 2*q3*q3) + 2*v2*(q2*q3 - q1*q0)
        z = 2*v0*(q3*q1 - q0*q2) + 2*v1*(q0*q1 + q3*q2) + v2*(1 - 2*q1*q1 - 2*q2*q2)
        return np.array([x, y, z], float)

    @staticmethod
    def normalize(vec):
        """Normalizes a vector in-place.

        :param vec: vector -- one-dimensional numpy array that will be normalized in-place
        :return: None
        """
        norm = math.sqrt(vec.dot(vec))
        if norm != 0.0:
            vec /= norm

    @staticmethod
    def gainFromTau(tau, Ts):
        r"""Calculates the gain for a first-order low-pass filter from the 1/e time constant.

        :math:`k = 1 - \exp\left(-\frac{T_\mathrm{s}}{\tau}\right)`

        The cutoff frequency of the resulting filter is :math:`f_\mathrm{c} = \frac{1}{2\pi\tau}`.

        :param tau: time constant :math:`\tau` in seconds - use -1 to disable update (:math:`k=0`) or 0 to obtain
            unfiltered values (:math:`k=1`)
        :param Ts: sampling time :math:`T_\mathrm{s}` in seconds
        :return: filter gain *k*
        """
        assert Ts > 0
        if tau < 0:
            return 0.0  # k=0 for negative tau (disable update)
        elif tau == 0.0:
            return 1.0  # k=1 for tau=0
        else:
            return 1 - np.exp(-Ts/tau)  # fc = 1/(2*pi*tau)

    @staticmethod
    def filterCoeffs(tau, Ts):
        r"""Calculates coefficients for a second-order Butterworth low-pass filter.

        The filter is parametrized via the time constant of the dampened, non-oscillating part of step response and the
        resulting cutoff frequency is :math:`f_\mathrm{c} = \frac{\sqrt{2}}{2\pi\tau}`.

        :param tau: time constant :math:`\tau` in seconds
        :param Ts: sampling time :math:`T_\mathrm{s}` in seconds
        :return: numerator coefficients b as (3,) numpy array, denominator coefficients a (without :math:`a_0=1`) as
            (2,) numpy array
        """
        assert tau > 0
        assert Ts > 0
        # second order Butterworth filter based on https://stackoverflow.com/a/52764064
        fc = math.sqrt(2) / (2.0 * math.pi * tau)  # time constant of dampened, non-oscillating part of step response
        C = math.tan(math.pi*fc*Ts)
        D = C**2 + math.sqrt(2)*C + 1
        b0 = C*C/D
        b1 = 2*b0
        b2 = b0
        # a0 = 1.0
        a1 = 2*(C**2-1)/D
        a2 = (1-math.sqrt(2)*C+C**2)/D
        return np.array([b0, b1, b2], float), np.array([a1, a2], float)

    @staticmethod
    def filterInitialState(x0, b, a):
        r"""Calculates the initial filter state for a given steady-state value.

        :param x0: steady state value
        :param b: numerator coefficients
        :param a: denominator coefficients (without :math:`a_0=1`)
        :return: filter state -- numpy array with shape (2,)
        """
        assert b.shape == (3,)
        assert a.shape == (2,)
        # initial state for steady state (equivalent to scipy.signal.lfilter_zi, obtained by setting y=x=x0 in the
        # filter update equation)
        return np.array([
            x0*(1 - b[0]),
            x0*(b[2] - a[1])
        ], float)

    @staticmethod
    def filterAdaptStateForCoeffChange(last_y, b_old, a_old, b_new, a_new, state):
        r"""Adjusts the filter state when changing coefficients.

        This function assumes that the filter is currently in a steady state, i.e. the last input values and the last
        output values are all equal. Based on this, the filter state is adjusted to new filter coefficients so that the
        output does not jump.

        :param last_y: last filter output values -- numpy array with shape (N,)
        :param b_old: previous numerator coefficients -- numpy array with shape (3,)
        :param a_old: previous denominator coefficients (without :math:`a_0=1`) -- numpy array with shape (2,)
        :param b_new: new numerator coefficients -- numpy array with shape (3,)
        :param a_new: new denominator coefficients (without :math:`a_0=1`) -- numpy array with shape (2,)
        :param state: filter state -- numpy array with shape (N*2,), will be modified
        :return: None
        """
        N = last_y.shape[0]
        assert last_y.shape == (N,)
        assert b_old.shape == (3,)
        assert a_old.shape == (2,)
        assert b_new.shape == (3,)
        assert a_new.shape == (2,)
        assert state.shape == (2, N)

        if math.isnan(state[0, 0]):
            return

        state[0] = state[0] + (b_old[0] - b_new[0])*last_y
        state[1] = state[1] + (b_old[1] - b_new[1] - a_old[0] + a_new[0])*last_y

    @staticmethod
    def filterStep(x, b, a, state):
        r"""Performs a filter step.

        Note: Unlike the C++ implementation, this function is vectorized and can process multiple values at once.

        :param x: input values -- numpy array with shape (N,)
        :param b: numerator coefficients -- numpy array with shape (3,)
        :param a: denominator coefficients (without :math:`a_0=1`) -- numpy array with shape (2,)
        :param state: filter state -- numpy array with shape (2, N), will be modified
        :return: filtered values -- numpy array with shape (N,)
        """
        x = np.asarray(x)
        N = x.shape[0]  # this function is vectorized, unlike the C++ version
        assert x.shape == (N,)
        assert b.shape == (3,)
        assert a.shape == (2,)
        assert state.shape == (2, N)

        # difference equations based on scipy.signal.lfilter documentation
        # assumes that a0 == 1.0
        y = b[0] * x + state[0]
        state[0] = b[1] * x - a[0] * y + state[1]
        state[1] = b[2] * x - a[1] * y

        return y

    @staticmethod
    def filterVec(x, tau, Ts, b, a, state):
        r"""Performs filter step for vector-valued signal with averaging-based initialization.

        During the first :math:`\tau` seconds, the filter output is the mean of the previous samples. At :math:`t=\tau`,
        the initial conditions for the low-pass filter are calculated based on the current mean value and from then on,
        regular filtering with the rational transfer function described by the coefficients b and a is performed.

        :param x: input values -- numpy array with shape (N,)
        :param tau: filter time constant \:math:`\tau` in seconds (used for initialization)
        :param Ts: sampling time :math:`T_\mathrm{s}` in seconds (used for initialization)
        :param b: numerator coefficients -- numpy array with shape (3,)
        :param a: denominator coefficients (without :math:`a_0=1`) -- numpy array with shape (2,)
        :param state: filter state -- numpy array with shape (2, N), will be modified
        :return: filtered values -- numpy array with shape (N,)
        """
        N = x.shape[0]
        assert N >= 2
        assert x.shape == (N,)
        assert b.shape == (3,)
        assert a.shape == (2,)
        assert state.shape == (2, N)

        # to avoid depending on a single sample, average the first samples (for duration tau)
        # and then use this average to calculate the filter initial state
        if math.isnan(state[0, 0]):  # initialization phase
            if math.isnan(state[0, 1]):  # first sample
                state[0, 1] = 0  # state[0, 1] is used to store the sample count
                state[1, :] = 0  # state[1, :] is used to store the sum

            state[0, 1] += 1
            state[1, :] += x
            out = state[1]/state[0, 1]

            if state[0, 1]*Ts >= tau:
                for i in range(N):
                    state[:, i] = PyVQF.filterInitialState(out[i], b, a)
            return out

        return PyVQF.filterStep(x, b, a, state)

    def _setup(self):
        """Calculates coefficients based on parameters and sampling rates."""
        coeffs = self._coeffs
        params = self._params

        assert coeffs.gyrTs > 0
        assert coeffs.accTs > 0
        assert coeffs.magTs > 0

        coeffs.accLpB, coeffs.accLpA = self.filterCoeffs(params.tauAcc, coeffs.accTs)

        coeffs.kMag = self.gainFromTau(params.tauMag, coeffs.magTs)

        coeffs.biasP0 = (params.biasSigmaInit*100.0)**2
        # the system noise increases the variance from 0 to (0.1 °/s)^2 in biasForgettingTime seconds
        coeffs.biasV = (0.1*100.0)**2*coeffs.accTs/params.biasForgettingTime
        pMotion = (params.biasSigmaMotion*100.0)**2
        coeffs.biasMotionW = pMotion**2 / coeffs.biasV + pMotion
        coeffs.biasVerticalW = coeffs.biasMotionW / max(params.biasVerticalForgettingFactor, 1e-10)

        pRest = (params.biasSigmaRest*100.0)**2
        coeffs.biasRestW = pRest**2 / coeffs.biasV + pRest

        coeffs.restGyrLpB, coeffs.restGyrLpA = self.filterCoeffs(params.restFilterTau, coeffs.gyrTs)
        coeffs.restAccLpB, coeffs.restAccLpA = self.filterCoeffs(params.restFilterTau, coeffs.accTs)

        coeffs.kMagRef = self.gainFromTau(params.magRefTau, coeffs.magTs)
        if params.magCurrentTau > 0:
            coeffs.magNormDipLpB, coeffs.magNormDipLpA = self.filterCoeffs(params.magCurrentTau, coeffs.magTs)

        self.resetState()
