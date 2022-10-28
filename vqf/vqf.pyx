# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

# distutils: language = c++
# cython: language_level=3
# cython: embedsignature=True
# distutils: sources = vqf/cpp/vqf.cpp vqf/cpp/offline_vqf.cpp
# distutils: undef_macros = NDEBUG

import numpy as np

from libcpp cimport bool
from libc.string cimport memcpy
cimport numpy as np
cimport cython

ctypedef double vqf_real_t
vqf_real = np.double

cdef extern from 'cpp/vqf.hpp':
    cdef struct VQFParams:
        vqf_real_t tauAcc
        vqf_real_t tauMag
        bool motionBiasEstEnabled
        bool restBiasEstEnabled
        bool magDistRejectionEnabled
        vqf_real_t biasSigmaInit
        vqf_real_t biasForgettingTime
        vqf_real_t biasClip
        vqf_real_t biasSigmaMotion
        vqf_real_t biasVerticalForgettingFactor
        vqf_real_t biasSigmaRest
        vqf_real_t restMinT
        vqf_real_t restFilterTau
        vqf_real_t restThGyr
        vqf_real_t restThAcc
        vqf_real_t magCurrentTau
        vqf_real_t magRefTau
        vqf_real_t magNormTh
        vqf_real_t magDipTh
        vqf_real_t magNewTime
        vqf_real_t magNewFirstTime
        vqf_real_t magNewMinGyr
        vqf_real_t magMinUndisturbedTime
        vqf_real_t magMaxRejectionTime
        vqf_real_t magRejectionFactor

    cdef struct VQFState:
        vqf_real_t gyrQuat[4]
        vqf_real_t accQuat[4]
        vqf_real_t delta
        bool restDetected
        bool magDistDetected
        vqf_real_t lastAccLp[3]
        double accLpState[3*2]
        vqf_real_t lastAccCorrAngularRate
        vqf_real_t kMagInit
        vqf_real_t lastMagDisAngle
        vqf_real_t lastMagCorrAngularRate
        vqf_real_t bias[3]
        vqf_real_t biasP[9]
        double motionBiasEstRLpState[9*2]
        double motionBiasEstBiasLpState[2*2]
        vqf_real_t restLastSquaredDeviations[2]
        vqf_real_t restT
        vqf_real_t restLastGyrLp[3]
        double restGyrLpState[3*2]
        vqf_real_t restLastAccLp[3]
        double restAccLpState[3*2]
        vqf_real_t magRefNorm
        vqf_real_t magRefDip
        vqf_real_t magUndisturbedT
        vqf_real_t magRejectT
        vqf_real_t magCandidateNorm
        vqf_real_t magCandidateDip
        vqf_real_t magCandidateT
        vqf_real_t magNormDip[2]
        double magNormDipLpState[2*2]

    cdef struct VQFCoefficients:
        vqf_real_t gyrTs
        vqf_real_t accTs
        vqf_real_t magTs
        double accLpB[3]
        double accLpA[2]
        vqf_real_t kMag
        vqf_real_t biasP0
        vqf_real_t biasV
        vqf_real_t biasMotionW
        vqf_real_t biasVerticalW
        vqf_real_t biasRestW
        double restGyrLpB[3]
        double restGyrLpA[2]
        double restAccLpB[3]
        double restAccLpA[2]
        vqf_real_t kMagRef
        double magNormDipLpB[3]
        double magNormDipLpA[2]

    cdef cppclass C_VQF 'VQF':
        C_VQF(vqf_real_t gyrTs, vqf_real_t accTs, vqf_real_t magTs) except +
        C_VQF(const VQFParams& params, vqf_real_t gyrTs, vqf_real_t accTs, vqf_real_t magTs) except +

        void updateGyr(const vqf_real_t gyr[3])
        void updateAcc(const vqf_real_t acc[3])
        void updateMag(const vqf_real_t mag[3])
        void update(const vqf_real_t gyr[3], const vqf_real_t acc[3])
        void update(const vqf_real_t gyr[3], const vqf_real_t acc[3], const vqf_real_t mag[3])
        void updateBatch(const vqf_real_t gyr[], const vqf_real_t acc[], const vqf_real_t mag[], size_t N,
                         vqf_real_t out6D[], vqf_real_t out9D[], vqf_real_t outDelta[], vqf_real_t outBias[],
                         vqf_real_t outBiasSigma[], bool outRest[], bool outMagDist[])

        void getQuat3D(vqf_real_t out[4]) const
        void getQuat6D(vqf_real_t out[4]) const
        void getQuat9D(vqf_real_t out[4]) const
        vqf_real_t getDelta() const

        vqf_real_t getBiasEstimate(vqf_real_t out[3]) const
        void setBiasEstimate(vqf_real_t bias[3], vqf_real_t sigma)
        bool getRestDetected() const
        void getRelativeRestDeviations(vqf_real_t out[2]) const
        bool getMagDistDetected() const
        vqf_real_t getMagRefNorm() const
        vqf_real_t getMagRefDip() const
        void setMagRef(vqf_real_t norm, vqf_real_t dip)

        void setTauAcc(vqf_real_t tauAcc)
        void setTauMag(vqf_real_t tauMag)
        void setMotionBiasEstEnabled(bool enabled)
        void setRestBiasEstEnabled(bool enabled)
        void setMagDistRejectionEnabled(bool enabled)
        void setRestDetectionThresholds(vqf_real_t thGyr, vqf_real_t thAcc)

        const VQFParams& getParams() const
        const VQFCoefficients& getCoeffs() const
        const VQFState& getState() const
        void setState(const VQFState& state)
        void resetState()

        @staticmethod
        void quatMultiply(const vqf_real_t q1[4], const vqf_real_t q2[4], vqf_real_t out[4])
        @staticmethod
        void quatConj(const vqf_real_t q[4], vqf_real_t out[4])
        @staticmethod
        void quatSetToIdentity(vqf_real_t out[4])
        @staticmethod
        void quatApplyDelta(vqf_real_t q[4], vqf_real_t delta, vqf_real_t out[4])
        @staticmethod
        void quatRotate(const vqf_real_t q[4], const vqf_real_t v[3], vqf_real_t out[3])
        @staticmethod
        vqf_real_t norm(const vqf_real_t vec[], size_t N)
        @staticmethod
        void normalize(vqf_real_t vec[], size_t N)
        @staticmethod
        void clip(vqf_real_t vec[], size_t N, vqf_real_t min_, vqf_real_t max_)
        @staticmethod
        vqf_real_t gainFromTau(vqf_real_t tau, vqf_real_t Ts)
        @staticmethod
        void filterCoeffs(vqf_real_t fc, vqf_real_t Ts, double outB[3], double outA[2])
        @staticmethod
        void filterInitialState(vqf_real_t x0, const double b[3], const double a[2], double out[2])
        @staticmethod
        void filterAdaptStateForCoeffChange(vqf_real_t last_y[], size_t N, const double b_old[3],
                                            const double a_old[2], const double b_new[3],
                                            const double a_new[2], double state[])
        @staticmethod
        vqf_real_t filterStep(vqf_real_t x, const double b[3], const double a[2], double state[2])
        @staticmethod
        void filterVec(const vqf_real_t x[], size_t N, vqf_real_t tau, vqf_real_t Ts, const double b[3],
                       const double a[2], double state[], vqf_real_t out[])
        @staticmethod
        void matrix3SetToScaledIdentity(vqf_real_t scale, vqf_real_t out[9])
        @staticmethod
        void matrix3Multiply(const vqf_real_t in1[9], const vqf_real_t in2[9], vqf_real_t out[9])
        @staticmethod
        void matrix3MultiplyTpsFirst(const vqf_real_t in1[9], const vqf_real_t in2[9], vqf_real_t out[9])
        @staticmethod
        void matrix3MultiplyTpsSecond(const vqf_real_t in1[9], const vqf_real_t in2[9], vqf_real_t out[9])
        @staticmethod
        bool matrix3Inv(const vqf_real_t in_[9], vqf_real_t out[9])


cdef class VQF:
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

    If the full data is available in numpy arrays, you can use :meth:`updateBatch` and :meth:`updateBatchFullState`.

    This class is a Python wrapper (implemented in `Cython <https://cython.org/>`_) for the C++ implementation
    :cpp:class:`VQF`. Depending on use case and programming language of choice, the following alternatives might be
    useful:

    +------------------------+---------------------------+--------------------------+---------------------------+
    |                        | Full Version              | Basic Version            | Offline Version           |
    |                        |                           |                          |                           |
    +========================+===========================+==========================+===========================+
    | **C++**                | :cpp:class:`VQF`          | :cpp:class:`BasicVQF`    | :cpp:func:`offlineVQF`    |
    +------------------------+---------------------------+--------------------------+---------------------------+
    | **Python/C++ (fast)**  | **vqf.VQF (this class)**  | :py:class:`vqf.BasicVQF` | :py:meth:`vqf.offlineVQF` |
    +------------------------+---------------------------+--------------------------+---------------------------+
    | **Pure Python (slow)** | :py:class:`vqf.PyVQF`     | --                       | --                        |
    +------------------------+---------------------------+--------------------------+---------------------------+
    | **Pure Matlab (slow)** | :mat:class:`VQF.m <VQF>`  | --                       | --                        |
    +------------------------+---------------------------+--------------------------+---------------------------+

    In the most common case (using the default parameters and all data being sampled with the same frequency, create the
    class like this:

    .. code-block::

         from vqf import VQF
         vqf = VQF(0.01)  # 0.01 s sampling time, i.e. 100 Hz

    Example code to create an object with magnetic disturbance rejection disabled (use the `**-operator
    <https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists>`_ to pass  parameters from a dict):

    .. code-block::

        from vqf import VQF
        vqf = VQF(0.01, magDistRejectionEnabled=False)  # 0.01 s sampling time, i.e. 100 Hz

    See :cpp:struct:`VQFParams` for a detailed description of all parameters.
    """
    cdef C_VQF* c_obj

    def __cinit__(self, gyrTs, accTs=-1.0, magTs=-1.0, tauAcc=None, tauMag=None,
                  motionBiasEstEnabled=None, restBiasEstEnabled=None, magDistRejectionEnabled=None,
                  biasSigmaInit=None, biasForgettingTime=None, biasClip=None,
                  biasSigmaMotion=None, biasVerticalForgettingFactor=None,
                  biasSigmaRest=None, restMinT=None, restFilterTau=None, restThGyr=None, restThAcc=None,
                  magCurrentTau=None, magRefTau=None, magNormTh=None, magDipTh=None, magNewTime=None,
                  magNewFirstTime=None, magNewMinGyr=None, magMinUndisturbedTime=None, magMaxRejectionTime=None,
                  magRejectionFactor=None):
        cdef VQFParams params
        if tauAcc is not None:
            params.tauAcc = tauAcc
        if tauMag is not None:
            params.tauMag = tauMag
        if motionBiasEstEnabled is not None:
            params.motionBiasEstEnabled = motionBiasEstEnabled
        if restBiasEstEnabled is not None:
            params.restBiasEstEnabled = restBiasEstEnabled
        if magDistRejectionEnabled is not None:
            params.magDistRejectionEnabled = magDistRejectionEnabled
        if biasSigmaInit is not None:
            params.biasSigmaInit = biasSigmaInit
        if biasForgettingTime is not None:
            params.biasForgettingTime = biasForgettingTime
        if biasClip is not None:
            params.biasClip = biasClip
        if biasSigmaMotion is not None:
            params.biasSigmaMotion = biasSigmaMotion
        if biasVerticalForgettingFactor is not None:
            params.biasVerticalForgettingFactor = biasVerticalForgettingFactor
        if biasSigmaRest is not None:
            params.biasSigmaRest = biasSigmaRest
        if restMinT is not None:
            params.restMinT = restMinT
        if restFilterTau is not None:
            params.restFilterTau = restFilterTau
        if restThGyr is not None:
            params.restThGyr = restThGyr
        if restThAcc is not None:
            params.restThAcc = restThAcc
        if magCurrentTau is not None:
            params.magCurrentTau = magCurrentTau
        if magRefTau is not None:
            params.magRefTau = magRefTau
        if magNormTh is not None:
            params.magNormTh = magNormTh
        if magDipTh is not None:
            params.magDipTh = magDipTh
        if magNewTime is not None:
            params.magNewTime = magNewTime
        if magNewFirstTime is not None:
            params.magNewFirstTime = magNewFirstTime
        if magNewMinGyr is not None:
            params.magNewMinGyr = magNewMinGyr
        if magMinUndisturbedTime is not None:
            params.magMinUndisturbedTime = magMinUndisturbedTime
        if magMaxRejectionTime is not None:
            params.magMaxRejectionTime = magMaxRejectionTime
        if magRejectionFactor is not None:
            params.magRejectionFactor = magRejectionFactor

        self.c_obj = new C_VQF(params, <vqf_real_t> gyrTs, <vqf_real_t> accTs, <vqf_real_t> magTs)

    # dummy function to get at least some documentation, cf. https://stackoverflow.com/a/42733794/1971363
    def __init__(self, gyrTs, accTs=-1.0, magTs=-1.0, tauAcc=None, tauMag=None,
                 motionBiasEstEnabled=None, restBiasEstEnabled=None, magDistRejectionEnabled=None,
                 biasSigmaInit=None, biasForgettingTime=None, biasClip=None,
                 biasSigmaMotion=None, biasVerticalForgettingFactor=None,
                 biasSigmaRest=None, restMinT=None, restFilterTau=None, restThGyr=None, restThAcc=None,
                 magCurrentTau=None, magRefTau=None, magNormTh=None, magDipTh=None, magNewTime=None,
                 magNewFirstTime=None, magNewMinGyr=None, magMinUndisturbedTime=None, magMaxRejectionTime=None,
                 magRejectionFactor=None):
        """
        :param gyrTs: sampling time of the gyroscope measurements in seconds
        :param accTs: sampling time of the accelerometer measurements in seconds
            (the value of `gyrTs` is used if set to -1)
        :param magTs: sampling time of the magnetometer measurements in seconds
            (the value of `gyrTs` is used if set to -1)
        :param (params): optional parameters to override the defaults
            (see :cpp:struct:`VQFParams` for a full list and detailed descriptions)
        """
        pass

    def __dealloc__(self):
        del self.c_obj

    def updateGyr(self, np.ndarray[vqf_real_t, ndim=1, mode='c'] gyr not None):
        """Performs gyroscope update step.

        It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
        different sampling rates. Otherwise, simply use :meth:`update`.

        :param gyr: gyroscope measurement in rad/s -- numpy array with shape (3,)
        :return: None
        """
        assert gyr.shape[0] == 3
        self.c_obj.updateGyr(<vqf_real_t*> np.PyArray_DATA(gyr))

    def updateAcc(self, np.ndarray[vqf_real_t, ndim=1, mode='c'] acc not None):
        """Performs accelerometer update step.

        It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
        different sampling rates. Otherwise, simply use :meth:`update`.

        Should be called after :meth:`updateGyr` and before :meth:`updateMag`.

        :param acc: accelerometer measurement in m/s² -- numpy array with shape (3,)
        :return: None
        """
        assert acc.shape[0] == 3
        self.c_obj.updateAcc(<vqf_real_t*> np.PyArray_DATA(acc))

    def updateMag(self, np.ndarray[vqf_real_t, ndim=1, mode='c'] mag not None):
        """Performs magnetometer update step.

        It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
        different sampling rates. Otherwise, simply use :meth:`update`.

        Should be called after :meth:`updateAcc`.

        :param mag: magnetometer measurement in arbitrary units -- numpy array with shape (3,)
        :return: None
        """
        assert mag.shape[0] == 3
        self.c_obj.updateMag(<vqf_real_t*> np.PyArray_DATA(mag))

    def update(self, np.ndarray[vqf_real_t, ndim=1, mode='c'] gyr not None,
               np.ndarray[vqf_real_t, ndim=1, mode='c'] acc not None,
               np.ndarray[vqf_real_t, ndim=1, mode='c'] mag=None):
        """Performs filter update step for one sample.

        :param gyr: gyr gyroscope measurement in rad/s -- numpy array with shape (3,)
        :param acc: acc accelerometer measurement in m/s² -- numpy array with shape (3,)
        :param mag: optional mag magnetometer measurement in arbitrary units -- numpy array with shape (3,)
        :return: None
        """
        assert gyr.shape[0] == 3
        assert acc.shape[0] == 3
        if mag is not None:
            assert mag.shape[0] == 3
            self.c_obj.update(<vqf_real_t*> np.PyArray_DATA(gyr), <vqf_real_t*> np.PyArray_DATA(acc),
                              <vqf_real_t*> np.PyArray_DATA(mag))
        else:
            self.c_obj.update(<vqf_real_t*> np.PyArray_DATA(gyr), <vqf_real_t*> np.PyArray_DATA(acc))

    def updateBatch(self, np.ndarray[vqf_real_t, ndim=2, mode='c'] gyr not None,
                    np.ndarray[vqf_real_t, ndim=2, mode='c'] acc not None,
                    np.ndarray[vqf_real_t, ndim=2, mode='c'] mag=None):
        """Performs batch update for multiple samples at once.

        In order to use this function, all input data must have the same sampling rate and be provided as a
        contiguous numpy array. As looping over the samples is performed in compiled
        code, it is much faster than using a Python loop. The output is a dictionary containing

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
        cdef int N = gyr.shape[0]
        assert acc.shape[0] == N
        assert gyr.shape[1] == 3
        assert acc.shape[1] == 3

        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out6D = np.zeros(shape=(N, 4), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out9D
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] outDelta
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] outBias = np.zeros(shape=(N, 3), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] outBiasSigma = np.zeros(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[bool, ndim=1, mode='c'] outRest = np.zeros(shape=(N,), dtype=np.bool_)
        cdef np.ndarray[bool, ndim=1, mode='c'] outMagDist

        if mag is not None:
            assert mag.shape[0] == N
            assert mag.shape[1] == 3
            out9D = np.zeros(shape=(N, 4), dtype=vqf_real)
            outDelta = np.zeros(shape=(N,), dtype=vqf_real)
            outMagDist = np.zeros(shape=(N,), dtype=np.bool_)
            self.c_obj.updateBatch(<vqf_real_t*> np.PyArray_DATA(gyr), <vqf_real_t*> np.PyArray_DATA(acc),
                                   <vqf_real_t*> np.PyArray_DATA(mag), N,
                                   <vqf_real_t*> np.PyArray_DATA(out6D), <vqf_real_t*> np.PyArray_DATA(out9D),
                                   <vqf_real_t *> np.PyArray_DATA(outDelta),
                                   <vqf_real_t*> np.PyArray_DATA(outBias), <vqf_real_t*> np.PyArray_DATA(outBiasSigma),
                                   <bool*> np.PyArray_DATA(outRest), <bool*> np.PyArray_DATA(outMagDist))
            return dict(quat6D=out6D, quat9D=out9D, delta=outDelta, bias=outBias, biasSigma=outBiasSigma,
                        restDetected=outRest, magDistDetected=outMagDist)
        else:
            self.c_obj.updateBatch(<vqf_real_t*> np.PyArray_DATA(gyr), <vqf_real_t*> np.PyArray_DATA(acc), NULL, N,
                                   <vqf_real_t*> np.PyArray_DATA(out6D), NULL, NULL,
                                   <vqf_real_t*> np.PyArray_DATA(outBias), <vqf_real_t *> np.PyArray_DATA(outBiasSigma),
                                   <bool*> np.PyArray_DATA(outRest), NULL)
            return dict(quat6D=out6D, bias=outBias, biasSigma=outBiasSigma, restDetected=outRest)

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def updateBatchFullState(self, np.ndarray[vqf_real_t, ndim=2, mode='c'] gyr not None,
                             np.ndarray[vqf_real_t, ndim=2, mode='c'] acc not None,
                             np.ndarray[vqf_real_t, ndim=2, mode='c'] mag=None):
        """Performs batch update and returns full state.

        Works similar to :meth:`updateBatch` but returns a dictionary containing **quat6D**, **quat9D** and every value
        of :attr:`state` at every sampling step in a numpy array. As looping over the samples is performed in compiled
        code, it is much faster than using a Python loop.

        :param gyr: gyroscope measurement in rad/s -- numpy array with shape (N,3)
        :param acc: accelerometer measurement in m/s² -- numpy array with shape (N,3)
        :param mag: optional magnetometer measurement in arbitrary units -- numpy array with shape (N,3)
        :return: dict with full state as numpy array
        """
        cdef int N = gyr.shape[0]
        assert acc.shape[0] == N
        assert gyr.shape[1] == 3
        assert acc.shape[1] == 3
        if mag is not None:
            assert mag.shape[0] == N
            assert mag.shape[1] == 3

        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out6D = np.empty(shape=(N, 4), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out9D = np.empty(shape=(N, 4), dtype=vqf_real)

        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] gyrQuat = np.empty(shape=(N, 4), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] accQuat = np.empty(shape=(N, 4), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] delta = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[bool, ndim=1, mode='c'] restDetected = np.empty(shape=(N,), dtype=np.bool_)
        cdef np.ndarray[bool, ndim=1, mode='c'] magDistDetected = np.empty(shape=(N,), dtype=np.bool_)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] lastAccLp = np.empty(shape=(N, 3), dtype=vqf_real)
        cdef np.ndarray[double, ndim=2, mode='c'] accLpState = np.empty(shape=(N, 3*2), dtype=np.float64)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] lastAccCorrAngularRate = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] kMagInit = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] lastMagDisAngle = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] lastMagCorrAngularRate = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] bias = np.empty(shape=(N, 3), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] biasP = np.empty(shape=(N, 9), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] biasSigma = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[double, ndim=2, mode='c'] motionBiasEstRLpState = np.empty(shape=(N, 9*2), dtype=np.float64)
        cdef np.ndarray[double, ndim=2, mode='c'] motionBiasEstBiasLpState = np.empty(shape=(N, 2*2), dtype=np.float64)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] restLastSquaredDeviations = np.empty(shape=(N, 2), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] relativeRestDeviations = np.empty(shape=(N, 2), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] restT = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] restLastGyrLp = np.empty(shape=(N, 3), dtype=vqf_real)
        cdef np.ndarray[double, ndim=2, mode='c'] restGyrLpState = np.empty(shape=(N, 3*2), dtype=np.float64)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] restLastAccLp = np.empty(shape=(N, 3), dtype=vqf_real)
        cdef np.ndarray[double, ndim=2, mode='c'] restAccLpState = np.empty(shape=(N, 3*2), dtype=np.float64)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] magRefNorm = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] magRefDip = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] magUndisturbedT = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] magRejectT = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] magCandidateNorm = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] magCandidateDip = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] magCandidateT = np.empty(shape=(N,), dtype=vqf_real)
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] magNormDip = np.empty(shape=(N, 2), dtype=vqf_real)
        cdef np.ndarray[double, ndim=2, mode='c'] magNormDipLpState = np.empty(shape=(N, 2*2), dtype=np.float64)

        cdef VQFState state
        cdef int i = 0
        for i in range(N):
            if mag is not None:
                self.c_obj.update((<vqf_real_t*> np.PyArray_DATA(gyr))+3*i,
                                  (<vqf_real_t*> np.PyArray_DATA(acc))+3*i,
                                  (<vqf_real_t*> np.PyArray_DATA(mag))+3*i)
            else:
                self.c_obj.update((<vqf_real_t*> np.PyArray_DATA(gyr))+3*i,
                                  (<vqf_real_t*> np.PyArray_DATA(acc))+3*i)

            self.c_obj.getQuat6D((<vqf_real_t*> np.PyArray_DATA(out6D))+4*i)
            self.c_obj.getQuat9D((<vqf_real_t*> np.PyArray_DATA(out9D))+4*i)

            state = self.c_obj.getState()
            memcpy(<vqf_real_t*> np.PyArray_DATA(gyrQuat)+4*i, state.gyrQuat, 4*sizeof(vqf_real_t))
            memcpy(<vqf_real_t*> np.PyArray_DATA(accQuat)+4*i, state.accQuat, 4*sizeof(vqf_real_t))
            delta[i] = state.delta
            restDetected[i] = state.restDetected
            magDistDetected[i] = state.magDistDetected
            memcpy(<vqf_real_t*> np.PyArray_DATA(lastAccLp)+3*i, state.lastAccLp, 3*sizeof(vqf_real_t))
            memcpy(<double*> np.PyArray_DATA(accLpState)+3*2*i, state.accLpState, 3*2*sizeof(double))
            lastAccCorrAngularRate[i] = state.lastAccCorrAngularRate
            kMagInit[i] = state.kMagInit
            lastMagDisAngle[i] = state.lastMagDisAngle
            lastMagCorrAngularRate[i] = state.lastMagCorrAngularRate
            memcpy(<vqf_real_t*> np.PyArray_DATA(bias)+3*i, state.bias, 3*sizeof(vqf_real_t))
            memcpy(<vqf_real_t*> np.PyArray_DATA(biasP)+9*i, state.biasP, 9*sizeof(vqf_real_t))
            biasSigma[i] = self.c_obj.getBiasEstimate(NULL)
            memcpy(<double*> np.PyArray_DATA(motionBiasEstRLpState)+9*2*i, state.motionBiasEstRLpState,
                   9*2*sizeof(double))
            memcpy(<double*> np.PyArray_DATA(motionBiasEstBiasLpState)+2*2*i, state.motionBiasEstBiasLpState,
                   2*2*sizeof(double))
            memcpy(<vqf_real_t*> np.PyArray_DATA(restLastSquaredDeviations)+2*i, state.restLastSquaredDeviations,
                   2*sizeof(vqf_real_t))
            self.c_obj.getRelativeRestDeviations((<vqf_real_t *> np.PyArray_DATA(relativeRestDeviations)) + 2*i)
            restT[i] = state.restT
            memcpy(<vqf_real_t*> np.PyArray_DATA(restLastGyrLp)+3*i, state.restLastGyrLp, 3*sizeof(vqf_real_t))
            memcpy(<double*> np.PyArray_DATA(restGyrLpState)+3*2*i, state.restGyrLpState, 3*2*sizeof(double))
            memcpy(<vqf_real_t*> np.PyArray_DATA(restLastAccLp)+3*i, state.restLastAccLp, 3*sizeof(vqf_real_t))
            memcpy(<double*> np.PyArray_DATA(restAccLpState)+3*2*i, state.restAccLpState, 3*2*sizeof(double))
            magRefNorm[i] = state.magRefNorm
            magRefDip[i] = state.magRefDip
            magUndisturbedT[i] = state.magUndisturbedT
            magRejectT[i] = state.magRejectT
            magCandidateNorm[i] = state.magCandidateNorm
            magCandidateDip[i] = state.magCandidateDip
            magCandidateT[i] = state.magCandidateT
            memcpy(<vqf_real_t*> np.PyArray_DATA(magNormDip)+2*i, state.magNormDip, 2*sizeof(vqf_real_t))
            memcpy(<double*> np.PyArray_DATA(magNormDipLpState)+2*2*i, state.magNormDipLpState, 2*2*sizeof(double))

        return dict(
            quat6D=out6D,
            quat9D=out9D,
            gyrQuat=gyrQuat,
            accQuat=accQuat,
            delta=delta,
            restDetected=restDetected,
            magDistDetected=magDistDetected,
            lastAccLp=lastAccLp,
            accLpState=accLpState,
            lastAccCorrAngularRate=lastAccCorrAngularRate,
            kMagInit=kMagInit,
            lastMagDisAngle=lastMagDisAngle,
            lastMagCorrAngularRate=lastMagCorrAngularRate,
            bias=bias,
            biasP=biasP,
            biasSigma=biasSigma,
            motionBiasEstRLpState=motionBiasEstRLpState,
            motionBiasEstBiasLpState=motionBiasEstBiasLpState,
            restLastSquaredDeviations=restLastSquaredDeviations,
            relativeRestDeviations=relativeRestDeviations,
            restT=restT,
            restLastGyrLp=restLastGyrLp,
            restGyrLpState=restGyrLpState,
            restLastAccLp=restLastAccLp,
            restAccLpState=restAccLpState,
            magRefNorm=magRefNorm,
            magRefDip=magRefDip,
            magUndisturbedT=magUndisturbedT,
            magRejectT=magRejectT,
            magCandidateNorm=magCandidateNorm,
            magCandidateDip=magCandidateDip,
            magCandidateT=magCandidateT,
            magNormDip=magNormDip,
            magNormDipLpState=magNormDipLpState,
        )

    def getQuat3D(self):
        r"""Returns the angular velocity strapdown integration quaternion
        :math:`^{\mathcal{S}_i}_{\mathcal{I}_i}\mathbf{q}`.

        :return: quaternion as numpy array with shape (4,)
        """
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] out = np.empty(shape=(4,), dtype=vqf_real)
        self.c_obj.getQuat3D(<vqf_real_t*> np.PyArray_DATA(out))
        return out

    def getQuat6D(self):
        r"""Returns the 6D (magnetometer-free) orientation quaternion
        :math:`^{\mathcal{S}_i}_{\mathcal{E}_i}\mathbf{q}`.

        :return: quaternion as numpy array with shape (4,)
        """
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] out = np.empty(shape=(4,), dtype=vqf_real)
        self.c_obj.getQuat6D(<vqf_real_t*> np.PyArray_DATA(out))
        return out

    def getQuat9D(self):
        r"""Returns the 9D (with magnetometers) orientation quaternion
        :math:`^{\mathcal{S}_i}_{\mathcal{E}}\mathbf{q}`.

        :return: quaternion as numpy array with shape (4,)
        """
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] out = np.empty(shape=(4,), dtype=vqf_real)
        self.c_obj.getQuat9D(<vqf_real_t*> np.PyArray_DATA(out))
        return out

    def getDelta(self):
        r""" Returns the heading difference :math:`\delta` between :math:`\mathcal{E}_i` and :math:`\mathcal{E}`.

        :math:`^{\mathcal{E}_i}_{\mathcal{E}}\mathbf{q} = \begin{bmatrix}\cos\frac{\delta}{2} & 0 & 0 &
        \sin\frac{\delta}{2}\end{bmatrix}^T`.

        :return: delta angle in rad (:cpp:member:`VQFState::delta`)
        """
        return self.c_obj.getDelta()

    def getBiasEstimate(self):
        """Returns the current gyroscope bias estimate and the uncertainty.

        The returned standard deviation sigma represents the estimation uncertainty in the worst direction and is based
        on an upper bound of the largest eigenvalue of the covariance matrix.

        :return: gyroscope bias estimate (rad/s) as (3,) numpy array and standard deviation sigma of the estimation
            uncertainty (rad/s)
        """
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] bias = np.empty(shape=(3,), dtype=vqf_real)
        sigma = self.c_obj.getBiasEstimate(<vqf_real_t*> np.PyArray_DATA(bias))
        return bias, sigma

    def setBiasEstimate(self, np.ndarray[vqf_real_t, ndim=1, mode='c'] bias not None, vqf_real_t sigma=-1.0):
        """Sets the current gyroscope bias estimate and the uncertainty.

        If a value for the uncertainty sigma is given, the covariance matrix is set to a corresponding scaled identity
        matrix.

        :param bias: gyroscope bias estimate (rad/s)
        :param sigma: standard deviation of the estimation uncertainty (rad/s) - set to -1 (default) in order to not
            change the estimation covariance matrix
        """
        assert bias.shape[0] == 3
        self.c_obj.setBiasEstimate(<vqf_real_t*> np.PyArray_DATA(bias), <vqf_real_t> sigma)

    def getRestDetected(self):
        """Returns true if rest was detected."""
        return self.c_obj.getRestDetected()

    def getMagDistDetected(self):
        """Returns true if a disturbed magnetic field was detected."""
        return self.c_obj.getMagDistDetected()

    def getRelativeRestDeviations(self):
        """Returns the relative deviations used in rest detection.

         Looking at those values can be useful to understand how rest detection is working and which thresholds are
         suitable. The output array is filled with the last values for gyroscope and accelerometer,
         relative to the threshold. In order for rest to be detected, both values must stay below 1.

        :return: relative rest deviations as (2,) numpy array
        """
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] out = np.empty(shape=(2,), dtype=vqf_real)
        self.c_obj.getRelativeRestDeviations(<vqf_real_t *> np.PyArray_DATA(out))
        return out

    def getMagRefNorm(self):
        """Returns the norm of the currently accepted magnetic field reference."""
        return self.c_obj.getMagRefNorm()

    def getMagRefDip(self):
        """Returns the dip angle of the currently accepted magnetic field reference."""
        return self.c_obj.getMagRefDip()

    def setMagRef(self, norm, dip):
        """Overwrites the current magnetic field reference.

        :param norm: norm of the magnetic field reference
        :param dip: dip angle of the magnetic field reference
        """
        self.c_obj.setMagRef(<vqf_real_t> norm, <vqf_real_t> dip)

    def setTauAcc(self, tauAcc):
        r"""Sets the time constant for accelerometer low-pass filtering.

        For more details, see :cpp:member:`VQFParams::tauAcc`.

        :param tauAcc: time constant :math:`\tau_\mathrm{acc}` in seconds
        """
        self.c_obj.setTauAcc(tauAcc)

    def setTauMag(self, tauMag):
        r"""Sets the time constant for the magnetometer update.

        For more details, see :cpp:member:`VQFParams::tauMag`.

        :param tauMag: time constant :math:`\tau_\mathrm{mag}` in seconds
        """
        self.c_obj.setTauMag(tauMag)

    def setMotionBiasEstEnabled(self, enabled):
        """Enables/disabled gyroscope bias estimation during motion."""
        self.c_obj.setMotionBiasEstEnabled(enabled)

    def setRestBiasEstEnabled(self, enabled):
        """Enables/disables rest detection and bias estimation during rest."""
        self.c_obj.setRestBiasEstEnabled(enabled)

    def setMagDistRejectionEnabled(self, enabled):
        """Enables/disables magnetic disturbance detection and rejection."""
        self.c_obj.setMagDistRejectionEnabled(enabled)

    def setRestDetectionThresholds(self, thGyr, thAcc):
        """Sets the current thresholds for rest detection.

        :param thGyr: new value for :cpp:member:`VQFParams::restThGyr`
        :param thAcc: new value for :cpp:member:`VQFParams::restThAcc`
        """
        self.c_obj.setRestDetectionThresholds(thGyr, thAcc)

    @property
    def params(self):
        """Read-only property to access the current parameters.

        :return: dict with entries corresponding to :cpp:struct:`VQFParams`
        """
        return self.c_obj.getParams()

    @property
    def coeffs(self):
        """Read-only property to access the coefficients used by the algorithm.

        :return: dict with entries corresponding to :cpp:struct:`VQFCoefficients`
        """
        return self.c_obj.getCoeffs()

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
        return self.c_obj.getState()

    @state.setter
    def state(self, dict val):
        val = val.copy()

        cdef VQFState state
        state.gyrQuat = val.pop('gyrQuat')
        state.accQuat = val.pop('accQuat')
        state.delta = val.pop('delta')
        state.restDetected = val.pop('restDetected')
        state.magDistDetected = val.pop('magDistDetected')
        state.lastAccLp = val.pop('lastAccLp')
        state.accLpState = val.pop('accLpState')
        state.lastAccCorrAngularRate = val.pop('lastAccCorrAngularRate')
        state.kMagInit = val.pop('kMagInit')
        state.lastMagDisAngle = val.pop('lastMagDisAngle')
        state.lastMagCorrAngularRate = val.pop('lastMagCorrAngularRate')
        state.bias = val.pop('bias')
        state.biasP = val.pop('biasP')
        state.motionBiasEstRLpState = val.pop('motionBiasEstRLpState')
        state.motionBiasEstBiasLpState = val.pop('motionBiasEstBiasLpState')
        state.restLastSquaredDeviations = val.pop('restLastSquaredDeviations')
        state.restT = val.pop('restT')
        state.restLastGyrLp = val.pop('restLastGyrLp')
        state.restGyrLpState = val.pop('restGyrLpState')
        state.restLastAccLp = val.pop('restLastAccLp')
        state.restAccLpState = val.pop('restAccLpState')
        state.magRefNorm = val.pop('magRefNorm')
        state.magRefDip = val.pop('magRefDip')
        state.magUndisturbedT = val.pop('magUndisturbedT')
        state.magRejectT = val.pop('magRejectT')
        state.magCandidateNorm = val.pop('magCandidateNorm')
        state.magCandidateDip = val.pop('magCandidateDip')
        state.magCandidateT = val.pop('magCandidateT')
        state.magNormDip = val.pop('magNormDip')
        state.magNormDipLpState = val.pop('magNormDipLpState')

        assert len(val) == 0, 'invalid keys passed: ' + str(val.keys())

        self.c_obj.setState(state)

    def resetState(self):
        """Resets the state to the default values at initialization.

        Resetting the state is equivalent to creating a new instance of this class.
        """
        self.c_obj.resetState()

    @staticmethod
    def quatMultiply(np.ndarray[vqf_real_t, ndim=1, mode='c'] q1 not None,
                     np.ndarray[vqf_real_t, ndim=1, mode='c'] q2 not None):
        r"""Performs quaternion multiplication (:math:`\mathbf{q}_\mathrm{out} = \mathbf{q}_1 \otimes \mathbf{q}_2`).

        :param q1: input quaternion 1 -- numpy array with shape (4,)
        :param q2: input quaternion 2 -- numpy array with shape (4,)
        :return: output quaternion -- numpy array with shape (4,)
        """
        assert q1.shape[0] == 4
        assert q2.shape[0] == 4
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] out = np.empty(shape=(4,), dtype=vqf_real)
        C_VQF.quatMultiply(<vqf_real_t*> np.PyArray_DATA(q1), <vqf_real_t*> np.PyArray_DATA(q2),
                           <vqf_real_t*> np.PyArray_DATA(out))
        return out

    @staticmethod
    def quatConj(np.ndarray[vqf_real_t, ndim=1, mode='c'] q not None):
        r"""Calculates the quaternion conjugate (:math:`\mathbf{q}_\mathrm{out} = \mathbf{q}^*`).

        :param q: input quaternion -- numpy array with shape (4,)
        :return: output quaternion -- numpy array with shape (4,)
        """
        assert q.shape[0] == 4
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] out = np.empty(shape=(4,), dtype=vqf_real)
        C_VQF.quatConj(<vqf_real_t*> np.PyArray_DATA(q), <vqf_real_t*> np.PyArray_DATA(out))
        return out

    @staticmethod
    def quatSetToIdentity(np.ndarray[vqf_real_t, ndim=1, mode='c'] out not None):
        r"""Sets the output quaternion to the identity quaternion
        (:math:`\mathbf{q}_\mathrm{out} = \begin{bmatrix}1 & 0 & 0 & 0\end{bmatrix}`).

        :param out: output array that will be modified -- numpy array with shape (4,)
        :return: None
        """
        assert out.shape[0] == 4
        C_VQF.quatSetToIdentity(<vqf_real_t*> np.PyArray_DATA(out))

    @staticmethod
    def quatApplyDelta(np.ndarray[vqf_real_t, ndim=1, mode='c'] q not None, vqf_real_t delta):
        r""" Applies a heading rotation by the angle delta (in rad) to a quaternion.

        :math:`\mathbf{q}_\mathrm{out} = \begin{bmatrix}\cos\frac{\delta}{2} & 0 & 0 &
        \sin\frac{\delta}{2}\end{bmatrix} \otimes \mathbf{q}`

        :param q: input quaternion -- numpy array with shape (4,)
        :param delta: heading rotation angle in rad
        :return: output quaternion -- numpy array with shape (4,)
        """
        assert q.shape[0] == 4
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] out = np.empty(shape=(4,), dtype=vqf_real)
        C_VQF.quatApplyDelta(<vqf_real_t*> np.PyArray_DATA(q), <vqf_real_t> delta, <vqf_real_t*> np.PyArray_DATA(out))
        return out

    @staticmethod
    def quatRotate(np.ndarray[vqf_real_t, ndim=1, mode='c'] q not None,
                   np.ndarray[vqf_real_t, ndim=1, mode='c'] v not None):
        r"""Rotates a vector with a given quaternion.

        :math:`\begin{bmatrix}0 & \mathbf{v}_\mathrm{out}\end{bmatrix}
        = \mathbf{q} \otimes \begin{bmatrix}0 & \mathbf{v}\end{bmatrix} \otimes \mathbf{q}^*`

        :param q: input quaternion -- numpy array with shape (4,)
        :param v: input vector -- numpy array with shape (3,)
        :return: output vector -- numpy array with shape (3,)
        """
        assert q.shape[0] == 4
        assert v.shape[0] == 3
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] out = np.empty(shape=(3,), dtype=vqf_real)
        C_VQF.quatRotate(<vqf_real_t*> np.PyArray_DATA(q), <vqf_real_t*> np.PyArray_DATA(v),
                         <vqf_real_t*> np.PyArray_DATA(out))
        return out

    @staticmethod
    def norm(np.ndarray[vqf_real_t, ndim=1, mode='c'] vec not None):
        """Calculates the Euclidean norm of a vector.

        :param vec: input vector -- one-dimensional numpy array
        :return: Euclidean norm of the input vector
        """
        cdef size_t N = vec.shape[0]
        return C_VQF.norm(<vqf_real_t*> np.PyArray_DATA(vec), <size_t> N)

    @staticmethod
    def normalize(np.ndarray[vqf_real_t, ndim=1, mode='c'] vec not None):
        """Normalizes a vector in-place.

        :param vec: vector -- one-dimensional numpy array that will be normalized in-place
        :return: None
        """
        cdef size_t N = vec.shape[0]
        C_VQF.normalize(<vqf_real_t*> np.PyArray_DATA(vec), <size_t> N)

    @staticmethod
    def clip(np.ndarray[vqf_real_t, ndim=1, mode='c'] vec not None, min_, max_):
        """Clips a vector in-place.

        :param vec: vector -- one-dimensional numpy array that will be clipped in-place
        :param min_: smallest allowed value
        :param max_: largest allowed value
        :return: None
        """
        cdef size_t N = vec.shape[0]
        C_VQF.clip(<vqf_real_t*> np.PyArray_DATA(vec), <size_t> N, <vqf_real_t> min_, <vqf_real_t> max_)

    @staticmethod
    def gainFromTau(vqf_real_t tau, vqf_real_t Ts):
        r"""Calculates the gain for a first-order low-pass filter from the 1/e time constant.

        :math:`k = 1 - \exp\left(-\frac{T_\mathrm{s}}{\tau}\right)`

        The cutoff frequency of the resulting filter is :math:`f_\mathrm{c} = \frac{1}{2\pi\tau}`.

        :param tau: time constant :math:`\tau` in seconds - use -1 to disable update (:math:`k=0`) or 0 to obtain
            unfiltered values (:math:`k=1`)
        :param Ts: sampling time :math:`T_\mathrm{s}` in seconds
        :return: filter gain *k*
        """
        return C_VQF.gainFromTau(<vqf_real_t> tau, <vqf_real_t> Ts)

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
        cdef np.ndarray[double, ndim=1, mode='c'] outB = np.empty(shape=(3,), dtype=np.float64)
        cdef np.ndarray[double, ndim=1, mode='c'] outA = np.empty(shape=(2,), dtype=np.float64)
        C_VQF.filterCoeffs(<vqf_real_t> tau, <vqf_real_t> Ts, <double*> np.PyArray_DATA(outB),
                           <double*> np.PyArray_DATA(outA))
        return outB, outA

    @staticmethod
    def filterInitialState(x0, np.ndarray[double, ndim=1, mode='c'] b not None,
                           np.ndarray[double, ndim=1, mode='c'] a not None):
        r"""Calculates the initial filter state for a given steady-state value.

        :param x0: steady state value
        :param b: numerator coefficients
        :param a: denominator coefficients (without :math:`a_0=1`)
        :return: filter state -- numpy array with shape (2,)
        """
        assert b.shape[0] == 3
        assert a.shape[0] == 2
        cdef np.ndarray[double, ndim=1, mode='c'] out = np.empty(shape=(2,), dtype=np.float64)
        C_VQF.filterInitialState(<vqf_real_t> x0, <double*> np.PyArray_DATA(b), <double*> np.PyArray_DATA(a),
                                 <double*> np.PyArray_DATA(out))
        return out

    @staticmethod
    def filterAdaptStateForCoeffChange(np.ndarray[vqf_real_t, ndim=1, mode='c'] last_y not None,
                                       np.ndarray[double, ndim=1, mode='c'] b_old not None,
                                       np.ndarray[double, ndim=1, mode='c'] a_old not None,
                                       np.ndarray[double, ndim=1, mode='c'] b_new not None,
                                       np.ndarray[double, ndim=1, mode='c'] a_new not None,
                                       np.ndarray[double, ndim=1, mode='c'] state not None):
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
        assert b_old.shape[0] == 3
        assert a_old.shape[0] == 2
        assert b_new.shape[0] == 3
        assert a_new.shape[0] == 2
        assert state.shape[0] == 2*N
        C_VQF.filterAdaptStateForCoeffChange(<vqf_real_t*> np.PyArray_DATA(last_y), <size_t> N,
                                             <double*> np.PyArray_DATA(b_old), <double*> np.PyArray_DATA(a_old),
                                             <double*> np.PyArray_DATA(b_new), <double*> np.PyArray_DATA(a_new),
                                             <double*> np.PyArray_DATA(state))

    @staticmethod
    def filterStep(x, np.ndarray[double, ndim=1, mode='c'] b not None,
                   np.ndarray[double, ndim=1, mode='c'] a not None,
                   np.ndarray[double, ndim=1, mode='c'] state not None):
        r"""Performs a filter step for a scalar value.

        :param x: input value
        :param b: numerator coefficients -- numpy array with shape (3,)
        :param a: denominator coefficients (without :math:`a_0=1`) -- numpy array with shape (2,)
        :param state: filter state -- numpy array with shape (2,), will be modified
        :return: filtered value
        """
        assert b.shape[0] == 3
        assert a.shape[0] == 2
        assert state.shape[0] == 2
        return C_VQF.filterStep(<vqf_real_t> x, <double*> np.PyArray_DATA(b), <double*> np.PyArray_DATA(a),
                                <double*> np.PyArray_DATA(state))

    @staticmethod
    def filterVec(np.ndarray[double, ndim=1, mode='c'] x not None, tau, Ts,
                  np.ndarray[double, ndim=1, mode='c'] b not None,
                  np.ndarray[double, ndim=1, mode='c'] a not None,
                  np.ndarray[double, ndim=1, mode='c'] state not None):
        r"""Performs filter step for vector-valued signal with averaging-based initialization.

        During the first :math:`\tau` seconds, the filter output is the mean of the previous samples. At :math:`t=\tau`,
        the initial conditions for the low-pass filter are calculated based on the current mean value and from then on,
        regular filtering with the rational transfer function described by the coefficients b and a is performed.

        :param x: input values -- numpy array with shape (N,)
        :param tau: filter time constant \:math:`\tau` in seconds (used for initialization)
        :param Ts: sampling time :math:`T_\mathrm{s}` in seconds (used for initialization)
        :param b: numerator coefficients -- numpy array with shape (3,)
        :param a: denominator coefficients (without :math:`a_0=1`) -- numpy array with shape (2,)
        :param state: filter state -- numpy array with shape (N*2,), will be modified
        :return: filtered values -- numpy array with shape (N,)
        """
        N = x.shape[0]
        assert N >= 2
        assert b.shape[0] == 3
        assert a.shape[0] == 2
        assert state.shape[0] == 2*N
        cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] out = np.empty(shape=(N,), dtype=vqf_real)
        C_VQF.filterVec(<vqf_real_t*> np.PyArray_DATA(x), <size_t> N, <vqf_real_t> tau, <vqf_real_t> Ts,
                        <double*> np.PyArray_DATA(b), <double*> np.PyArray_DATA(a),
                        <double*> np.PyArray_DATA(state), <vqf_real_t*> np.PyArray_DATA(out))
        return out

    @staticmethod
    def matrix3SetToScaledIdentity(scale, np.ndarray[vqf_real_t, ndim=2, mode='c'] out not None):
        """Sets a 3x3 matrix to a scaled version of the identity matrix.

        :param scale: value of diagonal elements
        :param out: output array -- numpy array with shape (3,3), will be modified
        :return: None
        """
        assert out.shape[0] == 3
        assert out.shape[1] == 3
        C_VQF.matrix3SetToScaledIdentity(<vqf_real_t> scale, <vqf_real_t*> np.PyArray_DATA(out))

    @staticmethod
    def matrix3Multiply(np.ndarray[vqf_real_t, ndim=2, mode='c'] in1 not None,
                        np.ndarray[vqf_real_t, ndim=2, mode='c'] in2 not None):
        r"""Performs 3x3 matrix multiplication (:math:`\mathbf{M}_\mathrm{out} = \mathbf{M}_1\mathbf{M}_2`).

        :param in1: input 3x3 matrix :math:`\mathbf{M}_1` -- numpy array with shape (3,3)
        :param in2: input 3x3 matrix :math:`\mathbf{M}_2` -- numpy array with shape (3,3)
        :return: output 3x3 matrix :math:`\mathbf{M}_\mathrm{out}` -- numpy array with shape (3,3)
        """
        assert in1.shape[0] == 3
        assert in1.shape[1] == 3
        assert in2.shape[0] == 3
        assert in2.shape[1] == 3
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out = np.zeros(shape=(3, 3), dtype=vqf_real)
        C_VQF.matrix3Multiply(<vqf_real_t*> np.PyArray_DATA(in1), <vqf_real_t*> np.PyArray_DATA(in2),
                              <vqf_real_t*> np.PyArray_DATA(out))
        return out

    @staticmethod
    def matrix3MultiplyTpsFirst(np.ndarray[vqf_real_t, ndim=2, mode='c'] in1 not None,
                                np.ndarray[vqf_real_t, ndim=2, mode='c'] in2 not None):
        r"""Performs 3x3 matrix multiplication after transposing the first matrix
        (:math:`\mathbf{M}_\mathrm{out} = \mathbf{M}_1^T\mathbf{M}_2`).

        :param in1: input 3x3 matrix :math:`\mathbf{M}_1` -- numpy array with shape (3,3)
        :param in2: input 3x3 matrix :math:`\mathbf{M}_2` -- numpy array with shape (3,3)
        :return: output 3x3 matrix :math:`\mathbf{M}_\mathrm{out}` -- numpy array with shape (3,3)
        """
        assert in1.shape[0] == 3
        assert in1.shape[1] == 3
        assert in2.shape[0] == 3
        assert in2.shape[1] == 3
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out = np.zeros(shape=(3, 3), dtype=vqf_real)
        C_VQF.matrix3MultiplyTpsFirst(<vqf_real_t*> np.PyArray_DATA(in1), <vqf_real_t*> np.PyArray_DATA(in2),
                                      <vqf_real_t*> np.PyArray_DATA(out))
        return out

    @staticmethod
    def matrix3MultiplyTpsSecond(np.ndarray[vqf_real_t, ndim=2, mode='c'] in1 not None,
                                 np.ndarray[vqf_real_t, ndim=2, mode='c'] in2 not None):
        r"""Performs 3x3 matrix multiplication after transposing the second matrix
        (:math:`\mathbf{M}_\mathrm{out} = \mathbf{M}_1\mathbf{M}_2^T`).

        :param in1: input 3x3 matrix :math:`\mathbf{M}_1` -- numpy array with shape (3,3)
        :param in2: input 3x3 matrix :math:`\mathbf{M}_2` -- numpy array with shape (3,3)
        :return: output 3x3 matrix :math:`\mathbf{M}_\mathrm{out}` -- numpy array with shape (3,3)
        """
        assert in1.shape[0] == 3
        assert in1.shape[1] == 3
        assert in2.shape[0] == 3
        assert in2.shape[1] == 3
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out = np.zeros(shape=(3, 3), dtype=vqf_real)
        C_VQF.matrix3MultiplyTpsSecond(<vqf_real_t*> np.PyArray_DATA(in1), <vqf_real_t*> np.PyArray_DATA(in2),
                                       <vqf_real_t*> np.PyArray_DATA(out))
        return out

    @staticmethod
    def matrix3Inv(np.ndarray[vqf_real_t, ndim=2, mode='c'] in_ not None):
        r"""Calculates the inverse of a 3x3 matrix (:math:`\mathbf{M}_\mathrm{out} = \mathbf{M}^{-1}`).

        :param in_: input 3x3 matrix :math:`\mathbf{M}` -- numpy array with shape (3,3)
        :return: output 3x3 matrix :math:`\mathbf{M}_\mathrm{out}` -- numpy array with shape (3,3)
        """
        assert in_.shape[0] == 3
        assert in_.shape[1] == 3
        cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out = np.zeros(shape=(3, 3), dtype=vqf_real)
        invertible = C_VQF.matrix3Inv(<vqf_real_t*> np.PyArray_DATA(in_), <vqf_real_t*> np.PyArray_DATA(out))
        return invertible, out


cdef extern from 'cpp/offline_vqf.hpp':
    void c_offlineVQF 'offlineVQF'(const vqf_real_t *gyr, const vqf_real_t *acc, const vqf_real_t *mag,
                                   size_t N, vqf_real_t Ts, VQFParams params,
                                   vqf_real_t out6D[], vqf_real_t out9D[], vqf_real_t outDelta[],
                                   vqf_real_t outBias[], vqf_real_t outBiasSigma[], bool outRest[], bool outMagDist[])


def offlineVQF(np.ndarray[vqf_real_t, ndim=2, mode='c'] gyr not None,
               np.ndarray[vqf_real_t, ndim=2, mode='c'] acc not None,
               np.ndarray[vqf_real_t, ndim=2, mode='c'] mag, Ts, params=None):
    """A Versatile Quaternion-based Filter for IMU Orientation Estimation.

    This function implements the offline variant of orientation estimation filter described in the following
    publication:

        D. Laidig and T. Seel. "VQF: Highly Accurate IMU Orientation Estimation with Bias Estimation and Magnetic
        Disturbance Rejection." Information Fusion 2023, 91, 187--204.
        `doi:10.1016/j.inffus.2022.10.014 <https://doi.org/10.1016/j.inffus.2022.10.014>`_.
        [Accepted manuscript available at `arXiv:2203.17024 <https://arxiv.org/abs/2203.17024>`_.]

    The filter can perform simultaneous 6D (magnetometer-free) and 9D (gyr+acc+mag) sensor fusion and can also be used
    without magnetometer data. It performs rest detection, gyroscope bias estimation during rest and motion, and
    magnetic disturbance detection and rejection.

    To use this offline implementation, all data must have the same sampling rate and be provided as a contiguous
    numpy array.

    This function is a Python wrapper (implemented in `Cython <https://cython.org/>`_) for the C++ implementation
    of the offline variant :cpp:func:`offlineVQF`. The offline variant makes use of the fact that the whole data is
    available and performs acausal signal processing, i.e. future samples are used to calculate the current
    estimate. Depending on use case and programming language of choice, the following alternatives might be useful:

    +------------------------+--------------------------+--------------------------+-----------------------------------+
    |                        | Full Version             | Basic Version            | Offline Version                   |
    |                        |                          |                          |                                   |
    +========================+==========================+==========================+===================================+
    | **C++**                | :cpp:class:`VQF`         | :cpp:class:`BasicVQF`    | :cpp:func:`offlineVQF`            |
    +------------------------+--------------------------+--------------------------+-----------------------------------+
    | **Python/C++ (fast)**  | :py:class:`vqf.VQF`      | :py:class:`vqf.BasicVQF` | **vqf.offlineVQF (this function)**|
    +------------------------+--------------------------+--------------------------+-----------------------------------+
    | **Pure Python (slow)** | :py:class:`vqf.PyVQF`    | --                       | --                                |
    +------------------------+--------------------------+--------------------------+-----------------------------------+
    | **Pure Matlab (slow)** | :mat:class:`VQF.m <VQF>` | --                       | --                                |
    +------------------------+--------------------------+--------------------------+-----------------------------------+

    The output is a dictionary containing

        - **quat6D** -- the 6D quaternion -- numpy array with shape (N, 4)
        - **bias** -- gyroscope bias estimate in rad/s -- numpy array with shape (N, 3)
        - **biasSigma** -- uncertainty of gyroscope bias estimate in rad/s -- numpy array with shape (N,)
        - **restDetected** -- rest detection state -- boolean numpy array with shape (N,)

    in all cases and if magnetometer data is provided additionally

        - **quat9D** -- the 9D quaternion -- numpy array with shape (N, 4)
        - **delta** -- heading difference angle between 6D and 9D quaternion in rad -- numpy array with shape (N,)
        - **magDistDetected** -- magnetic disturbance detection state -- boolean numpy array with shape (N,)

    :param gyr: gyroscope measurement in rad/s -- numpy array with shape (N, 3)
    :param acc: accelerometer measurement in m/s² -- numpy array with shape (N, 3)
    :param mag: magnetometer measurement in arbitrary units -- numpy array with shape (N, 3) or None
    :param Ts: sampling time in seconds
    :param params: parameters (pass ``{}`` or None to use the default parameters and
        see :cpp:struct:`VQFParams` for a full list and detailed descriptions)
    :return: dict with entries as described above
    """
    cdef int N = gyr.shape[0]
    assert acc.shape[0] == N
    assert gyr.shape[1] == 3
    assert acc.shape[1] == 3

    cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out6D = np.zeros(shape=(N, 4), dtype=vqf_real)
    cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] out9D
    cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] outDelta
    cdef np.ndarray[vqf_real_t, ndim=2, mode='c'] outBias = np.zeros(shape=(N, 3), dtype=vqf_real)
    cdef np.ndarray[vqf_real_t, ndim=1, mode='c'] outBiasSigma = np.zeros(shape=(N,), dtype=vqf_real)
    cdef np.ndarray[bool, ndim=1, mode='c'] outRest = np.zeros(shape=(N,), dtype=np.bool_)
    cdef np.ndarray[bool, ndim=1, mode='c'] outMagDist

    # use the VQF class constructor to convert the params dict into a VQFParams object
    if params is None:
        params = {}
    cdef VQFParams paramObj = VQF(0.01, 0.01, 0.01, **params).c_obj.getParams()

    if mag is not None:
        assert mag.shape[0] == N
        assert mag.shape[1] == 3
        out9D = np.zeros(shape=(N, 4), dtype=vqf_real)
        outDelta = np.zeros(shape=(N,), dtype=vqf_real)
        outMagDist = np.zeros(shape=(N,), dtype=np.bool_)
        c_offlineVQF(<vqf_real_t*> np.PyArray_DATA(gyr),
                     <vqf_real_t*> np.PyArray_DATA(acc),
                     <vqf_real_t*> np.PyArray_DATA(mag),
                     N, <vqf_real_t> Ts, paramObj,
                     <vqf_real_t*> np.PyArray_DATA(out6D),
                     <vqf_real_t*> np.PyArray_DATA(out9D),
                     <vqf_real_t*> np.PyArray_DATA(outDelta),
                     <vqf_real_t*> np.PyArray_DATA(outBias),
                     <vqf_real_t*> np.PyArray_DATA(outBiasSigma),
                     <bool*> np.PyArray_DATA(outRest),
                     <bool*> np.PyArray_DATA(outMagDist))
        return dict(quat6D=out6D, quat9D=out9D, delta=outDelta, bias=outBias, biasSigma=outBiasSigma,
                    restDetected=outRest, magDistDetected=outMagDist)
    else:
        c_offlineVQF(<vqf_real_t*> np.PyArray_DATA(gyr),
                     <vqf_real_t*> np.PyArray_DATA(acc),
                     NULL,
                     N, <vqf_real_t> Ts, paramObj,
                     <vqf_real_t*> np.PyArray_DATA(out6D),
                     NULL,
                     NULL,
                     <vqf_real_t*> np.PyArray_DATA(outBias),
                     <vqf_real_t*> np.PyArray_DATA(outBiasSigma),
                     <bool*> np.PyArray_DATA(outRest),
                     NULL)
        return dict(quat6D=out6D, bias=outBias, biasSigma=outBiasSigma, restDetected=outRest)
