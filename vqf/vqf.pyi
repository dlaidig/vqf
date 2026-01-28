# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import TypedDict, overload, Any
import numpy as np

class VQFParams(TypedDict):
    tauAcc: float
    tauMag: float
    motionBiasEstEnabled: bool
    restBiasEstEnabled: bool
    magDistRejectionEnabled: bool
    biasSigmaInit: float
    biasForgettingTime: float
    biasClip: float
    biasSigmaMotion: float
    biasVerticalForgettingFactor: float
    biasSigmaRest: float
    restMinT: float
    restFilterTau: float
    restThGyr: float
    restThAcc: float
    magCurrentTau: float
    magRefTau: float
    magNormTh: float
    magDipTh: float
    magNewTime: float
    magNewFirstTime: float
    magNewMinGyr: float
    magMinUndisturbedTime: float
    magMaxRejectionTime: float
    magRejectionFactor: float


class VQFCoefficients(TypedDict):
    gyrTs: float
    accTs: float
    magTs: float

    accLpB: list
    accLpA: list

    kMag: float

    biasP0: float
    biasV: float
    biasMotionW: float
    biasVerticalW: float
    biasRestW: float

    restGyrLpB: list
    restGyrLpA: list
    restAccLpB: list
    restAccLpA: list

    kMagRef: float
    magNormDipLpB: list
    magNormDipLpA: list


class VQFState(TypedDict):
    gyrQuat: list
    accQuat: list
    delta: float

    restDetected: bool
    magDistDetected: bool

    lastAccLp: list
    accLpState: list
    lastAccCorrAngularRate: float

    kMagInit: float
    lastMagDisAngle: float
    lastMagCorrAngularRate: float

    bias: list
    biasP: list

    motionBiasEstRLpState: list
    motionBiasEstBiasLpState: list

    restLastSquaredDeviations: list
    restT: float
    restLastGyrLp: list
    restGyrLpState: list
    restLastAccLp: list
    restAccLpState: list

    magRefNorm: float
    magRefDip: float
    magUndisturbedT: float
    magRejectT: float
    magCandidateNorm: float
    magCandidateDip: float
    magCandidateT: float
    magNormDip: list
    magNormDipLpState: list


class VQFBatchResults9D(TypedDict):
    quat6D: np.ndarray
    quat9D: np.ndarray
    delta: np.ndarray
    bias: np.ndarray
    biasSigma: np.ndarray
    restDetected: np.ndarray
    magDistDetected: np.ndarray


class VQFBatchResults6D(TypedDict):
    quat6D: np.ndarray
    bias: np.ndarray
    biasSigma: np.ndarray
    restDetected: np.ndarray


VQFBatchResults = VQFBatchResults9D | VQFBatchResults6D


class VQFBatchFullState(TypedDict):
    quat6D: np.ndarray
    quat9D: np.ndarray
    gyrQuat: np.ndarray
    accQuat: np.ndarray
    delta: np.ndarray
    restDetected: np.ndarray
    magDistDetected: np.ndarray
    lastAccLp: np.ndarray
    accLpState: np.ndarray
    lastAccCorrAngularRate: np.ndarray
    kMagInit: np.ndarray
    lastMagDisAngle: np.ndarray
    lastMagCorrAngularRate: np.ndarray
    bias: np.ndarray
    biasP: np.ndarray
    biasSigma: np.ndarray
    motionBiasEstRLpState: np.ndarray
    motionBiasEstBiasLpState: np.ndarray
    restLastSquaredDeviations: np.ndarray
    relativeRestDeviations: np.ndarray
    restT: np.ndarray
    restLastGyrLp: np.ndarray
    restGyrLpState: np.ndarray
    restLastAccLp: np.ndarray
    restAccLpState: np.ndarray
    magRefNorm: np.ndarray
    magRefDip: np.ndarray
    magUndisturbedT: np.ndarray
    magRejectT: np.ndarray
    magCandidateNorm: np.ndarray
    magCandidateDip: np.ndarray
    magCandidateT: np.ndarray
    magNormDip: np.ndarray
    magNormDipLpState: np.ndarray


class VQF:
    def __init__(
        self,
        gyrTs: float,
        accTs: float = -1.0,
        magTs: float = -1.0,
        tauAcc: float | None = None,
        tauMag: float | None = None,
        motionBiasEstEnabled: bool | None = None,
        restBiasEstEnabled: bool | None = None,
        magDistRejectionEnabled: bool | None = None,
        biasSigmaInit: float | None = None,
        biasForgettingTime: float | None = None,
        biasClip: float | None = None,
        biasSigmaMotion: float | None = None,
        biasVerticalForgettingFactor: float | None = None,
        biasSigmaRest: float | None = None,
        restMinT: float | None = None,
        restFilterTau: float | None = None,
        restThGyr: float | None = None,
        restThAcc: float | None = None,
        magCurrentTau: float | None = None,
        magRefTau: float | None = None,
        magNormTh: float | None = None,
        magDipTh: float | None = None,
        magNewTime: float | None = None,
        magNewFirstTime: float | None = None,
        magNewMinGyr: float | None = None,
        magMinUndisturbedTime: float | None = None,
        magMaxRejectionTime: float | None = None,
        magRejectionFactor: float | None = None,
    ) -> None: ...

    def updateGyr(self, gyr: np.ndarray) -> None: ...
    def updateAcc(self, acc: np.ndarray) -> None: ...
    def updateMag(self, mag: np.ndarray) -> None: ...
    def update(self, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray | None = None) -> None: ...

    @overload
    def updateBatch(self, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> VQFBatchResults9D: ...
    @overload
    def updateBatch(self, gyr: np.ndarray, acc: np.ndarray, mag: None = None) -> VQFBatchResults6D: ...
    def updateBatch(self, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray | None = None) -> VQFBatchResults: ...

    def updateBatchFullState(self, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray | None = None) -> VQFBatchFullState: ...

    def getQuat3D(self) -> np.ndarray: ...
    def getQuat6D(self) -> np.ndarray: ...
    def getQuat9D(self) -> np.ndarray: ...
    def getDelta(self) -> float: ...
    def getBiasEstimate(self) -> tuple[np.ndarray, float]: ...
    def setBiasEstimate(self, bias: np.ndarray, sigma: float = -1.0) -> None: ...
    def getRestDetected(self) -> bool: ...
    def getMagDistDetected(self) -> bool: ...
    def getRelativeRestDeviations(self) -> np.ndarray: ...
    def getMagRefNorm(self) -> float: ...
    def getMagRefDip(self) -> float: ...
    def setMagRef(self, norm: float, dip: float) -> None: ...
    def setTauAcc(self, tauAcc: float) -> None: ...
    def setTauMag(self, tauMag: float) -> None: ...
    def setMotionBiasEstEnabled(self, enabled: bool) -> None: ...
    def setRestBiasEstEnabled(self, enabled: bool) -> None: ...
    def setMagDistRejectionEnabled(self, enabled: bool) -> None: ...
    def setRestDetectionThresholds(self, thGyr: float, thAcc: float) -> None: ...

    @property
    def params(self) -> VQFParams: ...
    @property
    def coeffs(self) -> VQFCoefficients: ...
    @property
    def state(self) -> VQFState: ...
    @state.setter
    def state(self, state: VQFState) -> None: ...

    def resetState(self) -> None: ...

    @staticmethod
    def quatMultiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def quatConj(q: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def quatSetToIdentity(q: np.ndarray) -> None: ...
    @staticmethod
    def quatApplyDelta(q: np.ndarray, delta: float) -> np.ndarray: ...
    @staticmethod
    def quatRotate(q: np.ndarray, v: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def norm(vec: np.ndarray) -> float: ...
    @staticmethod
    def normalize(vec: np.ndarray) -> None: ...
    @staticmethod
    def clip(vec: np.ndarray, min: float, max: float) -> None: ...
    @staticmethod
    def gainFromTau(tau: float, Ts: float) -> float: ...
    @staticmethod
    def filterCoeffs(tau: float, Ts: float) -> tuple[np.ndarray, np.ndarray]: ...
    @staticmethod
    def filterInitialState(x0: float, b: np.ndarray, a: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def filterAdaptStateForCoeffChange(
        last_y: np.ndarray,
        b_old: np.ndarray,
        a_old: np.ndarray,
        b_new: np.ndarray,
        a_new: np.ndarray,
        state: np.ndarray,
    ) -> None: ...
    @staticmethod
    def filterStep(x: float, b: np.ndarray, a: np.ndarray, state: np.ndarray) -> float: ...
    @staticmethod
    def filterVec(x: np.ndarray, tau: float, Ts: float, b: np.ndarray, a: np.ndarray, state: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def matrix3SetToScaledIdentity(scale: float, out: np.ndarray) -> None: ...
    @staticmethod
    def matrix3Multiply(in1: np.ndarray, in2: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def matrix3MultiplyTpsFirst(in1: np.ndarray, in2: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def matrix3MultiplyTpsSecond(in1: np.ndarray, in2: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def matrix3Inv(mat: np.ndarray) -> tuple[bool, np.ndarray]: ...


@overload
def offlineVQF(
    gyr: np.ndarray,
    acc: np.ndarray,
    mag: np.ndarray,
    Ts: float,
    params: dict[str, Any] | None = None,
) -> VQFBatchResults9D: ...

@overload
def offlineVQF(
    gyr: np.ndarray,
    acc: np.ndarray,
    mag: None,
    Ts: float,
    params: dict[str, Any] | None = None,
) -> VQFBatchResults6D: ...

def offlineVQF(
    gyr: np.ndarray,
    acc: np.ndarray,
    mag: np.ndarray | None,
    Ts: float,
    params: dict[str, Any] | None = None,
) -> VQFBatchResults: ...
