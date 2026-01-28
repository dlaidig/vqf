# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import TypedDict, overload
import numpy as np

class BasicVQFParams(TypedDict):
    tauAcc: float
    tauMag: float


class BasicVQFCoefficients(TypedDict):
    gyrTs: float
    accTs: float
    magTs: float
    accLpB: list
    accLpA: list
    kMag: float


class BasicVQFState(TypedDict):
    gyrQuat: list
    accQuat: list
    delta: float
    lastAccLp: list
    accLpState: list
    kMagInit: float


class BasicVQFBatchResults9D(TypedDict):
    quat6D: np.ndarray
    quat9D: np.ndarray
    delta: np.ndarray


class BasicVQFBatchResults6D(TypedDict):
    quat6D: np.ndarray


BasicVQFBatchResults = BasicVQFBatchResults9D | BasicVQFBatchResults6D


class BasicVQFBatchFullState(TypedDict):
    quat6D: np.ndarray
    quat9D: np.ndarray
    gyrQuat: np.ndarray
    accQuat: np.ndarray
    delta: np.ndarray
    lastAccLp: np.ndarray
    accLpState: np.ndarray
    kMagInit: np.ndarray


class BasicVQF:
    def __init__(
        self,
        gyrTs: float,
        accTs: float = -1.0,
        magTs: float = -1.0,
        tauAcc: float | None = None,
        tauMag: float | None = None,
    ) -> None: ...

    def updateGyr(self, gyr: np.ndarray) -> None: ...
    def updateAcc(self, acc: np.ndarray) -> None: ...
    def updateMag(self, mag: np.ndarray) -> None: ...
    def update(self, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray | None = None) -> None: ...

    @overload
    def updateBatch(self, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> BasicVQFBatchResults9D: ...
    @overload
    def updateBatch(self, gyr: np.ndarray, acc: np.ndarray, mag: None = None) -> BasicVQFBatchResults6D: ...
    def updateBatch(
        self, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray | None = None
    ) -> BasicVQFBatchResults: ...

    def updateBatchFullState(
        self, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray | None = None
    ) -> BasicVQFBatchFullState: ...

    def getQuat3D(self) -> np.ndarray: ...
    def getQuat6D(self) -> np.ndarray: ...
    def getQuat9D(self) -> np.ndarray: ...
    def getDelta(self) -> float: ...
    def setTauAcc(self, tauAcc: float) -> None: ...
    def setTauMag(self, tauMag: float) -> None: ...

    @property
    def params(self) -> BasicVQFParams: ...
    @property
    def coeffs(self) -> BasicVQFCoefficients: ...
    @property
    def state(self) -> BasicVQFState: ...
    @state.setter
    def state(self, state: BasicVQFState) -> None: ...

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
