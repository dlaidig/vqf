.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

.. module:: vqf
.. mat:module:: matlab
    :noindex:

vqf.PyVQF
#########

.. autoclass:: PyVQF
    :members:

    **Update Methods**

    .. autosummary::
        updateGyr
        updateAcc
        updateMag
        update
        updateBatch

    **Methods to Get/Set State**

    .. autosummary::
        getQuat3D
        getQuat6D
        getQuat9D
        getBiasEstimate
        setBiasEstimate
        getRestDetected
        getMagDistDetected
        getRelativeRestDeviations
        getMagRefNorm
        getMagRefDip
        setMagRef

    **Methods to Change Parameters**

    .. autosummary::
        setTauAcc
        setTauMag
        setMotionBiasEstEnabled
        setRestBiasEstEnabled
        setMagDistRejectionEnabled
        setRestDetectionThresholds

    **Access to Full Params/Coeffs/State**

    .. autosummary::
        params
        coeffs
        state
        resetState

    **Static Utility Functions**

    .. autosummary::
        quatMultiply
        quatConj
        quatApplyDelta
        quatRotate
        normalize
        gainFromTau
        filterCoeffs
        filterInitialState
        filterAdaptStateForCoeffChange
        filterStep
        filterVec
