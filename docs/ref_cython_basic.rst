.. SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
..
.. SPDX-License-Identifier: MIT

.. module:: vqf
   :noindex:
.. mat:module:: matlab
    :noindex:

vqf.BasicVQF
############

..
    Note that the :members: are specified manually in order to ensure that the members are ordered by source which is
    otherwise not possible with Cython modules.

.. autoclass:: BasicVQF
    :members: updateGyr,
        updateAcc,
        updateMag,
        update,
        updateBatch,
        updateBatchFullState,
        getQuat3D,
        getQuat6D,
        getQuat9D,
        getDelta,
        setTauAcc,
        setTauMag,
        params,
        coeffs,
        state,
        resetState,
        quatMultiply,
        quatConj,
        quatSetToIdentity,
        quatApplyDelta,
        quatRotate,
        norm,
        normalize,
        clip,
        gainFromTau,
        filterCoeffs,
        filterInitialState,
        filterAdaptStateForCoeffChange,
        filterStep,
        filterVec

    **Update Methods**

    .. autosummary::
        updateGyr
        updateAcc
        updateMag
        update
        updateBatch
        updateBatchFullState

    **Methods to Get State**

    .. autosummary::
        getQuat3D
        getQuat6D
        getQuat9D
        getDelta

    **Methods to Change Parameters**

    .. autosummary::
        setTauAcc
        setTauMag

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
        quatSetToIdentity
        quatApplyDelta
        quatRotate
        norm
        normalize
        clip
        gainFromTau
        filterCoeffs
        filterInitialState
        filterAdaptStateForCoeffChange
        filterStep
        filterVec
