// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

#include "offline_vqf.hpp"

#include <algorithm>
#include <limits>
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

#define EPS std::numeric_limits<vqf_real_t>::epsilon()
#define NaN std::numeric_limits<vqf_real_t>::quiet_NaN()


void matrix3MultiplyVec(const vqf_real_t inR[9], const vqf_real_t inV[9], vqf_real_t out[3])
{
    vqf_real_t tmp[3];
    tmp[0] = inR[0]*inV[0] + inR[1]*inV[1] + inR[2]*inV[2];
    tmp[1] = inR[3]*inV[0] + inR[4]*inV[1] + inR[5]*inV[2];
    tmp[2] = inR[6]*inV[0] + inR[7]*inV[1] + inR[8]*inV[2];
    std::copy(tmp, tmp+3, out);
}


void integrateGyr(const vqf_real_t *gyr, const vqf_real_t *bias, size_t N, vqf_real_t Ts, vqf_real_t *out)
{
    vqf_real_t q[4] = {1, 0, 0, 0};
    for (size_t i = 0; i < N; i++) {
        vqf_real_t gyrNoBias[3] = {gyr[3*i]-bias[3*i], gyr[3*i+1]-bias[3*i+1], gyr[3*i+2]-bias[3*i+2]};
        vqf_real_t gyrnorm = VQF::norm(gyrNoBias, 3);
        vqf_real_t angle = gyrnorm * Ts;
        if (gyrnorm > EPS) {
            vqf_real_t c = cos(angle/2);
            vqf_real_t s = sin(angle/2)/gyrnorm;
            vqf_real_t gyrStepQuat[4] = {c, s*gyrNoBias[0], s*gyrNoBias[1], s*gyrNoBias[2]};
            VQF::quatMultiply(q, gyrStepQuat, q);
            VQF::normalize(q, 4);
        }
        std::copy(q, q+4, out+4*i);
    }
}

void lowpassButterFiltfilt(vqf_real_t *accI, size_t N, vqf_real_t Ts, vqf_real_t tau)
{
    double b[3]; // check if everything compiles with float
    double a[3];
    double state[3*2];

    VQF::filterCoeffs(tau, Ts, b, a);

    // forward filter
    std::fill(state, state + 3*2, NaN); // let filterVec average the first samples
    for(size_t i = 0; i < N; i++) {
        VQF::filterVec(accI+3*i, 3, tau, Ts, b, a, state, accI+3*i);
    }

    // backward filter
    for(size_t j = 0; j < 3; j++) {
        VQF::filterInitialState(accI[3*N-3+j], b, a, state+2*j); // calculate initial state based on last sample
    }
    for(size_t i = N-1; i != size_t(-1); i--) {
        VQF::filterVec(accI+3*i, 3, tau, Ts, b, a, state, accI+3*i);
    }
}

void accCorrection(const vqf_real_t* quat3D, const vqf_real_t* accI, size_t N, vqf_real_t* quat6D)
{
    vqf_real_t accQuat[4] = {1, 0, 0, 0};

    for(size_t i = 0; i < N; i++) {
        // transform acc from inertial frame to 6D earth frame and normalize
        vqf_real_t accEarth[3];
        VQF::quatRotate(accQuat, accI + 3*i, accEarth);
        VQF::normalize(accEarth, 3);

        // inclination correction
        vqf_real_t accCorrQuat[4];
        vqf_real_t q_w = sqrt((accEarth[2]+1)/2);
        if (q_w > 1e-6) {
            accCorrQuat[0] = q_w;
            accCorrQuat[1] = 0.5*accEarth[1]/q_w;
            accCorrQuat[2] = -0.5*accEarth[0]/q_w;
            accCorrQuat[3] = 0;
        } else {
            // to avoid numeric issues when acc is close to [0 0 -1], i.e. the correction step is close (<= 0.00011°) to 180°:
            accCorrQuat[0] = 0;
            accCorrQuat[1] = 1;
            accCorrQuat[2] = 0;
            accCorrQuat[3] = 0;
        }
        VQF::quatMultiply(accCorrQuat, accQuat, accQuat);
        VQF::normalize(accQuat, 4);

        // calculate output quaternion
        VQF::quatMultiply(accQuat, quat3D + 4*i, quat6D + 4*i);
    }
}

void calculateDelta(const vqf_real_t* quat6D, const vqf_real_t* mag, size_t N, vqf_real_t* delta)
{
    vqf_real_t magEarth[3];

    for(size_t i = 0; i < N; i++) {
        // bring magnetometer measurement into 6D earth frame
        VQF::quatRotate(quat6D + 4*i, mag + 3*i, magEarth);

        // calculate disagreement angle based on current magnetometer measurement
        delta[i] = atan2(magEarth[0], magEarth[1]);
    }
}

void filterDelta(const bool* magDist, size_t N, vqf_real_t Ts, const VQFParams& params, bool backward,
                 vqf_real_t* delta)
{
    vqf_real_t d = backward ? delta[N-1] : delta[0];
    vqf_real_t kMag = VQF::gainFromTau(params.tauMag, Ts);
    vqf_real_t kMagInit = 1.0;
    vqf_real_t magRejectT = 0;

    for(size_t i = 0; i < N; i++) {
        size_t j = backward ? N-i-1 : i;
        vqf_real_t disAngle = delta[j] - d;

        // make sure the disagreement angle is in the range [-pi, pi]
        if (disAngle > vqf_real_t(M_PI)) {
            disAngle -= vqf_real_t(2*M_PI);
        } else if (disAngle < vqf_real_t(-M_PI)) {
            disAngle += vqf_real_t(2*M_PI);
        }

        vqf_real_t k = kMag;

        if (params.magDistRejectionEnabled) {
            // magnetic disturbance rejection
            if (magDist[j]) {
                if (magRejectT <= params.magMaxRejectionTime) {
                    magRejectT += Ts;
                    k = 0;
                } else {
                    k /= params.magRejectionFactor;
                }
            } else {
                magRejectT = std::max(magRejectT - params.magRejectionFactor*Ts, vqf_real_t(0.0));
            }
        }

        // ensure fast initial convergence
        if (kMagInit != vqf_real_t(0.0)) {
            // make sure that the gain k is at least 1/N, N=1,2,3,... in the first few samples
            if (k < kMagInit) {
                k = kMagInit;
            }

            // iterative expression to calculate 1/N
            kMagInit = kMagInit/(kMagInit+1);

            // disable if t > tauMag
            if (kMagInit*params.tauMag < Ts) {
                kMagInit = 0.0;
            }
        }

        // first-order filter step
        d += k*disAngle;

        // make sure delta is in the range [-pi, pi]
        if (d > vqf_real_t(M_PI)) {
            d -= vqf_real_t(2*M_PI);
        } else if (d < vqf_real_t(-M_PI)) {
            d += vqf_real_t(2*M_PI);
        }

        // write output back into delta array
        delta[j] = d;
    }
}


void offlineVQF(const vqf_real_t gyr[], const vqf_real_t acc[], const vqf_real_t mag[],
                size_t N, vqf_real_t Ts, const VQFParams &params, vqf_real_t out6D[], vqf_real_t out9D[],
                vqf_real_t outDelta[], vqf_real_t outBias[], vqf_real_t outBiasSigma[],
                bool outRest[], bool outMagDist[])
{
    // create temporary buffers if null pointers are passed, otherwise use output buffers directly
    vqf_real_t *quat6D = out6D ? out6D : new vqf_real_t[N*4]();
    vqf_real_t *quat9D = out9D ? out9D : new vqf_real_t[N*4]();
    vqf_real_t *delta = outDelta ? outDelta : new vqf_real_t[N]();
    vqf_real_t *bias = outBias ? outBias : new vqf_real_t[N*3]();
    vqf_real_t *biasSigma = outBiasSigma ? outBiasSigma : new vqf_real_t[N]();
    bool *rest = outRest ? outRest : new bool[N]();
    bool *magDist = outMagDist ? outMagDist : new bool[N]();

    // run real-time VQF implementation in forward direction
    VQF vqf(params, Ts);
    vqf_real_t *biasPInv1 = new vqf_real_t[N*9]();
    for (size_t i = 0; i < N; i++) {
        if (mag) {
            vqf.update(gyr+3*i, acc+3*i, mag+3*i);
        } else {
            vqf.update(gyr+3*i, acc+3*i);
        }
        rest[i] = vqf.getRestDetected();
        magDist[i] = vqf.getMagDistDetected();
        vqf.getBiasEstimate(bias+3*i);
        VQF::matrix3Inv(vqf.getState().biasP, biasPInv1+9*i);
    }

    // run real-time VQF implementation in backward direction
    vqf.resetState();
    for (size_t i = N-1; i != size_t(-1); i--) {
        vqf_real_t tempGyr[3] = {-gyr[3*i], -gyr[3*i+1], -gyr[3*i+2]};
        if (mag) {
            vqf.update(tempGyr, acc+3*i, mag+3*i);
        } else {
            vqf.update(tempGyr, acc+3*i);
        }
        rest[i] = rest[i] || vqf.getRestDetected();
        magDist[i] = magDist[i] && vqf.getMagDistDetected();

        vqf_real_t bias2[3];
        vqf_real_t biasPInv2[9];
        vqf.getBiasEstimate(bias2);
        VQF::matrix3Inv(vqf.getState().biasP, biasPInv2);

        // determine bias estimate by averaging both estimates via the covariances
        // P_1^-1 * b_1
        matrix3MultiplyVec(biasPInv1+9*i, bias+3*i, bias+3*i);
        // P_2^-1 * b_2
        matrix3MultiplyVec(biasPInv2, bias2, bias2);
        // P_1^-1 * b_1 - P_2^-1 * b_2
        bias[3*i] -= bias2[0];
        bias[3*i+1] -= bias2[1];
        bias[3*i+2] -= bias2[2];
        // (P_1^-1 + P_2^-1)^-1
        for (size_t j = 0; j < 9; j++) {
            biasPInv1[9*i+j] += biasPInv2[j];
        }
        VQF::matrix3Inv(biasPInv1+9*i, biasPInv1+9*i);
        // (P_1^-1 + P_2^-1)^-1 * (P_1^-1 * b_1 - P_2^-1 * b_2)
        matrix3MultiplyVec(biasPInv1+9*i, bias+3*i, bias+3*i);

        // determine bias estimation uncertainty based on new covariance (P_1^-1 + P_2^-1)^-1
        // (cf. VQF::getBiasEstimate)
        vqf_real_t sum1 = fabs(biasPInv1[9*i+0]) + fabs(biasPInv1[9*i+1]) + fabs(biasPInv1[9*i+2]);
        vqf_real_t sum2 = fabs(biasPInv1[9*i+3]) + fabs(biasPInv1[9*i+4]) + fabs(biasPInv1[9*i+5]);
        vqf_real_t sum3 = fabs(biasPInv1[9*i+6]) + fabs(biasPInv1[9*i+7]) + fabs(biasPInv1[9*i+8]);
        vqf_real_t P = std::max(std::max(sum1, sum2), sum3);
        biasSigma[i] = std::min(sqrt(P)*vqf_real_t(M_PI/100.0/180.0), params.biasSigmaInit);
    }

    // perform gyroscope integration
    vqf_real_t *quat3D = new vqf_real_t[N*4]();
    integrateGyr(gyr, bias, N, Ts, quat3D);

    // transform acc to inertial frame
    vqf_real_t *accI = new vqf_real_t[N*3]();
    for (size_t i = 0; i < N; i++) {
        VQF::quatRotate(quat3D+4*i, acc+3*i, accI+3*i);
    }

    // filter acc in inertial frame
    lowpassButterFiltfilt(accI, N, Ts, params.tauAcc);

    // inclination correction
    accCorrection(quat3D, accI, N, quat6D);

    // heading correction
    if (mag) {
        calculateDelta(quat6D, mag, N, delta);
        filterDelta(magDist, N, Ts, params, false, delta); // forward direction
        filterDelta(magDist, N, Ts, params, true, delta); // backward direction
        for (size_t i = 0; i < N; i++) {
           VQF::quatApplyDelta(quat6D + 4*i, delta[i], quat9D + 4*i);
        }
    } else {
        std::fill(delta, delta + N, 0);
        std::fill(quat9D, quat9D + 4*N, 0);
    }

    // delete temporary buffers
    if (!out6D) {
        delete[] quat6D;
    }
    if (!out9D) {
        delete[] quat9D;
    }
    if (!outDelta) {
        delete[] delta;
    }
    if (!outBias) {
        delete[] bias;
    }
    if (!outBiasSigma) {
        delete[] biasSigma;
    }
    if (!outRest) {
        delete[] rest;
    }
    if (!outMagDist) {
        delete[] magDist;
    }
    delete[] biasPInv1;
    delete[] quat3D;
    delete[] accI;
}
