// SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
//
// SPDX-License-Identifier: MIT

// This is a simple main function that runs VQF with simple dummy data and a few different options.
// The main purpose of this file is to test compilation and to have an executable to run memtest:
//     valgrind --tool=memcheck --track-origins=yes --leak-check=full ./vqf_dummy

#include "vqf.hpp"
#include "basicvqf.hpp"
#ifndef VQF_NO_MOTION_BIAS_ESTIMATION
#include "offline_vqf.hpp"
#endif

#include <iostream>

VQFParams getParams(int mode)
{
    VQFParams params;

    switch (mode) {
    case 0:
        break; // default params
    case 1:
        params.magDistRejectionEnabled = false;
        break;
    case 2:
        params.restBiasEstEnabled = false;
        break;
    case 3:
#ifndef VQF_NO_MOTION_BIAS_ESTIMATION
        params.motionBiasEstEnabled = false;
#endif
        break;
    case 4:
        params.restBiasEstEnabled = false;
#ifndef VQF_NO_MOTION_BIAS_ESTIMATION
        params.motionBiasEstEnabled = false;
#endif
        break;
    case 5:
        params.magDistRejectionEnabled = false;
        params.restBiasEstEnabled = false;
#ifndef VQF_NO_MOTION_BIAS_ESTIMATION
        params.motionBiasEstEnabled = false;
#endif
        break;
    default:
        std::cout << "invalid mode!" << std::endl;
    }

    return params;
}

void run(int mode)
{
    vqf_real_t gyr[3] = {0.01, 0.01, 0.01};
    vqf_real_t acc[3] = {0, 9.8, 0};
    vqf_real_t mag[3] = {0.5, 0.8, 0};
    vqf_real_t quat[4] = {0, 0, 0, 0};

    VQFParams params = getParams(mode);

    VQF vqf(params, 0.01);
    for (size_t i=0; i < 10000; i++) {
        vqf.update(gyr, acc, mag);
        vqf.getQuat9D(quat);
    }

    std::cout << "VQF, mode: " << mode << ", quat: [" <<
                 quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << "]" << std::endl;
}

void runBasic()
{
    vqf_real_t gyr[3] = {0.01, 0.01, 0.01};
    vqf_real_t acc[3] = {0, 9.8, 0};
    vqf_real_t mag[3] = {0.5, 0.8, 0};
    vqf_real_t quat[4] = {0, 0, 0, 0};

    BasicVQF vqf(0.01);
    for (size_t i=0; i < 10000; i++) {
        vqf.update(gyr, acc, mag);
        vqf.getQuat9D(quat);
    }

    std::cout << "BasicVQF, quat: [" <<
                 quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << "]" << std::endl;
}

#ifndef VQF_NO_MOTION_BIAS_ESTIMATION // setting this disables functions needed by offline estimation
void runOffline(int mode)
{
    size_t N = 10000;
    vqf_real_t *gyr = new vqf_real_t[N*3]();
    vqf_real_t *acc = new vqf_real_t[N*3]();
    vqf_real_t *mag = new vqf_real_t[N*3]();
    vqf_real_t *quat = new vqf_real_t[N*4]();
    for (size_t i = 0; i < N; i++) {
        gyr[3*i] = 0.01;
        gyr[3*i + 1] = 0.01;
        gyr[3*i + 2] = 0.01;
        acc[3*i] = 0;
        acc[3*i + 1] = 9.8;
        acc[3*i + 2] = 0;
        mag[3*i] = 0.5;
        mag[3*i + 1] = 0.8;
        mag[3*i + 2] = 0;
    }

    VQFParams params = getParams(mode);

    offlineVQF(gyr, acc, mag, N, 0.01, params, 0, quat, 0, 0, 0, 0, 0);

    std::cout << "offlineVQF, mode: " << mode << ", quat[0]: [" <<
                 quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << "]" << ", quat[N]: [" <<
                 quat[4*(N-1)] << ", " << quat[4*(N-1)+1] << ", " << quat[4*(N-1)+2] << ", " << quat[4*(N-1)+3] <<
                 "]" << std::endl;
    delete[] acc;
    delete[] gyr;
    delete[] mag;
    delete[] quat;
}
#endif

int main()
{
    for (int mode=0; mode <= 5; mode++) {
        run(mode);
    }
    runBasic();

#ifndef VQF_NO_MOTION_BIAS_ESTIMATION
    for (int mode=0; mode <= 5; mode++) {
        runOffline(mode);
    }
#endif

    return 0;
}
