% SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
%
% SPDX-License-Identifier: MIT

classdef VQF < handle
    % A Versatile Quaternion-based Filter for IMU Orientation Estimation.
    %
    % This class implements the orientation estimation filter described in the following publication:
    %
    %     D. Laidig and T. Seel. "VQF: Highly Accurate IMU Orientation Estimation with Bias Estimation and Magnetic
    %     Disturbance Rejection." Information Fusion 2023, 91, 187--204.
    %     `doi:10.1016/j.inffus.2022.10.014 <https://doi.org/10.1016/j.inffus.2022.10.014>`_.
    %     [Accepted manuscript available at `arXiv:2203.17024 <https://arxiv.org/abs/2203.17024>`_.]
    %
    % The filter can perform simultaneous 6D (magnetometer-free) and 9D (gyr+acc+mag) sensor fusion and can also be used
    % without magnetometer data. It performs rest detection, gyroscope bias estimation during rest and motion, and magnetic
    % disturbance detection and rejection. Different sampling rates for gyroscopes, accelerometers, and magnetometers are
    % supported as well. While in most cases, the defaults will be reasonable, the algorithm can be influenced via a
    % number of tuning parameters.
    %
    % To use this class for online (sample-by-sample) processing,
    %
    % 1. create a instance of the class and provide the sampling time and, optionally, parameters
    % 2. for every sample, call one of the update functions to feed the algorithm with IMU data
    % 3. access the estimation results with :meth:`getQuat6D`, :meth:`getQuat9D` and the other getter methods.
    %
    % If the full data is available in matrices, you can use :meth:`updateBatch`.
    %
    % This class is a pure Matlab implementation of the algorithm. Note that the C++ implementation :cpp:class:`VQF`
    % and the Python wrapper :class:`vqf.VQF` are much faster than this pure Matlab implementation.
    % Depending on use case and programming language of choice, the following alternatives might be useful:
    %
    % +------------------------+------------------------+--------------------------+---------------------------+
    % |                        | Full Version           | Basic Version            | Offline Version           |
    % |                        |                        |                          |                           |
    % +========================+========================+==========================+===========================+
    % | **C++**                | :cpp:class:`VQF`       | :cpp:class:`BasicVQF`    | :cpp:func:`offlineVQF`    |
    % +------------------------+------------------------+--------------------------+---------------------------+
    % | **Python/C++ (fast)**  | :py:class:`vqf.VQF`    | :py:class:`vqf.BasicVQF` | :py:meth:`vqf.offlineVQF` |
    % +------------------------+------------------------+--------------------------+---------------------------+
    % | **Pure Python (slow)** | :py:class:`vqf.PyVQF`  | --                       | --                        |
    % +------------------------+------------------------+--------------------------+---------------------------+
    % | **Pure Matlab (slow)** | **VQF.m (this class)** | --                       | --                        |
    % +------------------------+------------------------+--------------------------+---------------------------+
    %
    % The constructor accepts the following arguments:
    %
    % .. code-block:: matlab
    %
    %      vqf = VQF(gyrTs);
    %      vqf = VQF(gyrTs, accTs);
    %      vqf = VQF(gyrTs, accTs, magTs);
    %      vqf = VQF(gyrTs, params);
    %      vqf = VQF(gyrTs, accTs, params);
    %      vqf = VQF(gyrTs, accTs, magTs, params);
    %
    %
    % In the most common case (using the default parameters and all data being sampled with the same frequency, create the
    % class like this:
    %
    % .. code-block:: matlab
    %
    %      vqf = VQF(0.01); % 0.01 s sampling time, i.e. 100 Hz
    %
    % Example code to create an object with magnetic disturbance rejection disabled:
    %
    % .. code-block:: matlab
    %
    %     vqf = VQF(0.01, struct('magDistRejectionEnabled', false)); % 0.01 s sampling time, i.e. 100 Hz
    %
    % To use this class as a replacement for the basic version BasicVQF, pass the following parameters:
    %
    % .. code-block:: matlab
    %
    %     vqf = VQF(0.01, struct('motionBiasEstEnabled', false, 'restBiasEstEnabled', false, 'magDistRejectionEnabled', false));
    %
    % See :cpp:struct:`VQFParams` for a detailed description of all parameters.
    %
    % This class can be used in Simulink via a Matlab function block. See the following minimum example to get an idea
    % of how to get started:
    %
    % .. code-block:: matlab
    %
    %     function quat = vqf_block(gyr, acc)
    %         persistent vqf
    %         if isempty(vqf)
    %             vqf = VQF(0.01, struct('magDistRejectionEnabled', false));
    %         end
    %         vqf.update(gyr', acc');
    %         quat = vqf.getQuat6D();
    %     end
    %
    % :param gyrTs: sampling time of the gyroscope measurements in seconds
    % :param accTs: sampling time of the accelerometer measurements in seconds
    %     (the value of `gyrTs` is used if set to -1)
    % :param magTs: sampling time of the magnetometer measurements in seconds
    %     (the value of `gyrTs` is used if set to -1)
    % :param params: struct containing optional parameters to override the defaults
    %     (see :cpp:struct:`VQFParams` for a full list and detailed descriptions)

    properties
        params = struct(...
            'tauAcc', 3.0, ...
            'tauMag', 9.0, ...
            'motionBiasEstEnabled', true, ...
            'restBiasEstEnabled', true, ...
            'magDistRejectionEnabled', true, ...
            'biasSigmaInit', 0.5, ...
            'biasForgettingTime', 100.0, ...
            'biasClip', 2.0, ...
            'biasSigmaMotion', 0.1, ...
            'biasVerticalForgettingFactor', 0.0001, ...
            'biasSigmaRest', 0.03, ...
            'restMinT', 1.5, ...
            'restFilterTau', 0.5, ...
            'restThGyr', 2.0, ...
            'restThAcc', 0.5, ...
            'magCurrentTau', 0.05, ...
            'magRefTau', 20.0, ...
            'magNormTh', 0.1, ...
            'magDipTh', 10.0, ...
            'magNewTime', 20.0, ...
            'magNewFirstTime', 5.0, ...
            'magNewMinGyr', 20.0, ...
            'magMinUndisturbedTime', 0.5, ...
            'magMaxRejectionTime', 60.0, ...
            'magRejectionFactor', 2.0 ...
        ) % Struct containing the current parameters (see :cpp:struct:`VQFParams`).
        state = struct(...
            'gyrQuat', [1 0 0 0], ...
            'accQuat', [1 0 0 0], ...
            'delta', 0.0, ...
            'restDetected', false, ...
            'magDistDetected', true, ...
            'lastAccLp', [0 0 0], ...
            'accLpState', NaN(2, 3), ...
            'lastAccCorrAngularRate', 0, ...
            'kMagInit', 1.0, ...
            'lastMagDisAngle', 0.0, ...
            'lastMagCorrAngularRate', 0.0, ...
            'bias', [0 0 0], ...
            'biasP', NaN(3, 3), ...))
            'motionBiasEstRLpState', NaN(2, 9), ...
            'motionBiasEstBiasLpState', NaN(2, 2), ...
            'restLastSquaredDeviations', [0 0 0], ...
            'restT', 0.0, ...
            'restLastGyrLp', [0 0 0], ...
            'restGyrLpState', NaN(2, 3), ...
            'restLastAccLp', [0 0 0], ...
            'restAccLpState', NaN(2, 3), ...
            'magRefNorm', 0.0, ...
            'magRefDip', 0.0, ...
            'magUndisturbedT', 0.0, ...
            'magRejectT', -1.0, ...
            'magCandidateNorm', -1.0, ...
            'magCandidateDip', 0.0, ...
            'magCandidateT', 0.0, ...
            'magNormDip', [0 0], ...
            'magNormDipLpState', NaN(2, 2) ...
        ) % Struct containing the current state (see :cpp:struct:`VQFState`).
        coeffs = struct(...
            'gyrTs', -1.0, ...
            'accTs', -1.0, ...
            'magTs', -1.0, ...
            'accLpB', NaN(1, 3), ...
            'accLpA', NaN(1, 2), ...
            'kMag', -1.0, ...
            'biasP0', -1.0, ...
            'biasV', -1.0, ...
            'biasMotionW', -1.0, ...
            'biasVerticalW', -1.0, ...
            'biasRestW', -1.0, ...
            'restGyrLpB', NaN(1, 3), ...
            'restGyrLpA', NaN(1, 2), ...
            'restAccLpB', NaN(1, 3), ...
            'restAccLpA', NaN(1, 2), ...
            'kMagRef', -1.0, ...
            'magNormDipLpB', NaN(1, 3), ...
            'magNormDipLpA', NaN(1, 2) ...
        ) % Struct containing the coefficients used by the algorithm (see :cpp:struct:`VQFCoefficients`).
    end
    methods
        function obj = VQF(gyrTs, varargin)
            obj.coeffs.gyrTs = gyrTs;

            if nargin > 1 && isstruct(varargin{nargin-1})
                params = varargin{nargin-1};
                fields = fieldnames(params);
                for i=1:length(fields)
                    obj.params.(fields{i}) = params.(fields{i});
                end
                args = nargin - 2;
            else
                args = nargin - 1;
            end

            if args == 1
                obj.coeffs.accTs = varargin{1};
            elseif args == 2
                obj.coeffs.accTs = varargin{1};
                obj.coeffs.magTs = varargin{2};
            elseif args ~= 0
                error('unexpected number of arguments');
            end

            if obj.coeffs.accTs < 0
                obj.coeffs.accTs = obj.coeffs.gyrTs;
            end
            if obj.coeffs.magTs < 0
                obj.coeffs.magTs = obj.coeffs.gyrTs;
            end

            assert(isnumeric(obj.coeffs.gyrTs) && isscalar(obj.coeffs.gyrTs) && obj.coeffs.gyrTs > 0);
            assert(isnumeric(obj.coeffs.accTs) && isscalar(obj.coeffs.accTs) && obj.coeffs.accTs > 0);
            assert(isnumeric(obj.coeffs.magTs) && isscalar(obj.coeffs.magTs) && obj.coeffs.magTs > 0);

            obj.setup();
        end
        function updateGyr(obj, gyr)
            % Performs gyroscope update step.
            %
            % It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
            % different sampling rates. Otherwise, simply use :meth:`update`.
            %
            % :param gyr: gyroscope measurement in rad/s -- 1x3 matrix
            assert(all(size(gyr) == [1, 3]), 'gyr has wrong size');

            % rest detection
            if obj.params.restBiasEstEnabled || obj.params.magDistRejectionEnabled
                [gyrLp, obj.state.restGyrLpState] = obj.filterVec(gyr, obj.params.restFilterTau, obj.coeffs.gyrTs, ...
                    obj.coeffs.restGyrLpB, obj.coeffs.restGyrLpA, obj.state.restGyrLpState);

                deviation = gyr - gyrLp;
                squaredDeviation = dot(deviation, deviation);

                biasClip = obj.params.biasClip*pi/180;
                if squaredDeviation >= (obj.params.restThGyr*pi/180.0)^2 || max(abs(gyrLp)) > biasClip
                    obj.state.restT = 0.0;
                    obj.state.restDetected = false;
                end

                obj.state.restLastGyrLp = gyrLp;
                obj.state.restLastSquaredDeviations(1) = squaredDeviation;
            end

            % remove estimated gyro bias
            gyrNoBias = gyr - obj.state.bias;

            % gyroscope prediction step
            gyrNorm = norm(gyrNoBias);
            angle = gyrNorm * obj.coeffs.gyrTs;
            if gyrNorm > eps
                c = cos(angle/2);
                s = sin(angle/2);
                gyrStepQuat = [c, s*gyrNoBias./gyrNorm];
                obj.state.gyrQuat = obj.quatMultiply(obj.state.gyrQuat, gyrStepQuat);
                obj.state.gyrQuat = obj.normalize(obj.state.gyrQuat);
            end
        end
        function updateAcc(obj, acc)
            % Performs accelerometer update step.
            %
            % It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
            % different sampling rates. Otherwise, simply use :meth:`update`.
            %
            % Should be called after :meth:`updateGyr` and before :meth:`updateMag`.
            %
            % :param acc: accelerometer measurement in m/s² -- 1x3 matrix
            assert(all(size(acc) == [1, 3]), 'acc has wrong size');

            % ignore [0 0 0] samples
            if all(acc == 0)
                return
            end

            accTs = obj.coeffs.accTs;

            % rest detection
            if obj.params.restBiasEstEnabled
                [accLp, obj.state.restAccLpState] = obj.filterVec(acc, obj.params.restFilterTau, accTs, ...
                    obj.coeffs.restAccLpB, obj.coeffs.restAccLpA, obj.state.restAccLpState);

                 deviation = acc - accLp;
                 squaredDeviation = dot(deviation, deviation);

                if squaredDeviation >= obj.params.restThAcc^2
                    obj.state.restT = 0.0;
                    obj.state.restDetected = false;
                else
                    obj.state.restT = obj.state.restT + accTs;
                    if obj.state.restT >= obj.params.restMinT
                        obj.state.restDetected = true;
                    end
                end
                obj.state.restLastAccLp = accLp;
                obj.state.restLastSquaredDeviations(2) = squaredDeviation;
            end

            % filter acc in inertial frame
            accEarth = obj.quatRotate(obj.state.gyrQuat, acc);
            [obj.state.lastAccLp, obj.state.accLpState] = obj.filterVec(accEarth, obj.params.tauAcc, accTs, obj.coeffs.accLpB, obj.coeffs.accLpA, obj.state.accLpState);

            % transform to 6D earth frame and normalize
            accEarth = obj.quatRotate(obj.state.accQuat, obj.state.lastAccLp);
            accEarth = obj.normalize(accEarth);

            % inclination correction
            q_w = sqrt((accEarth(3)+1)/2);
            if q_w > 1e-6
                accCorrQuat = [q_w, 0.5*accEarth(2)/q_w, -0.5*accEarth(1)/q_w, 0];
            else
                % to avoid numeric issues when acc is close to [0 0 -1], i.e. the correction step is close (<= 0.00011°) to 180°:
                accCorrQuat = [0 1 0 0];
            end
            obj.state.accQuat = obj.quatMultiply(accCorrQuat, obj.state.accQuat);
            obj.state.accQuat = obj.normalize(obj.state.accQuat);

            % calculate correction angular rate to facilitate debugging
            obj.state.lastAccCorrAngularRate = acos(accEarth(3))/obj.coeffs.accTs;

            % bias estimation
            if obj.params.motionBiasEstEnabled || obj.params.restBiasEstEnabled
                biasClip = obj.params.biasClip*pi/180;
                bias = obj.state.bias;

                % get rotation matrix corresponding to accGyrQuat
                accGyrQuat = obj.getQuat6D();
                R = zeros(1, 9);
                R(1) = 1 - 2*accGyrQuat(3)^2 - 2*accGyrQuat(4)^2; % r11
                R(2) = 2*(accGyrQuat(3)*accGyrQuat(2) - accGyrQuat(1)*accGyrQuat(4)); % r12
                R(3) = 2*(accGyrQuat(1)*accGyrQuat(3) + accGyrQuat(4)*accGyrQuat(2)); % r13
                R(4) = 2*(accGyrQuat(1)*accGyrQuat(4) + accGyrQuat(3)*accGyrQuat(2)); % r21
                R(5) = 1 - 2*accGyrQuat(2)^2 - 2*accGyrQuat(4)^2; % r22
                R(6) = 2*(accGyrQuat(3)*accGyrQuat(4) - accGyrQuat(2)*accGyrQuat(1)); % r23
                R(7) = 2*(accGyrQuat(4)*accGyrQuat(2) - accGyrQuat(1)*accGyrQuat(3)); % r31
                R(8) = 2*(accGyrQuat(1)*accGyrQuat(2) + accGyrQuat(4)*accGyrQuat(3)); % r32
                R(9) = 1 - 2*accGyrQuat(2)^2 - 2*accGyrQuat(3)^2; % r33

                % calculate R*b_hat (only the x and y component, as z is not needed)
                biasLp = zeros(1, 2);
                biasLp(1) = R(1)*obj.state.bias(1) + R(2)*bias(2) + R(3)*bias(3);
                biasLp(2) = R(4)*bias(1) + R(5)*bias(2) + R(6)*bias(3);

                % low-pass filter R and R*b_hat
                [R, obj.state.motionBiasEstRLpState] = obj.filterVec(R, obj.params.tauAcc, accTs, ...
                    obj.coeffs.accLpB, obj.coeffs.accLpA, obj.state.motionBiasEstRLpState);
                [biasLp, obj.state.motionBiasEstBiasLpState] = obj.filterVec(biasLp, obj.params.tauAcc, ...
                    accTs, obj.coeffs.accLpB, obj.coeffs.accLpA, obj.state.motionBiasEstBiasLpState);

                % set measurement error and covariance for the respective Kalman filter update
                if obj.state.restDetected && obj.params.restBiasEstEnabled
                    e = obj.state.restLastGyrLp - obj.state.bias;
                    R = eye(3);
                    w = obj.coeffs.biasRestW*[1 1 1];
                elseif obj.params.motionBiasEstEnabled
                    e = zeros(1, 3);
                    e(1) = -accEarth(2)/accTs + biasLp(1) - R(1)*bias(1) - R(2)*bias(2) - R(3)*bias(3);
                    e(2) = accEarth(1)/accTs + biasLp(2) - R(4)*bias(1) - R(5)*bias(2) - R(6)*bias(3);
                    e(3) = - R(7)*bias(1) - R(8)*bias(2) - R(9)*bias(3);
                    R = reshape(R, 3, 3)';
                    w = [obj.coeffs.biasMotionW, obj.coeffs.biasMotionW, obj.coeffs.biasVerticalW];
                else
                    e = zeros(1, 3); % needed for codegen
                    w = -1; % disable update
                end

                % Kalman filter update
                % step 1: P = P + V (also increase covariance if there is no measurement update!)
                if obj.state.biasP(1,1) < obj.coeffs.biasP0
                    obj.state.biasP(1,1) = obj.state.biasP(1,1) + obj.coeffs.biasV;
                end
                if obj.state.biasP(2,2) < obj.coeffs.biasP0
                    obj.state.biasP(2,2) = obj.state.biasP(2,2) + obj.coeffs.biasV;
                end
                if obj.state.biasP(3,3) < obj.coeffs.biasP0
                    obj.state.biasP(3,3) = obj.state.biasP(3,3) +obj.coeffs.biasV;
                end
                if w(1) >= 0
                    % clip disagreement to -2..2 °/s
                    % (this also effectively limits the harm done by the first inclination correction step)
                    e = obj.clip(e, -biasClip, biasClip);

                    % step 2: K = P R^T inv(W + R P R^T)
                    %K = (obj.state.biasP * R') * inv(diag(w) + R*obj.state.biasP*R');
                    K = (obj.state.biasP * R') / (diag(w) + R*obj.state.biasP*R');

                    % step 3: bias = bias + K (y - R bias) = bias + K e
                    bias = bias + (K * e')';

                    % step 4: P = P - K R P
                    obj.state.biasP = obj.state.biasP - K*R*obj.state.biasP;

                    % clip bias estimate to -2..2 °/s
                    obj.state.bias = obj.clip(bias, -biasClip, biasClip);
                end
            end
        end
        function updateMag(obj, mag)
            % Performs magnetometer update step.
            %
            % It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
            % different sampling rates. Otherwise, simply use :meth:`update`.
            %
            % Should be called after :meth:`updateAcc`.
            %
            % :param mag: magnetometer measurement in arbitrary units -- 1x3 matrix
            assert(all(size(mag) == [1, 3]), 'mag has wrong size');

            % ignore [0 0 0] samples
            if all(mag == 0)
                return
            end

            magTs = obj.coeffs.magTs;

            % bring magnetometer measurement into 6D earth frame
            magEarth = obj.quatRotate(obj.getQuat6D(), mag);

            if obj.params.magDistRejectionEnabled
                magNormDip = zeros(1, 2);
                magNormDip(1) = norm(magEarth);
                magNormDip(2) = -asin(magEarth(3)/magNormDip(1));

                if obj.params.magCurrentTau > 0
                    [magNormDip, obj.state.magNormDipLpState] = obj.filterVec(magNormDip, obj.params.magCurrentTau, ...
                        magTs, obj.coeffs.magNormDipLpB, obj.coeffs.magNormDipLpA, obj.state.magNormDipLpState);
                end
                obj.state.magNormDip = magNormDip;

                % magnetic disturbance detection
                if abs(magNormDip(1) - obj.state.magRefNorm) < obj.params.magNormTh*obj.state.magRefNorm && ...
                        abs(magNormDip(2) - obj.state.magRefDip) < obj.params.magDipTh*pi/180.0
                    obj.state.magUndisturbedT = obj.state.magUndisturbedT + magTs;
                    if obj.state.magUndisturbedT >= obj.params.magMinUndisturbedTime
                        obj.state.magDistDetected = false;
                        obj.state.magRefNorm = obj.state.magRefNorm + obj.coeffs.kMagRef*(magNormDip(1) - obj.state.magRefNorm);
                        obj.state.magRefDip = obj.state.magRefDip + obj.coeffs.kMagRef*(magNormDip(2) - obj.state.magRefDip);
                    end
                else
                    obj.state.magUndisturbedT = 0.0;
                    obj.state.magDistDetected = true;
                end

                % new magnetic field acceptance
                if abs(magNormDip(1) - obj.state.magCandidateNorm) < obj.params.magNormTh*obj.state.magCandidateNorm && ...
                        abs(magNormDip(2) - obj.state.magCandidateDip) < obj.params.magDipTh*pi/180.0
                    if norm(obj.state.restLastGyrLp) >= obj.params.magNewMinGyr*pi/180.0
                        obj.state.magCandidateT = obj.state.magCandidateT + magTs;
                    end
                    obj.state.magCandidateNorm = obj.state.magCandidateNorm + obj.coeffs.kMagRef*(magNormDip(1) - obj.state.magCandidateNorm);
                    obj.state.magCandidateDip = obj.state.magCandidateDip + obj.coeffs.kMagRef*(magNormDip(2) - obj.state.magCandidateDip);

                    if obj.state.magDistDetected &&  (obj.state.magCandidateT >= obj.params.magNewTime || ...
                            (obj.state.magRefNorm == 0.0 && obj.state.magCandidateT >= obj.params.magNewFirstTime))
                        obj.state.magRefNorm = obj.state.magCandidateNorm;
                        obj.state.magRefDip = obj.state.magCandidateDip;
                        obj.state.magDistDetected = false;
                        obj.state.magUndisturbedT = obj.params.magMinUndisturbedTime;
                    end
                else
                    obj.state.magCandidateT = 0.0;
                    obj.state.magCandidateNorm = magNormDip(1);
                    obj.state.magCandidateDip = magNormDip(2);
                end
            end

            % calculate disagreement angle based on current magnetometer measurement
            obj.state.lastMagDisAngle = atan2(magEarth(1), magEarth(2)) - obj.state.delta;

            % make sure the disagreement angle is in the range [-pi, pi]
            if obj.state.lastMagDisAngle > pi
                obj.state.lastMagDisAngle = obj.state.lastMagDisAngle - 2*pi;
            elseif obj.state.lastMagDisAngle < -pi
                obj.state.lastMagDisAngle = obj.state.lastMagDisAngle + 2*pi;
            end

            k = obj.coeffs.kMag;

            if obj.params.magDistRejectionEnabled
                % magnetic disturbance rejection
                if obj.state.magDistDetected
                    if obj.state.magRejectT <= obj.params.magMaxRejectionTime
                        obj.state.magRejectT = obj.state.magRejectT + magTs;
                        k = 0;
                    else
                        k = k/obj.params.magRejectionFactor;
                    end
                else
                    obj.state.magRejectT = max(obj.state.magRejectT - obj.params.magRejectionFactor*magTs, 0.0);
                end
            end

            % ensure fast initial convergence
            if obj.state.kMagInit ~= 0.0
                % make sure that the gain k is at least 1/N, N=1,2,3,... in the first few samples
                if k < obj.state.kMagInit
                    k = obj.state.kMagInit;
                end

                % iterative expression to calculate 1/N
                obj.state.kMagInit = obj.state.kMagInit/(obj.state.kMagInit+1);

                % disable if t > tauMag
                if obj.state.kMagInit*obj.params.tauMag < magTs
                    obj.state.kMagInit = 0.0;
                end
            end

            % first-order filter step
            obj.state.delta = obj.state.delta + k*obj.state.lastMagDisAngle;
            % calculate correction angular rate to facilitate debugging
            obj.state.lastMagCorrAngularRate = k*obj.state.lastMagDisAngle/obj.coeffs.magTs;

            % make sure delta is in the range [-pi, pi]
            if obj.state.delta > pi
                obj.state.delta = obj.state.delta - 2*pi;
            elseif obj.state.delta < -pi
                obj.state.delta = obj.state.delta + 2*pi;
            end
        end
        function update(obj, gyr, acc, mag)
            % Performs filter update step for one sample.
            %
            % :param gyr: gyr gyroscope measurement in rad/s -- 1x3 matrix
            % :param acc: acc accelerometer measurement in m/s² -- 1x3 matrix
            % :param mag: optional mag magnetometer measurement in arbitrary units -- 1x3 matrix
            obj.updateGyr(gyr);
            obj.updateAcc(acc);
            if nargin >= 4
                obj.updateMag(mag);
            end
        end
        function out = updateBatch(obj, gyr, acc, mag)
            % Performs batch update for multiple samples at once.
            %
            % In order to use this function, all input data must have the same sampling rate and be contained in Nx3
            % matrices. The output is a struct containing
            %
            % - **quat6D** -- the 6D quaternion -- Nx4 matrix
            % - **bias** -- gyroscope bias estimate -- Nx4 matrix
            % - **biasSigma** -- uncertainty of gyroscope bias estimate in rad/s -- Nx1 matrix
            % - **restDetected** -- rest detection state -- boolean Nx1 matrix
            %
            % in all cases and if magnetometer data is provided additionally
            %
            % - **quat9D** -- the 9D quaternion -- Nx4 matrix
            % - **delta** -- heading difference angle between 6D and 9D quaternion in rad -- Nx1 matrix
            % - **magDistDetected** -- magnetic disturbance detection state -- Nx1 boolean matrix
            %
            % :param gyr: gyroscope measurement in rad/s -- Nx3 matrix
            % :param acc: accelerometer measurement in m/s² -- Nx3 matrix
            % :param mag: optional magnetometer measurement in arbitrary units -- Nx3 matrix
            % :return: struct with entries as described above
            N = size(gyr, 1);
            assert(all(size(gyr) == [N, 3]), 'gyr has wrong size');
            assert(all(size(acc) == [N, 3]), 'acc has wrong size');

            out6D = zeros(N, 4);
            outBias = zeros(N, 3);
            outBiasSigma = zeros(N, 1);
            outRest = zeros(N, 1);

            if nargin >= 4
                assert(all(size(mag) == [N, 3]), 'mag has wrong size');
                out9D = zeros(N, 4);
                outDelta = zeros(N, 1);
                outMagDist = zeros(N, 1);
                for i=1:N
                    obj.update(gyr(i,:), acc(i,:), mag(i,:));
                    out6D(i,:) = obj.getQuat6D();
                    out9D(i,:) = obj.getQuat9D();
                    outDelta(i) = obj.state.delta;
                    [outBias(i,:), outBiasSigma(i)] = obj.getBiasEstimate();
                    outRest(i) = obj.state.restDetected;
                    outMagDist(i) = obj.state.magDistDetected;
                end
                out = struct('quat6D', out6D, 'quat9D', out9D, 'delta', outDelta,...
                    'bias', outBias, 'biasSigma', outBiasSigma, 'restDetected', outRest, ...
                    'magDistDetected', outMagDist);
            else
                for i=1:N
                    obj.update(gyr(i,:), acc(i,:));
                    out6D(i,:) = obj.getQuat6D();
                    [outBias(i,:), outBiasSigma(i)] = obj.getBiasEstimate();
                    outRest(i) = obj.state.restDetected;
                end
                out = struct('quat6D', out6D, 'bias', outBias, 'biasSigma', outBiasSigma, ...
                    'restDetected', outRest);
            end
        end
        function out = getQuat3D(obj)
            % Returns the angular velocity strapdown integration quaternion
            % :math:`^{\mathcal{S}_i}_{\mathcal{I}_i}\mathbf{q}`.
            %
            % :return: quaternion as 4x1 matrix
            out = obj.state.gyrQuat;
        end
        function out = getQuat6D(obj)
            % Returns the 6D (magnetometer-free) orientation quaternion
            % :math:`^{\mathcal{S}_i}_{\mathcal{E}_i}\mathbf{q}`.
            %
            % :return: quaternion as 4x1 matrix
            out = obj.quatMultiply(obj.state.accQuat, obj.state.gyrQuat);
        end
        function out = getQuat9D(obj)
            % Returns the 9D (with magnetometers) orientation quaternion
            % :math:`^{\mathcal{S}_i}_{\mathcal{E}}\mathbf{q}`.
            %
            % :return: quaternion as 4x1 matrix
            out = obj.quatMultiply(obj.state.accQuat, obj.state.gyrQuat);
            out = obj.quatApplyDelta(out, obj.state.delta);
        end
        function out = getDelta(obj)
            % Returns the heading difference :math:`\delta` between :math:`\mathcal{E}_i` and :math:`\mathcal{E}`.
            %
            % :math:`^{\mathcal{E}_i}_{\mathcal{E}}\mathbf{q} = \begin{bmatrix}\cos\frac{\delta}{2} & 0 & 0 &
            % \sin\frac{\delta}{2}\end{bmatrix}^T`.
            %
            % :return: delta angle in rad (:cpp:member:`VQFState::delta`)
            out = obj.state.delta;
        end
        function [bias, sigma] = getBiasEstimate(obj)
            % Returns the current gyroscope bias estimate and the uncertainty.
            %
            % The returned standard deviation sigma represents the estimation uncertainty in the worst direction and is
            % based on an upper bound of the largest eigenvalue of the covariance matrix.
            %
            % :return: gyroscope bias estimate (rad/s) as 4x1 matrix and standard deviation sigma of the estimation
            %     uncertainty (rad/s)
            bias = obj.state.bias;
            % use largest absolute row sum as upper bound estimate for largest eigenvalue (Gershgorin circle theorem)
            % and clip output to biasSigmaInit
            P = min(max(sum(abs(obj.state.biasP), 2)), obj.coeffs.biasP0);
            sigma = sqrt(P)*pi/100.0/180.0;
        end
        function setBiasEstimate(obj, bias, sigma)
            % Sets the current gyroscope bias estimate and the uncertainty.
            %
            % If a value for the uncertainty sigma is given, the covariance matrix is set to a corresponding scaled
            % identity matrix.
            %
            % :param bias: gyroscope bias estimate (rad/s)
            % :param sigma: standard deviation of the estimation uncertainty (rad/s) - set to -1 (default) in order to
            %     not change the estimation covariance matrix
            assert(all(size(bias) == [1, 3]), 'b has wrong size');
            obj.state.bias = bias;
            if nargin < 3 || sigma > 0
                assert(all(size(sigma) == [1, 1]), 'sigma has wrong size');
                obj.state.biasP = (sigma*180.0*100.0/pi)^2 * eye(3);
            end
        end
        function rest = getRestDetected(obj)
            % Returns true if rest was detected.
            rest = obj.state.restDetected;
        end
        function magDist = getMagDistDetected(obj)
            % Returns true if a disturbed magnetic field was detected.
            magDist = obj.state.magDistDetected;
        end
        function out = getRelativeRestDeviations(obj)
            % Returns the relative deviations used in rest detection.
            %
            % Looking at those values can be useful to understand how rest detection is working and which thresholds are
            % suitable. The output array is filled with the last values for gyroscope and accelerometer,
            % relative to the threshold. In order for rest to be detected, both values must stay below 1.
            %
            % :return: relative rest deviations as 2x1 matrix

            out = [
                sqrt(obj.state.restLastSquaredDeviations(1)) / (obj.params.restThGyr*pi/180.0),
                sqrt(obj.state.restLastSquaredDeviations(2)) / obj.params.restThAcc
            ];
        end
        function magRefNorm = getMagRefNorm(obj)
            % Returns the norm of the currently accepted magnetic field reference.
            magRefNorm = obj.state.magRefNorm;
        end
        function magRefDip = getMagRefDip(obj)
            % Returns the dip angle of the currently accepted magnetic field reference.
            magRefDip = obj.state.magRefDip;
        end
        function setMagRef(obj, norm, dip)
            % Overwrites the current magnetic field reference.
            %
            % :param norm: norm of the magnetic field reference
            % :param dip: dip angle of the magnetic field reference
            assert(all(size(norm) == [1, 1]), 'norm has wrong size');
            assert(all(size(dip) == [1, 1]), 'dip has wrong size');
            obj.state.magRefNorm = norm;
            obj.state.magRefDip = dip;
        end
        function setTauAcc(obj, tauAcc)
            % Sets the time constant for accelerometer low-pass filtering.
            %
            % For more details, see :cpp:member:`VQFParams::tauAcc`.
            %
            % :param tauAcc: time constant :math:`\tau_\mathrm{acc}` in seconds
            if obj.params.tauAcc == tauAcc
                return;
            end
            obj.params.tauAcc = tauAcc;
            [newB, newA] = obj.filterCoeffs(obj.params.tauAcc, obj.coeffs.accTs);

            obj.state.accLpState = obj.filterAdaptStateForCoeffChange(obj.state.lastAccLp, ...
                obj.coeffs.accLpB, obj.coeffs.accLpA, newB, newA, obj.state.accLpState);

            % For R and biasLP, the last value is not saved in the state.
            % Since b0 is small (at reasonable settings), the last output is close to state(1).
            obj.state.motionBiasEstRLpState = obj.filterAdaptStateForCoeffChange(obj.state.motionBiasEstRLpState(1,:), ...
                obj.coeffs.accLpB, obj.coeffs.accLpA, newB, newA, obj.state.motionBiasEstRLpState);
            obj.state.motionBiasEstBiasLpState = obj.filterAdaptStateForCoeffChange(obj.state.motionBiasEstBiasLpState(1,:), ...
                obj.coeffs.accLpB, obj.coeffs.accLpA, newB, newA, obj.state.motionBiasEstBiasLpState);

            obj.coeffs.accLpB = newB;
            obj.coeffs.accLpA = newA;
        end
        function setTauMag(obj, tauMag)
            % Sets the time constant for the magnetometer update.
            %
            % For more details, see :cpp:member:`VQFParams::tauMag`.
            %
            % :param tauMag: time constant :math:`\tau_\mathrm{mag}` in seconds
            obj.params.tauMag = tauMag;
            obj.coeffs.kMag = obj.gainFromTau(obj.params.tauMag, obj.coeffs.magTs);
        end
        function setMotionBiasEstEnabled(obj, enabled)
            % Enables/disabled gyroscope bias estimation during motion.
            if obj.params.motionBiasEstEnabled == enabled
                return;
            end
            obj.params.motionBiasEstEnabled = enabled;
            obj.state.motionBiasEstRLpState = NaN(2, 9);
            obj.state.motionBiasEstBiasLpState = NaN(2, 2);
        end
        function setRestBiasEstEnabled(obj, enabled)
            % Enables/disables rest detection and bias estimation during rest.
            if obj.params.restBiasEstEnabled == enabled
                return;
            end
            obj.params.restBiasEstEnabled = enabled;
            obj.state.restDetected = false;
            obj.state.restLastSquaredDeviations = [0 0 0];
            obj.state.restT = 0.0;
            obj.state.restLastGyrLp = [0 0 0];
            obj.state.restGyrLpState = NaN(2, 3);
            obj.state.restLastAccLp = [0 0 0];
            obj.state.restAccLpState = NaN(2, 3);
        end
        function setMagDistRejectionEnabled(obj, enabled)
            % Enables/disables magnetic disturbance detection and rejection.
            if obj.params.magDistRejectionEnabled == enabled
                return;
            end
            obj.params.magDistRejectionEnabled = enabled;
            obj.state.magDistDetected = true;
            obj.state.magRefNorm = 0.0;
            obj.state.magRefDip = 0.0;
            obj.state.magUndisturbedT = 0.0;
            obj.state.magRejectT = obj.params.magMaxRejectionTime;
            obj.state.magCandidateNorm = -1.0;
            obj.state.magCandidateDip = 0.0;
            obj.state.magCandidateT = 0.0;
            obj.state.magNormDip = [0 0];
            obj.state.magNormDipLpState = NaN(2, 2);
        end
        function setRestDetectionThresholds(obj, thGyr, thAcc)
            % Sets the current thresholds for rest detection.
            %
            % :param thGyr: new value for :cpp:member:`VQFParams::restThGyr`
            % :param thAcc: new value for :cpp:member:`VQFParams::restThAcc`
            obj.params.restThGyr = thGyr;
            obj.params.restThAcc = thAcc;
        end
        function resetState(obj)
            % Resets the state to the default values at initialization.
            %
            % Resetting the state is equivalent to creating a new instance of this class.
            obj.state.gyrQuat = [1 0 0 0];
            obj.state.accQuat = [1 0 0 0];
            obj.state.delta = 0;

            obj.state.restDetected = false;
            obj.state.magDistDetected = true;

            obj.state.lastAccLp = [0 0 0];
            obj.state.accLpState = NaN(2, 3);
            obj.state.lastAccCorrAngularRate = 0.0;

            obj.state.kMagInit = 1.0;
            obj.state.lastMagDisAngle = 0.0;
            obj.state.lastMagCorrAngularRate = 0.0;

            obj.state.bias = [0 0 0];
            obj.state.biasP = obj.coeffs.biasP0*eye(3);

            obj.state.motionBiasEstRLpState = NaN(2, 9);
            obj.state.motionBiasEstBiasLpState = NaN(2, 2);

            obj.state.restLastSquaredDeviations = [0 0 0];
            obj.state.restT = 0.0;
            obj.state.restLastGyrLp = [0 0 0];
            obj.state.restGyrLpState = NaN(2, 3);
            obj.state.restLastAccLp = [0 0 0];
            obj.state.restAccLpState = NaN(2, 3);

            obj.state.magRefNorm = 0.0;
            obj.state.magRefDip = 0.0;
            obj.state.magUndisturbedT = 0.0;
            obj.state.magRejectT = obj.params.magMaxRejectionTime;
            obj.state.magCandidateNorm = -1.0;
            obj.state.magCandidateDip = 0.0;
            obj.state.magCandidateT = 0.0;
            obj.state.magNormDip = [0 0];
            obj.state.magNormDipLpState = NaN(2, 2);
        end
    end
    methods(Static)
        function out = quatMultiply(q1, q2)
            % Performs quaternion multiplication (:math:`\mathbf{q}_\mathrm{out} = \mathbf{q}_1 \otimes \mathbf{q}_2`).
            %
            % :param q1: input quaternion 1 -- 4x1 matrix
            % :param q2: input quaternion 2 -- 4x1 matrix
            % :return: output quaternion -- 4x1 matrix
            w = q1(1) * q2(1) - q1(2) * q2(2) - q1(3) * q2(3) - q1(4) * q2(4);
            x = q1(1) * q2(2) + q1(2) * q2(1) + q1(3) * q2(4) - q1(4) * q2(3);
            y = q1(1) * q2(3) - q1(2) * q2(4) + q1(3) * q2(1) + q1(4) * q2(2);
            z = q1(1) * q2(4) + q1(2) * q2(3) - q1(3) * q2(2) + q1(4) * q2(1);
            out = [w x y z];
        end
        function out = quatConj(q)
            % Calculates the quaternion conjugate (:math:`\mathbf{q}_\mathrm{out} = \mathbf{q}^*`).
            %
            % :param q: input quaternion -- 4x1 matrix
            % :return: output quaternion -- 4x1 matrix
            out = [q(1) -q(2) -q(3) -q(4)];
        end
        function out = quatApplyDelta(q, delta)
            % Applies a heading rotation by the angle delta (in rad) to a quaternion.
            %
            % :math:`\mathbf{q}_\mathrm{out} = \begin{bmatrix}\cos\frac{\delta}{2} & 0 & 0 &
            % \sin\frac{\delta}{2}\end{bmatrix} \otimes \mathbf{q}`
            %
            % :param q: input quaternion -- 4x1 matrix
            % :param delta: heading rotation angle in rad
            % :return: output quaternion -- 4x1 matrix
            c = cos(delta/2);
            s = sin(delta/2);
            w = c * q(1) - s * q(4);
            x = c * q(2) - s * q(3);
            y = c * q(3) + s * q(2);
            z = c * q(4) + s * q(1);
            out = [w x y z];
        end
        function out = quatRotate(q, v)
            % Rotates a vector with a given quaternion.
            %
            % :math:`\begin{bmatrix}0 & \mathbf{v}_\mathrm{out}\end{bmatrix}
            % = \mathbf{q} \otimes \begin{bmatrix}0 & \mathbf{v}\end{bmatrix} \otimes \mathbf{q}^*`
            %
            % :param q: input quaternion -- 4x1 matrix
            % :param v: input vector -- 3x1 matrix
            % :return: output vector -- 3x1 matrix
            x = (1 - 2*q(3)*q(3) - 2*q(4)*q(4))*v(1) + 2*v(2)*(q(3)*q(2) - q(1)*q(4)) + 2*v(3)*(q(1)*q(3) + q(4)*q(2));
            y = 2*v(1)*(q(1)*q(4) + q(3)*q(2)) + v(2)*(1 - 2*q(2)*q(2) - 2*q(4)*q(4)) + 2*v(3)*(q(3)*q(4) - q(2)*q(1));
            z = 2*v(1)*(q(4)*q(2) - q(1)*q(3)) + 2*v(2)*(q(1)*q(2) + q(4)*q(3)) + v(3)*(1 - 2*q(2)*q(2) - 2*q(3)*q(3));
        	out = [x y z];
        end
        function vec = normalize(vec)
            % Normalizes a vector.
            %
            % :param vec: input vector -- Nx1 matrix
            % :return: normalized vector -- Nx1 matrix
            n = norm(vec);
            if n ~= 0.0
                vec = vec/n;
            end
        end
        function vec = clip(vec, min, max)
            % Clips a vector.
            %
            % :param vec: input vector -- Nx1 matrix
            % :param min_: smallest allowed value
            % :param max_: largest allowed value
            % :return: clipped vector -- Nx1 matrix
            vec(vec<min) = min;
            vec(vec>max) = max;
        end
        function k = gainFromTau(tau, Ts)
            % Calculates the gain for a first-order low-pass filter from the 1/e time constant.
            %
            % :math:`k = 1 - \exp\left(-\frac{T_\mathrm{s}}{\tau}\right)`
            %
            % The cutoff frequency of the resulting filter is :math:`f_\mathrm{c} = \frac{1}{2\pi\tau}`.
            %
            % :param tau: time constant :math:`\tau` in seconds - use -1 to disable update (:math:`k=0`) or 0 to obtain
            %     unfiltered values (:math:`k=1`)
            % :param Ts: sampling time :math:`T_\mathrm{s}` in seconds
            % :return: filter gain *k*
            assert(Ts > 0, 'Ts must be positive');
            if tau < 0.0
                k = 0;  % k=0 for negative tau (disable update)
            elseif tau == 0.0
                k = 1;  % k=1 for tau=0
            else
                k = 1 - exp(-Ts/tau);  % fc = 1/(2*pi*tau)
            end
        end
        function [b, a] = filterCoeffs(tau, Ts)
            % Calculates coefficients for a second-order Butterworth low-pass filter.
            %
            % The filter is parametrized via the time constant of the dampened, non-oscillating part of step response
            % and the resulting cutoff frequency is :math:`f_\mathrm{c} = \frac{\sqrt{2}}{2\pi\tau}`.
            %
            % :param tau: time constant :math:`\tau` in seconds
            % :param Ts: sampling time :math:`T_\mathrm{s}` in seconds
            % :return: numerator coefficients b as 1x3 matrix, denominator coefficients a (without :math:`a_0=1`) as
            %     2x1 matrix
            assert(tau > 0, 'tau must be positive');
            assert(Ts > 0, 'Ts must be positive');
            % second order Butterworth filter based on https://stackoverflow.com/a/52764064
            fc = sqrt(2) / (2.0*pi*tau);  % time constant of dampened, non-oscillating part of step response
            C = tan(pi*fc*Ts);
            D = C*C + sqrt(2)*C + 1;
            b0 = C*C/D;
            b1 = 2*b0;
            b2 = b0;
            % a0 = 1.0
            a1 = 2*(C*C-1)/D;
            a2 = (1-sqrt(2)*C+C*C)/D;
            b = [b0 b1 b2];
            a = [a1 a2];
        end
        function out = filterInitialState(x0, b, a)
            % Calculates the initial filter state for a given steady-state value.
            %
            % :param x0: steady state value
            % :param b: numerator coefficients
            % :param a: denominator coefficients (without :math:`a_0=1`)
            % :return: filter state -- 1x2 matrix

            % initial state for steady state (equivalent to scipy.signal.lfilter_zi, obtained by setting y=x=x0 in the
            % filter update equation)
            assert(all(size(b) == [1, 3]), 'b has wrong size');
            assert(all(size(a) == [1, 2]), 'a has wrong size');
            out = [x0*(1 - b(1)), x0*(b(3) - a(2))];
        end
        function state = filterAdaptStateForCoeffChange(last_y, b_old, a_old, b_new, a_new, state)
            % r"""Adjusts the filter state when changing coefficients.
            %
            % This function assumes that the filter is currently in a steady state, i.e. the last input values and the last
            % output values are all equal. Based on this, the filter state is adjusted to new filter coefficients so that the
            % output does not jump.
            %
            % :param last_y: last filter output values -- 1xN matrix
            % :param b_old: previous numerator coefficients --1x3 matrix
            % :param a_old: previous denominator coefficients (without :math:`a_0=1`) -- 1x2 matrix
            % :param b_new: new numerator coefficients -- 1x3 matrix
            % :param a_new: new denominator coefficients (without :math:`a_0=1`) -- 1x2 matrix
            % :param state: input filter state -- 2xN matrix
            % :return: modified filter state -- 2xN matrix
            N = length(last_y);
            assert(all(size(last_y) == [1, N]), 'last_y has wrong size');
            assert(all(size(b_old) == [1, 3]), 'b_old has wrong size');
            assert(all(size(a_old) == [1, 2]), 'a_old has wrong size');
            assert(all(size(b_new) == [1, 3]), 'b_new has wrong size');
            assert(all(size(a_new) == [1, 2]), 'a_new has wrong size');
            assert(all(size(state) == [2, N]), 'state has wrong size');

            if isnan(state(1,1))
                return;
            end

            state(1,:) = state(1,:) + (b_old(1) - b_new(1))*last_y;
            state(2,:) = state(2,:) + (b_old(2) - b_new(2) - a_old(1) + a_new(1))*last_y;
        end
        function [y, state] = filterStep(x, b, a, state)
            % Performs a filter step.
            %
            % Note: Unlike the C++ implementation, this function is vectorized and can process multiple values at once.
            %
            % :param x: input values -- 1xN matrix
            % :param b: numerator coefficients -- 1x3 matrix
            % :param a: denominator coefficients (without :math:`a_0=1`) -- 1x2 matrix
            % :param state: filter state -- 2xN matrix
            % :return: filtered values as 1xN matrix and new filter state as 2xN matrix

            % this function is vectorized, unlike the C++ version
            % difference equations based on scipy.signal.lfilter documentation
            % assumes that a0 == 1.0
            y = b(1)*x + state(1,:);
            state(1,:) = b(2)*x - a(1)*y + state(2,:);
            state(2,:) = b(3)*x - a(2)*y;
        end
        function [out, state] = filterVec(x, tau, Ts, b, a, state)
            % Performs filter step for vector-valued signal with averaging-based initialization.
            %
            % During the first :math:`\tau` seconds, the filter output is the mean of the previous samples. At :math:`t=\tau`,
            % the initial conditions for the low-pass filter are calculated based on the current mean value and from then on,
            % regular filtering with the rational transfer function described by the coefficients b and a is performed.
            %
            % :param x: input values -- 1xN matrix
            % :param tau: filter time constant \:math:`\tau` in seconds (used for initialization)
            % :param Ts: sampling time :math:`T_\mathrm{s}` in seconds (used for initialization)
            % :param b: numerator coefficients -- 1x3 matrix
            % :param a: denominator coefficients (without :math:`a_0=1`) -- 1x2 matrix
            % :param state: filter state -- 2xN matrix
            % :return: filtered values as 1xN matrix and new filter state as 2xN matrix

            N = size(x, 2);

            % to avoid depending on a single sample, average the first samples (for duration tau)
            % and then use this average to calculate the filter initial state
            if isnan(state(1,1))  % initialization phase
                if isnan(state(1,2))  % first sample
                    state(1,2) = 0;  % state(2) is used to store the sample count
                    state(2,:) = 0;  % state(2+i) is used to store the sum
                end
                state(1,2) = state(1,2) + 1;
                state(2,:) = state(2,:) + x;
                out = state(2,:)/state(1,2);
                if state(1,2)*Ts >= tau
                    for i=1:N
                        state(:,i) = VQF.filterInitialState(out(i), b, a);
                    end
                end
                return;
            end

            [out, state] = VQF.filterStep(x, b, a, state);
        end
    end
    methods(Access=protected)
        function setup(obj)
            % Calculates coefficients based on parameters and sampling rates.
            assert(obj.coeffs.gyrTs > 0);
            assert(obj.coeffs.accTs > 0);
            assert(obj.coeffs.magTs > 0);

            [obj.coeffs.accLpB, obj.coeffs.accLpA] = obj.filterCoeffs(obj.params.tauAcc, obj.coeffs.magTs);

            obj.coeffs.kMag = obj.gainFromTau(obj.params.tauMag, obj.coeffs.magTs);

            obj.coeffs.biasP0 = (obj.params.biasSigmaInit*100.0)^2;
            % the system noise increases the variance from 0 to (0.1 °/s)^2 in biasForgettingTime seconds
            obj.coeffs.biasV = (0.1*100.0)^2*obj.coeffs.accTs/obj.params.biasForgettingTime;

            pMotion = (obj.params.biasSigmaMotion*100.0)^2;
            obj.coeffs.biasMotionW = pMotion^2 / obj.coeffs.biasV + pMotion;
            obj.coeffs.biasVerticalW = obj.coeffs.biasMotionW / max(obj.params.biasVerticalForgettingFactor, 1e-10);

            pRest = (obj.params.biasSigmaRest*100.0)^2;
            obj.coeffs.biasRestW = pRest^2 / obj.coeffs.biasV + pRest;

            [obj.coeffs.restGyrLpB, obj.coeffs.restGyrLpA] = obj.filterCoeffs(obj.params.restFilterTau, obj.coeffs.gyrTs);
            [obj.coeffs.restAccLpB, obj.coeffs.restAccLpA] = obj.filterCoeffs(obj.params.restFilterTau, obj.coeffs.accTs);

            obj.coeffs.kMagRef = obj.gainFromTau(obj.params.magRefTau, obj.coeffs.magTs);
            if obj.params.magCurrentTau > 0
                [obj.coeffs.magNormDipLpB, obj.coeffs.magNormDipLpA] = obj.filterCoeffs(obj.params.magCurrentTau, obj.coeffs.magTs);
            else
                obj.coeffs.magNormDipLpB = NaN(1, 3);
                obj.coeffs.magNormDipLpA = NaN(1, 2);
            end

            obj.resetState();
        end
    end
end
