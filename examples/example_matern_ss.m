% GP regression using state-space formulation of Matern CF example
%
% Implementation of Gaussian Process (GP) regression based on the state-
% space formulation of spatio-temporal GPs, see [1], [2].
%
% [1] J. Hartikainen and S. Sarkka, "Kalman filtering and smoothing 
%     solutions to temporal Gaussian process regression models," in IEEE 
%     International Workshop on Machine Learning for Signal Processing 
%     (MLSP), pp. 379-384, August 2010.
%
% [2] A. Carron, M. Todescato, R. Carli, L. Schenato, and G. Pillonetto, 
%     "Machine learning meets Kalman filtering," in 55th IEEE Conference on
%     Decision and Control (CDC), pp. 4594-4599, December 2016.
% 
% Copyright (C) 2018 Roland Hostettler <roland.hostettler@aalto.fi>
% 
% This file is part of the libgp Matlab toolbox.
%
% libgp is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free 
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
% 
% libgp is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
% details.
% 
% You should have received a copy of the GNU General Public License along 
% with libgp. If not, see <http://www.gnu.org/licenses/>.

% Housekeeping
clear variables;
addpath ../src;

%% Parameters
ellx = 2;           % Spatial length scale
sigma2t = 0.5;      % Variance
ellt = 1;           % Temporal length scale
R = 0.01;           % Measurement noise variance

T = 100;            % Simulation time
Ts = 0.25;          % Sampling time

NTrain = 4;         % No. of spatial training points
NTest = 4;          % No. of spatial test points

% Spatial training and test points, randomly generated
xt = randn(2, NTrain);
xp = randn(2, NTest);

%% Batch model
% u = [x; t]
k_f = @(u1, u2) ( ...
    gpk_matern(u1(1:2, :), u2(1:2, :), ellx, 1, 2.5) ...
    .*gpk_matern(u1(3, :), u2(3, :), ellt, sigma2t, 2.5) ...
);

%% State space model
% Convert temporal covaraince matrix to state space system
gpk_t = @(t) gpk_matern_ss(ellt, sigma2t, 2.5);
gpk_u = @(x1, x2) gpk_matern(x1, x2, ellx, 1, 2.5);
[F, Q, C, m0, P0, N0] = gp_model_ss(gpk_t, gpk_u, xt, xp, Ts);

%% Generate data
t = 0:Ts:T;
N = length(t);
ut = [];
up = [];
for iTrain = 1:NTrain
    ut = [ut, [xt(:, iTrain)*ones(1, N); t]];
end
for iTest = 1:NTest
    up = [up, [xp(:, iTest)*ones(1, N); t]];
end

% Calculate covariance matrices (also for test data and cross-covariance,
% for simplicity)
Kuu = gp_calculate_covariance([ut, up], k_f);
Ktt = Kuu(1:NTrain*N, 1:NTrain*N);

% Draw samples
f = chol(Ktt).'*randn(NTrain*N, 1);
y = f + sqrt(R)*randn(NTrain*N, 1);

%% Batch prediction
Kpt = Kuu(NTrain*N+1:NTrain*N+NTest*N, 1:NTrain*N);
Kpp = Kuu(NTrain*N+1:NTrain*N+NTest*N, NTrain*N+1:NTrain*N+NTest*N);
tic;
fp_batch = Kpt/(Ktt + R*eye(NTrain*N))*y;
Cp = Kpp - Kpt/(Ktt + R*eye(NTrain*N))*Kpt';
toc;

fp_batch = reshape(fp_batch, [N, NTest]).';

%% State space prediction
y = reshape(y, [N, NTrain]).';
Nx = size(F, 1);

mp_s = zeros(Nx, N);
Pp_s = zeros(Nx, Nx, N);
Cp_s = zeros(Nx, Nx, N);
m_s = zeros(Nx, N);
P_s = zeros(Nx, Nx, N);

% Filtering
tic;
m = m0;
P = P0;
for n = 1:N
    % Prediction
    mp = F*m;
    Pp = F*P*F' + Q;
    Cp = F*P;
    
    % Measurement update
    S = C*Pp*C' + R*eye(NTrain);
    K = (Pp*C')/S;
    m = mp + K*(y(:, n) - C*mp);
    P = Pp - K*S*K';
    
    % Store
    mp_s(:, n) = mp;
    Pp_s(:, :, n) = Pp;
    Cp_s(:, :, n) = Cp;
    m_s(:, n) = m;
    P_s(:, :, n) = P;
end

% Smoothing
ms(:, N) = m;
Ps(:, :, N) = P;
for n = N-1:-1:1
    L = Cp_s(:, :, n+1)'/Pp_s(:, :, n+1);
    ms(:, n) = m_s(:, n) + L*(ms(:, n+1) - mp_s(:, n+1));
    Ps(:, :, n) = P_s(:, :, n) + L*(Ps(:, :, n+1) - Pp_s(:, :, n+1))*L';
end
toc;

% Get the prediction from the smoothed state vector
fp_kf = ms(3*NTrain+1:3:3*(NTrain+NTest), :);

%% Illustration
for i = 1:NTest
    figure(i); clf();
    plot(t, fp_batch(i, :)); hold on;
    plot(t, fp_kf(i, :));
    legend('Batch', 'State space');
end
