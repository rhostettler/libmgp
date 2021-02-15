% GP regression using state-space formulation of Matern covariance function
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
% 2018-present -- Roland Hostettler

%{
% This file is part of the libmgp toolbox.
%
% libmgp is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free 
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
% 
% libmgp is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
% details.
% 
% You should have received a copy of the GNU General Public License along 
% with libmgp. If not, see <http://www.gnu.org/licenses/>.
%}

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

Nt = 4;         % No. of spatial training points
Np = 4;          % No. of spatial test points

% Spatial training and test points, randomly generated
xt = randn(2, Nt);
xp = randn(2, Np);

%% Batch model
% u = [x; t]
k_f = @(u1, u2) ( ...
    mgp.k_matern(u1(1:2, :), u2(1:2, :), ellx, 1, 2.5) ...
    .*mgp.k_matern(u1(3, :), u2(3, :), ellt, sigma2t, 2.5) ...
);

%% State space model
% Convert temporal covaraince matrix to state space system
k_t = @() mgp.k_matern_ss(ellt, sigma2t, 2.5);
k_u = @(x1, x2) mgp.k_matern(x1, x2, ellx, 1, 2.5);
[F, Q, C, m0, P0, dxtime] = mgp.separable_model_ss(k_t, k_u, xt, xp, Ts);

%% Generate data
t = 0:Ts:T;
Ntime = length(t);
ut = [];
up = [];
for iTrain = 1:Nt
    ut = [ut, [xt(:, iTrain)*ones(1, Ntime); t]];
end
for iTest = 1:Np
    up = [up, [xp(:, iTest)*ones(1, Ntime); t]];
end

% Draw samples from the GP
f = mgp.sample(ut, k_f);
y = f + sqrt(R)*randn(1, Nt*Ntime);

%% Batch prediction
tic;
[fp_batch, Cp_batch] = mgp.predict(up, ut, y, R, [], k_f);
toc;

%% State-space prediction
yt = reshape(y, [Ntime, Nt]).';
dx = size(F, 1);

% Filtering
mps = zeros(dx, Ntime);
Pps = zeros(dx, dx, Ntime);
Cps = zeros(dx, dx, Ntime);
mfs = zeros(dx, Ntime);
Pfs = zeros(dx, dx, Ntime);

tic;
m = m0;
P = P0;
for n = 1:Ntime
    % Prediction
    mp = F*m;
    Pp = F*P*F' + Q;
    Cp = F*P;
    
    % Measurement update
    S = C*Pp*C' + R*eye(Nt);
    K = (Pp*C')/S;
    m = mp + K*(yt(:, n) - C*mp);
    P = Pp - K*S*K';
    
    % Store
    mps(:, n) = mp;
    Pps(:, :, n) = Pp;
    Cps(:, :, n) = Cp;
    mfs(:, n) = m;
    Pfs(:, :, n) = P;
end

% Smoothing
ms(:, Ntime) = m;
Ps(:, :, Ntime) = P;
for n = Ntime-1:-1:1
    L = Cps(:, :, n+1)'/Pps(:, :, n+1);
    ms(:, n) = mfs(:, n) + L*(ms(:, n+1) - mps(:, n+1));
    Ps(:, :, n) = Pfs(:, :, n) + L*(Ps(:, :, n+1) - Pps(:, :, n+1))*L';
end
toc;

%% Illustration
% Get the prediction from the smoothed state vector
fp_ks = ms(dxtime*Nt+1:dxtime:dxtime*(Nt+Np), :);
fp_batch = reshape(fp_batch, [Ntime, Np]).';

for i = 1:Np
    figure(i); clf();
    plot(t, fp_batch(i, :)); hold on;
    plot(t, fp_ks(i, :));
    legend('Batch', 'State space');
end
