% GP regression using state-space formulation and SE kernel
%
% Implementation of Gaussian Process (GP) regression based on the state-
% space formulation of temporal GPs (time-series), see [1], [2].
%
% [1] J. Hartikainen and S. Sarkka, "Kalman filtering and smoothing 
%     solutions to temporal Gaussian process regression models," in IEEE 
%     International Workshop on Machine Learning for Signal Processing 
%     (MLSP), pp. 379?384, August 2010.
%
% [2] A. Carron, M. Todescato, R. Carli, L. Schenato, and G. Pillonetto, 
%     "Machine learning meets Kalman filtering," in 55th IEEE Conference on
%     Decision and Control (CDC), pp. 4594? 4599, December 2016.
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
sigma2 = 0.5;   % Variance
ell = 1;        % Length scale
J = 6;          % State-space approximation order 
T = 1000;       % Simulation time
Ts = 0.5;       % Sampling time
R = 0.01;       % Measurement noise

%% Covariance Approximation
% State-space model and discretization thereof
[A, B, C, Sw, Pinf] = mgp.k_se_ss(ell, sigma2, J);
[F, Q] = lti_disc(A, B, Sw, Ts);
Q = (Q+Q')/2;
Lq = chol(Q).';

% Visualize the true spectral density and its approximation
w = (0:0.1:5);
Sww = mgp.k_se_psd(w, ell, sigma2);
s = tf('s');
G = C*(s*eye(size(A))-A)^(-1)*B;
[Sww_hat, ~] = bode((G*G')*Sw, w);

figure(1); clf();
plot(w, Sww); hold on;
plot(w, squeeze(Sww_hat));
legend('True', 'Approximation');
title('Comparison of the Spectral Densities');
xlabel('\omega / s^{-1}'); ylabel('S_{ww}(\omega)');

%% Data Generation
k = @(t1, t2) mgp.k_se(t1, t2, ell, sigma2);
t = (Ts:Ts:T);
fr = mgp.sample(t, k);
N = length(t);
yr = fr + sqrt(R)*randn(1, N);

%% Batch Regression
tic;
Ktt = mgp.calculate_covariance(t, k);
S = Ktt + R*eye(N);
fhat_batch = (Ktt/S*yr.').';
toc;

%% KF Regression
% Filtering
tic;
fhat_f = zeros(1, N);
m = zeros(J, 1);
mf = zeros(J, N);
P = Pinf;
Pf = zeros(J, J, N);
for n = 1:N
    % Prediction
    mp = F*m;
    Pp = F*P*F' + Q;
    
    % Update
    S = C*Pp*C' + R;
    K = Pp*C'/S;
    m = mp + K*(yr(n) - C*mp);
    P = Pp - K*S*K';
    
    mf(:, n) = m;
    Pf(:, :, n) = P;
    
    % Store
    fhat_f(n) = C*m;
end

% Smoothing
fhat_s = zeros(1, N);
ms = zeros(J, N);
Ps = zeros(J, J, N);
ms(:, N) = mf(:, N);
Ps(:, :, N) = Ps(:, :, N);
fhat_s(:, N) = fhat_f(:, N);
for n = N-1:-1:1
    Pp = F*Pf(:, :, n)*F' + Q;
    L = Pf(:, :, n)*F'/ Pp;
    ms(:, n)   = mf(:, n) + L*(ms(:, n+1) - F*mf(:, n));
    Ps(:, :, n) = Pf(:, :, n) + L*(Ps(:, :, n+1) - Pp)*L';
    
    fhat_s(:, n) = C*ms(:, n);
end
toc;

%% Results
fprintf('Batch RMSE: %.3f\n', rms(fr-fhat_batch));
fprintf('KF RMSE: %.3f\n', rms(fr-fhat_f));
fprintf('RTS RMSE: %.3f\n', rms(fr-fhat_s));

figure(2); clf();
plot(t, fr); hold on;
plot(t, fhat_batch);
plot(t, fhat_f);
plot(t, fhat_s);
legend('True', 'Batch', 'KF', 'RTS');
title('Estimated GP Values');
