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
sigma2 = 0.5;   % Variance
ell = 1;        % Length scale
J = 6;          % State-space approximation order 
T = 1000;       % Simulation time
Ts = 0.5;       % Sampling time
R = 0.01;       % Measurement noise

%% Covariance Approximation
[A, B, C, Sw, Pinf] = gpk_se_ss(ell, sigma2, J);

% Visualize the true spectral density and its approximation
w = (0:0.1:5);
Sww = sigma2*sqrt(2*pi*ell^2)*exp(-w.^2*ell^2/2);
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
k = @(t1, t2) gpk_se(t1, t2, ell, sigma2);
t = (Ts:Ts:T);
Ktt = gp_calculate_covariance(t, k);
N = length(t);
fr = chol(Ktt).'*randn(N, 1);
yr = fr + sqrt(R)*randn(N, 1);

%% Batch Regression
tic;
S = Ktt + R*eye(N);
fhat_batch = Ktt/S*yr;
toc;

%% KF Regression
% Discretization
[F, Q] = lti_disc(A, B, Sw, Ts); 
Q = (Q+Q')/2;
Lq = chol(Q).';

% Filtering
tic;
fhat_f = zeros(N, 1);
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
[ms, Ps] = rts_smooth(mf, Pf, F, Q);
fhat_s = C*ms;
toc;

%% Results
fprintf('Batch RMSE: %.3f\n', rms(fr-fhat_batch));
fprintf('KF RMSE: %.3f\n', rms(fr-fhat_f));
fprintf('RTS RMSE: %.3f\n', rms(fr-fhat_s.'));

figure(2); clf();
plot(t, fr); hold on;
plot(t, fhat_batch);
plot(t, fhat_f);
plot(t, fhat_s);
legend('True', 'Batch', 'KF', 'RTS');
title('Estimated GP Values');
