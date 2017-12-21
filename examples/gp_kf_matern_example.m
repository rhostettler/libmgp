% GP regression using state-space formulation of Matérn kernel example
%
% Implementation of Gaussian Process (GP) regression based on the state-
% space formulation of spatio-temporal GPs, see [1], [2].
%
% [1] J. Hartikainen and S. Särkkä, "Kalman filtering and smoothing 
%     solutions to temporal Gaussian process regression models," in IEEE 
%     International Workshop on Machine Learning for Signal Processing 
%     (MLSP), pp. 379–384, August 2010.
%
% [2] A. Carron, M. Todescato, R. Carli, L. Schenato, and G. Pillonetto, 
%     "Machine learning meets Kalman filtering," in 55th IEEE Conference on
%     Decision and Control (CDC), pp. 4594–4599, December 2016.
% 
% 2017-10-20 -- Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;
addpath lib;
rng(23);

%% Parameters
l_x = 2;            % Spatial length scale
var_t = 0.5;        % Variance
l_t = 1;            % Temporal length scale
R = 0.01;           % Measurement noise variance

T = 100;              % Simulation time
Ts = 0.25;          % Sampling time

NTrain = 4;         % No. of spatial training points
NTest = 4;          % No. of spatial test points

% Spatial training and test points, randomly generated
xt = randn(2, NTrain);
xp = randn(2, NTest);

%% Batch model
% u = [x; t]
k_f = @(u1, u2) ( ...
    k_matern(u1(1:2, :), u2(1:2, :), l_x, 1, 2.5) ...
    .*k_matern(u1(3, :), u2(3, :), l_t, var_t, 2.5) ...
);

%% State space model
% Convert temporal covaraince matrix to state space system
[A, B, C, Sw, Pinf] = matern2ss(l_t, var_t, 2.5);
[F, Q] = lti_disc(A, B, Sw, Ts);
Q = (Q+Q')/2;
N0 = size(F, 1);

% State transition matrix: Simply a block-diagonal matrix with the
% matrix F on each block (for each xi / xp)
F = kron(eye(NTrain+NTest), F);
    
% Measurement vector: "Block vector" similar to the above with no
% measurements for the prediction points.
C = [kron(eye(NTrain), C), zeros(NTrain, NTest*N0)];

% The covariance between the 'different' systems is given through the 
% covariance between the inducing points (times the covariance of the 
% temporal system)
xx = [xt, xp];
Kxx = gp_calculate_covariance(xx, @(x1, x2) k_matern(x1, x2, l_x, 1, 2.5));
Q = kron(Kxx, Q);
    
% Initial state -- same as for Q
P0 = kron(Kxx, Pinf);

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
m = zeros(Nx, 1);
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
