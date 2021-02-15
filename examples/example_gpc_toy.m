% Gaussian process classification toy example
%
% Re-implementation of the toy example (Section 3.7.2) in Rasmussen and 
% Williams (2006).
% 
% 2017-present -- Roland Hostettler

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

% TODO:
% * There are some bugs in the prediction functions that need to be sorted
%   out rigorously.

% Housekeeping
clear variables;
addpath ../src

%% Model
% Hyperparameters
ell = 2.6;
sigma2 = 7^2;

% Covariance function
k = @(x1, x2) mgp.k_se(x1, x2, ell, sigma2);

% Grid for prediction
xp = -9:0.1:4;

%% Data
x = [
    -6 + 0.8*randn(20, 1);
     0 + 0.8*randn(30, 1);
     2 + 0.8*randn(10, 1);
].';
y = [
       ones(20, 1);
    -1*ones(30, 1);
       ones(10, 1);
].';

%% Training
[f_laplace, Sigma_laplace, lpy_laplace] = mgp.train_classifier_laplace(x, y, k);
[nu_ep, tau_ep, f_ep, Sigma_ep, lpy_ep] = mgp.train_classifier_ep(x, y, k);

%% Prediction
[rhop_laplace, fp_laplace, Sigmap_laplace] = mgp.predict_class_laplace(xp, x, f_laplace, k);
[rhop_ep, fp_ep, Sigmap_ep] = mgp.predict_class_ep(xp, x, f_ep, k);

%% Visualization
figure(1); clf();
plot(x, f_laplace, '.'); hold on;
plot(x, f_ep, '.');
legend('Laplace', 'EP');
title('Posterior Mean');

figure(2); clf();
plot(xp, fp_laplace); hold on;
plot(xp, fp_ep);
legend('Laplace', 'EP');
title('Predicted Latent Function Values');

figure(3); clf();
plot(xp, rhop_laplace); hold on;
plot(xp, rhop_ep);
legend('Laplace', 'EP');
plot(x(y == 1), y(y == 1), 'xg');
plot(x(y == -1), 0*y(y == -1), 'og');
title('Predicted Class Probabilities');
xlim([-9, 4]);
