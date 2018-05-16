function [lpy, mu, Sigma, nu_tilde, tau_tilde] = gp_train_ep(x, y, k, py, par)
% Expectation propagation GP posterior approximation
%
% USAGE
%   [lpy, f, C] = GP_TRAIN_EP(y, k)
%   [lpy, f, C, nu, tau] = GP_TRAIN_EP(y, k, par)
%
% DESCRIPTION
%   Expectation propagation posterior approximation for latent Gaussian 
%   process models with nonlinear/non-Gaussian likelihoods. The
%   implementation is based on Algorithm 3.5 in [1] with the exception that
%   the moments are calculated using Monte Carlo integration.
%
%   There are dedicated training functions for the binary classification
%   problem, see gpc_train_ep.
%
% PARAMETERS
%   x   Training inputs.
%   y   Training class labels.
%   k   Covariance function @(x1, x2).
%   par Algorithm parameters:
%
%           J       Maximum number of iterations (default: 10)
%           Nmc     Number of Monte Carlo samples to use for numerical
%                   integration (default: 1000).
%           epsilon Convergence tolerance for the relative change in
%                   posterior mean (default: 1e-3).
%
% RETURNS
%   lpy     Marginal log-likelihood.
%   f, C    Posterior mean and covariance.
%   nu, tau Trained EP representation parameters (required for
%           predictions).
%
% REFERENCES
%   [1] C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for 
%       Machine Learning. The MIT Press, 2006.
% 
% AUTHORS
%   2018-05-16 -- Roland Hostettler <roland.hostettler@aalto.fi>

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

% TODO:
%   * Implicitly assumes py.fast!

    %% Defaults
    narginchk(4, 5);
    if nargin < 5
        par = [];
    end
    def = struct(...
        'J', 10, ...        % Maximum no. of iterations
        'Nmc', 1000, ...    % No. of MC samples for numerical integration
        'epsilon', 1e-3 ... % Convergence tolerance (in norm of posterior mean change)
    );
    par = parchk(par, def);
    
    %% Initialize
    % Calculate covaraince matrix
    N = size(y, 2);
    if isa(k, 'function_handle')
        Kxx = gp_calculate_covariance(x, k);
    elseif isnumeric(k)
        if length(k) ~= N || length(k) ~= length(x)
            error('Dimension mismatch in precomputed ''k''');
        end
        Kxx = k;
    else
        error('''k'' must either be a covariance function handle or a precomputed covariance matrix.');
    end
    
    % Preallocate

    nu_tilde = zeros(N, 1);
    nu_minus = zeros(N, 1);
    tau_tilde = zeros(N, 1);
    tau_minus = zeros(N, 1);
    mu_minus = zeros(N, 1);
    sigma2_minus = zeros(N, 1);
    Z_hat = zeros(N, 1);
    mu = zeros(N, 1);
    Sigma = Kxx;
    
    %% Train
    done = false;
    lpy = -Inf;
    j = 0;
    while ~done
        lpy_old = lpy;
        mu_old = mu;
        j = j + 1;
        for i = 1:N
            % Compute approximate cavity parameters
            sigma2_i = Sigma(i, i);
            mu_i = mu(i);
            tau_minus(i) = 1/sigma2_i - tau_tilde(i);
            nu_minus(i) = 1/sigma2_i*mu_i - nu_tilde(i);
            
            % Compute the marginal moments using Monte Carlo integration
            % (importance sampling, really)
            sigma2_minus(i) = 1/tau_minus(i);
            mu_minus(i) = nu_minus(i)/tau_minus(i);
            
            fs = mu_minus(i) + sqrt(sigma2_minus(i))*randn(par.Nmc, 1);
            ws = py.logpdf(y(i), fs);
            ws = exp(ws-max(ws));
            Z_hat(i) = sum(ws);
            ws = ws/Z_hat(i);
            mu_i_hat = ws'*fs;
            sigma2_i_hat = ws'*(fs-mu_i_hat).^2;
            
            % Update site parameters
            delta_tau_tilde = max(0, 1/sigma2_i_hat - tau_minus(i) - tau_tilde(i));
            tau_tilde(i) = tau_tilde(i) + delta_tau_tilde;
            nu_tilde(i) = 1/sigma2_i_hat*mu_i_hat - nu_minus(i);
            
            % Update 
            Sigma = Sigma - (1/delta_tau_tilde + Sigma(i, i))\(Sigma(:, i)*Sigma(:, i)');
            mu = Sigma*nu_tilde;
        end
        % Recompute the approximate posterior parameters
        S_tilde = diag(tau_tilde);
        B = eye(N) + sqrt(S_tilde)*Kxx*sqrt(S_tilde);
        B = (B + B')/2;
        L = chol(B, 'lower');
        V = L\sqrt(S_tilde)*Kxx;
        Sigma = Kxx - (V'*V);
        Sigma = (Sigma + Sigma')/2;
        mu = Sigma*nu_tilde;
        
        % Compute the marginal log-likelihood
        T = diag(1./sigma2_minus);
        lpy = ( ...
            + sum(log(Z_hat)) ...                       % Third term
            + 1/2*sum(log(1+tau_tilde./tau_minus)) ...    % Fourth and first terms
            - trace(log(L)) ...
            + 1/2*(nu_tilde'*(Kxx-Kxx*sqrt(S_tilde)/B*sqrt(S_tilde)*Kxx - eye(N)/(T+S_tilde))*nu_tilde) ...
            + 1/2*(mu_minus'*T/(S_tilde + T)*(S_tilde*mu_minus - 2*nu_tilde)) ...
        );
        
        done = ( ...
            j >= par.J ...
            || norm(mu-mu_old)/norm(mu_old) <= par.epsilon ...
            || abs(lpy-lpy_old)/abs(lpy_old) <= par.epsilon ...
        );
    end
end
