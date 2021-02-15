function [nu_tilde, tau_tilde, f, Sigma, lpy] = train_classifier_ep(x, y, k, epsilon, J)
% # Binary Gaussian process classifier training using expectation propagation
% ## Usage
% * `[nu, tau] = train_classifier_ep(y, k)`
% * `[nu, tau, lpy, mu, Sigma] = train_classifier_ep(y, k, epsilon, J)`
%
% ## Description
% Training of binary Gaussian process classifiers using the expectation
% propagation.
%
% ## Input
% * `x`: dx-times-N matrix of training input values.
% * `y`: 1-times-N vector of training class labels (-1 / +1).
% * `k`: Covariance function (default: `k_se`).
% * `epsilon`: Convergence tolerance (in relative marginal loglikelihood 
%   change; default: 1e-3).
% * `J`: Maximum number of iterations before stopping (default: 10).
%
% ## Output
% * `nu`, `tau`: Trained parameters.
% * `lpy`:  Marginal loglikelihood.
% * `mu`, `Sigma`: Latent GP posterior mean and covariance.
%
% ## Authors
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

    %% Defaults
    narginchk(2, 5);
    if nargin < 3 || isempty(k)
        k = @mgp.k_se;
    end
    if nargin < 4 || isempty(epsilon)
        epsilon = 1e-3;
    end
    if nargin < 5 || isempty(J)
        J = 10;
    end

    %% Initialize
    % Calculate covaraince matrix
    K = mgp.calculate_covariance(x, k);

    % Preallocate
    N = size(y, 2);
    nu_tilde = zeros(N, 1);
    nu_minus = zeros(N, 1);
    tau_tilde = zeros(N, 1);
    tau_minus = zeros(N, 1);
    mu_minus = zeros(N, 1);
    sigma2_minus = zeros(N, 1);
    Z_hat = zeros(N, 1);
    mu = zeros(N, 1);
    Sigma = K;

    %% Train
    done = false;
    lpy = -Inf;
    j = 0;
    while ~done
        lpy_old = lpy;
        j = j + 1;
        for i = 1:N
            % Compute approximate cavity parameters
            sigma2_i = Sigma(i, i);
            mu_i = mu(i);
            tau_minus(i) = 1/sigma2_i - tau_tilde(i);
            nu_minus(i) = 1/sigma2_i*mu_i - nu_tilde(i);
            
            % Compute the marginal moments
            sigma2_minus(i) = 1/tau_minus(i);
            mu_minus(i) = nu_minus(i)/tau_minus(i);
%             sigma_minus(i) = 1/(1/sigma2_i + tau_tilde(i));
%             mu_minus(i) = sigma_minus(i)*(1/sigma2_i*mu_i - nu_tilde(i));
            z_i = y(i)*mu_minus(i)/sqrt(1+sigma2_minus(i));
            Nzi = normpdf(z_i);
            Czi = normcdf(z_i);
            
            Z_hat(i) = Czi;
            mu_i_hat = mu_minus(i) + y(i)*sigma2_minus(i)*Nzi/(Czi*sqrt(1+sigma2_minus(i)));
            sigma_i_hat = sigma2_minus(i) - sigma2_minus(i)^2*Nzi/((1+sigma2_minus(i))*Czi)*(z_i + Nzi/Czi);
            
            % Update site parameters
            delta_tau_tilde = 1/sigma_i_hat - tau_minus(i) - tau_tilde(i);
            tau_tilde(i) = tau_tilde(i) + delta_tau_tilde;
            nu_tilde(i) = 1/sigma_i_hat*mu_i_hat - nu_minus(i);
            
            % Update 
            Sigma = Sigma - (1/delta_tau_tilde + Sigma(i, i))\(Sigma(:, i)*Sigma(:, i)');
            mu = Sigma*nu_tilde;
        end
        
        % Recompute the approximate posterior parameters
        S_tilde = diag(tau_tilde);
        B = eye(N) + sqrt(S_tilde)*K*sqrt(S_tilde);
        L = chol(B, 'lower');
        V = L\sqrt(S_tilde)*K;
        Sigma = K - (V'*V);
        mu = Sigma*nu_tilde;
        
        % Compute the marginal log-likelihood
        T = diag(1./sigma2_minus);
        lpy = ( ...
            + sum(log(Z_hat)) ...                       % Third term
            + 1/2*sum(log(1+tau_tilde./tau_minus)) ...    % Fourth and first terms
            - trace(log(L)) ...
            + 1/2*(nu_tilde'*(K-K*sqrt(S_tilde)/B*sqrt(S_tilde)*K - eye(N)/(T+S_tilde))*nu_tilde) ...
            + 1/2*(mu_minus'*T/(S_tilde + T)*(S_tilde*mu_minus - 2*nu_tilde)) ...
        );
    
        %% Convergence
        done = (abs(lpy_old-lpy)/abs(lpy) < epsilon) || (j >= J);       
%         done = (j >= par.J) || (norm(mu-mu_old)/N <= par.epsilon);
%        done = (j >= par.J) || (abs(lpy-lpy_old) <= par.epsilon);
    end
    
    %% Output
    nu_tilde = nu_tilde.';
    tau_tilde = tau_tilde.';
    f = mu.';
end
