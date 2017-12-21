function [nu_tilde, tau_tilde, lpy, mu, Sigma] = gpc_train_ep(x, y, k, par)
% Binary Gaussian process classifier training using Expectation Propagation
%
% SYNOPSIS
%   [nu, tau] = gpc_train_ep(y, k)
%   [nu, tau, lpy, mu, Sigma] = gpc_train_ep(y, k, par)
%
% DESCRIPTION
%   Algorithm 3.5 in [1]
%
% PARAMETERS
%   x   Training inputs
%   
%   y   Training class labels
%
%   k   Covariance function @(x1, x2)
%
%   par Algorithm parameters:
%
%           J       Maximum number of iterations
%           epsilon Convergence tolerance
%
% RETURNS
%   nu, tau Trained parameters
%
%   lpy     Marginal log-likelihood
%
%   mu      Posterior mean
%
%   Sigma   Posterior covariance
%
% REFERENCES
%   
%
% VERSION
%   2017-10-05
% 
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Documentation

    %% Defaults
    narginchk(3, 4);
    if nargin < 4
        par = [];
    end
    def = struct(...
        'J', 10, ...        % Maximum no. of iterations
        'epsilon', 1e-3 ... % Convergence tolerance (in norm of posterior mean change)
    );
    par = parchk(par, def);

    %% Initialize
    % Calculate covaraince matrix
    K = gp_calculate_covariance(x, k);

    % Preallocate
    N = size(y, 2);
    nu_tilde = zeros(N, 1);
    nu_minus = zeros(N, 1);
    tau_tilde = zeros(N, 1);
    tau_minus = zeros(N, 1);
    mu_minus = zeros(N, 1);
    sigma_minus = zeros(N, 1);
    Z_hat = zeros(N, 1);
    mu = zeros(N, 1);
    Sigma = K;

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
            sigma_i = Sigma(i, i);
            mu_i = mu(i);
            tau_minus(i) = 1/sigma_i - tau_tilde(i);
            nu_minus(i) = 1/sigma_i*mu_i - nu_tilde(i);
            
            % Compute the marginal moments
            sigma_minus(i) = 1/tau_minus(i); %1/(1/sigma_i + tau_tilde(i));
            mu_minus(i) = nu_minus(i)/tau_minus(i); % sigma_minus(i)*(1/sigma_i*mu_i - nu_tilde(i));
            z_i = y(i)*mu_minus(i)/sqrt(1+sigma_minus(i));
            Nzi = normpdf(z_i);
            Czi = normcdf(z_i);
            
            Z_hat(i) = Czi;
            mu_i_hat = mu_minus(i) + y(i)*sigma_minus(i)*Nzi/(Czi*sqrt(1+sigma_minus(i)));
            sigma_i_hat = sigma_minus(i) - sigma_minus(i)^2*Nzi/((1+sigma_minus(i))*Czi)*(z_i + Nzi/Czi);
            
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
        T = diag(1./sigma_minus);
        lpy = ( ...
            + sum(log(Z_hat)) ...                       % Third term
            + 1/2*sum(log(1+tau_tilde./tau_minus)) ...    % Fourth and first terms
            - trace(log(L)) ...
            + 1/2*(nu_tilde'*(K-K*sqrt(S_tilde)/B*sqrt(S_tilde)*K - eye(N)/(T+S_tilde))*nu_tilde) ...
            + 1/2*(mu_minus'*T/(S_tilde + T)*(S_tilde*mu_minus - 2*nu_tilde)) ...
        );
        
        done = (j >= par.J) || (norm(mu-mu_old)/N <= par.epsilon);
%        done = (j >= par.J) || (abs(lpy-lpy_old) <= par.epsilon);
    end
end
