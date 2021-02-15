function [f, Sigma, lpy] = train_classifier_laplace(x, y, k, epsilon, J)
% # Binary Gaussian process classifier training using Laplace approximation
% ## Usage
% * `[f, Sigma, lpy] = train_classifier_laplace(x, y)`
% * `[f, Sigma, lpy] = train_classifier_laplace(x, y, k, epsilon, J)`
%
% ## Description
% Training of binary Gaussian process classifiers using the Laplace
% approximation.
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
% * `f`, `Sigma`: Latent GP function values and variance.
% * `lpy`: Marginal log-likelihood.
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
    y = y(:);
    N = size(y, 1);
    f = zeros(N, 1);
    lpy = -Inf;

    done = false;
    j = 0;
    while ~done
        %% Training iteration
        lpy_old = lpy;
        j = j+1;
        
        Nyf = normpdf(y.*f);
        Cyf = normcdf(y.*f);
        H = -Nyf.^2./Cyf.^2 - y.*f.*Nyf./Cyf;
        W = -diag(H);
        B = eye(N) + sqrt(W)*K*sqrt(W);
        L = chol(B, 'lower');

        g = y.*Nyf./Cyf;
        b = W*f + g;
        v = L\(sqrt(W)*K);
        a = b - sqrt(W)*(L'\(v*b));
        f = K*a;
        Sigma = K - v'*v;

        % Calculate approximate marginal loglikelihood
        lpy = -1/2*a'*f + sum(log(normcdf(y.*f))) - trace(log(L));

        %% Convergence
        done = (abs(lpy_old-lpy)/abs(lpy) < epsilon) || (j >= J);
    end   
    f = f.';
end
