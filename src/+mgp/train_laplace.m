function [lpy, f, C] = train_laplace(x, y, k, py, par)
% Laplace GP posterior approximation for nonlinear/non-Gaussian likelihoods
%
% USAGE
%   [lpy, f, C] = GP_TRAIN_LAPLACE(y, x, k, py)
%   [lpy, f, C] = GP_TRAIN_LAPLACE(y, x, k, py, par)
%
% DESCRIPTION
%   Posterior approximation of latent Gaussian process models of the form
%
%       f ~ GP(0, k(x, x'))
%       y ~ p(y | f)
%
%   using the Laplace approximation.
%
%   This is an implementation of Algorithm 3.1 in [1] (but generalized for
%   arbitrary likelihoods).
%
% PARAMETERS
%   x   Inputs to the function.
%   y   Observations.
%   k   Covariance function (function handle @(x1, x2)) or precomputed
%       covariance matrix. In the latter case, the dimensions of x, y, and
%       k must match.
%   py  Likelihood, must be a structure with the following fields:
%
%           fast            Boolean flag that indicates whether the
%                           likelihood can be evaluated for all f at once
%                           or not.
%           logpdf(y, f)    Evaluation of the log-pdf.
%           dlogpdf(y, f)   Gradient of the log-likelihood.
%           d2logpdf(y, f)  Hessian of the log-likelihood.
%
%   par Additional parameters:
%
%           J       Maximum number of Newton iterations (default: 10).
%           epsilon Tolerance for change in objective function or change in
%                   f (default: 1e-3).
%
% RETURNS
%   lpy The approximate marginal log-likelihood.
%   f   Posterior mean.
%   C   Posterior covariance.
%
% REFERENCES
%   [1] C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for 
%       Machine Learning. The MIT Press, 2006.
%
% AUTHORS
%   2018-05-14 -- Roland Hostettler <roland.hostettler@aalto.fi>

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
        par = struct();
    end
    def = struct(...
        'J', 10, ...        % Maximum no. of iterations
        'epsilon', 1e-3 ... %
    );
    par = parchk(par, def);
    
    % Preallocate
    y = y(:);
    N = size(y, 1);
    
    %% Initialize
    % Prior covaraince matrix
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
    f = zeros(N, 1);                % Prior mean
    fobj = sum(py.logpdf(y, f));    % Objective function at the prior mean
    
    %% Newton iterations
    done = false;
    j = 1;
    while ~done        
        % Calculate gradient and Hessian
        g = py.dlogpdf(y, f);
        H = py.d2logpdf(y, f);
        
        % Newton step
        W = -diag(H);
        B = eye(N) + sqrt(W)*Kxx*sqrt(W);
        L = chol(B).';
        b = W*f + g;
        v = L\(sqrt(W)*Kxx);
        a = b - sqrt(W)*(L'\(v*b));
        
        % Update mean & covariance, objective function value, and marginal
        % log-likelihood
        fp = Kxx*a;
        Cp = Kxx - v'*v;
        fobjp = -1/2*a'*fp + sum(py.logpdf(y, fp));
        lpyp = fobjp - trace(log(L));        
        
        % Convergence criterion
        done = ( ...
            j >= par.J ...
            || abs((fobjp-fobj)/fobj) < par.epsilon ...
            || norm(fp-f)/norm(f) < par.epsilon ...
        );
        
        % Store for next iteration
        f = fp;
        C = Cp;
        lpy = lpyp;
        fobj = fobjp;
        j = j+1;
    end
    f = f.';                % Put into the same format as the input (1xN)
end
