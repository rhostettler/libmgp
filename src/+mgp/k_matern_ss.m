function [A, B, C, Sw, Pinf] = k_matern_ss(ell, sigma2, nu)
% State-space representation of the Matern covariance function
%
% SYNOPSIS
%   [A, B, C, Sw, Pinf] = gpk_matern_ss(l, sigma2, nu)
%
% DESCRIPTION
%   Calculates the linear state-space representation of a temporal Gaussian
%   process with Mat?rn covariance kernel. Let
%
%       f ~ GP(0, k(t, t')),
%
%   with the covariance kernel k(t, t') given by
%                                       _                   _            _                   _                                              
%                           2^(1-nu)  | (2*nu)^(1/2)*|t-t'| |^nu       | (2*nu)^(1/2)*|t-t'| |
%       k(t, t') = sigma2 * --------- | ------------------- |    * K_nu| ------------------- |.
%                           Gamma(nu) |_        l          _|          |_        l          _|
%
%   This can be written as a state-space system of the form
%
%       dx = A*x*dt + B*dw(t),
%       f = C*x,
%
%   where Sw is the spectral density of w(t). Furthermore, Pinf is the
%   stationary covariance and solves the continuous Ricatti equation
%
%       A'*Pinf + Pinv*A + B*Sw*B' = 0.
%
%   The conversion is exact for positive integer values of p = nu - 0.5, 
%   see [1]. For non-integer values of p, the approximation introduced in 
%   [2] is used.
%
% PARAMETERS
%   ell, sigma2, nu
%       Parameters of the covariance kernel.
%
% RETURNS
%   A, B, C, Sw, Pinf
%       Parameters of the linear state-space equivalent as described above.
%
% SEE ALSO
%   gpk_se_ss
%
% REFERENCES
%   [1] J. Hartikainen and S. S?rkk?, "Kalman filtering and smoothing 
%       solutions to temporal Gaussian process regression models," in IEEE 
%       International Workshop on Machine Learning for Signal Processing 
%       (MLSP), pp. 379-384, August 2010.
%
%   [2] T. Karvonen and S. S?kk?, "Approximate state-space Gaussian 
%       processes via spectral transformation,? in 26th IEEE International 
%       Workshop on Machine Learning for Signal Processing (MLSP), 
%       September 2016.
%
% AUTHOR
%   2017-12-21 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Implement approximation for non-integer nu's

    %% Defaults
    narginchk(0, 3);
    if nargin < 1 || isempty(ell)
        ell = 1;
    end
    if nargin < 2 || isempty(sigma2)
        sigma2 = 1;
    end
    if nargin < 3 || isempty(nu)
        nu = 1.5;
    end
    
    %% Conversion
    p = nu-0.5;
    if sign(p) < 0 || round(p) ~= p
        [A, B, C, Sw] = convert_approximate(ell, sigma2, nu);
    else
        [A, B, C, Sw] = convert_exact(ell, sigma2, nu);
    end
    Pinf = care(A', zeros(p+1), B*Sw*B');
end

%% Exact Conversion
% According to [1]
function [A, B, C, Sw] = convert_exact(ell, sigma2, nu)
    p = nu - 0.5;
    lambda = sqrt(2*nu)/ell;
    Sw = 2*sigma2*sqrt(pi)*lambda^(2*p+1)*gamma(p+1)/gamma(p+1/2);
    
    % Binomial coefficients
    n = p+1;
    k = 1:n;
    c = [1 cumprod((n-k+1)./k)].*lambda.^(n:-1:0);
    A = [
            zeros(p, 1), eye(p);
                      -c(1:p+1);
    ];
    B = [zeros(1, p), 1].';
    C = [1, zeros(1, p)];    
end

%% Approximation
% According to [2]
% TODO: Needs to be implemented
function [A, B, C, Sw] = convert_approximate(ell, sigma2, nu)
    error('Approximation for non-integer p not implemented yet.');
    A = [];
    B = [];
    C = [];
    Sw = [];
end
