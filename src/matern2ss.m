function [A, B, C, Sw, Pinf] = matern2ss(l_gp, var_gp, nu_gp)
% Conversion of the Mat?rn kernel to a state-space GP representation
%
% SYNOPSIS
%   [A, B, C, Sw, Pinf] = matern2ss(l, var, nu)
%
% DESCRIPTION
%   Calculates the linear state-space representation of a temporal Gaussian
%   process with Mat?rn covariance kernel. Let
%
%       f ~ GP(0, k(t, t')),
%
%   with the covariance kernel k(t, t') given by
%                                       _                   _            _                   _                                              
%                            2^(1-nu)  | (2*nu)^(1/2)*|t-t'| |^nu       | (2*nu)^(1/2)*|t-t'| |
%       k(t, t') = sigma^2 * --------- | ------------------- |    * K_nu| ------------------- |.
%                            Gamma(nu) |_        l          _|          |_        l          _|
%
%   This can be written as a state-space system of the form
%
%       dx = A*x*dt + B*w(t),
%       f = C*x,
%
%   where Sw is the spectral density of w(t). Furthermore, Pinf is the
%   stationary covariance and solves the continuous Ricatti equation
%
%       A'*Pinv + Pinv*A + B*Sw*B' = 0.
%
%   The conversion is exact for positive integer values of p = nu - 0.5, 
%   see [1]. For non-integer values of p, the approximation introduced in 
%   [2] is used.
%
% PARAMETERS
%   l, var, nu
%       Parameters of the covariance kernel.
%
% RETURNS
%   A, B, C, Sw, Pinf
%       Parameters of the linear state-space equivalent as described above.
%
% SEE ALSO
%   se2ss
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
% VERSION
%   2017-04-18
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Implement approximation for non-integer p's (nu?)

    %% Defaults
    if nargin < 1 || isempty(l_gp)
        l_gp = 1;
    end
    if nargin < 2 || isempty(var_gp)
        var_gp = 1;
    end
    if nargin < 3 || isempty(nu_gp)
        nu_gp = 1.5;
    end
    
    %% Conversion
    p = nu_gp-0.5;
    if sign(p) ~= 1 || round(p) ~= p
%        error('p must be a non-negative integer (%.2d)', p)
        [A, B, C, Sw] = convert_approximate(l_gp, var_gp, nu_gp);
    else
        [A, B, C, Sw] = convert_exact(l_gp, var_gp, nu_gp);
    end
    Pinf = care(A', zeros(p+1), B*Sw*B');
end

%% Exact Conversion
% According to [1]
function [A, B, C, Sw] = convert_exact(l_gp, var_gp, nu_gp)
    %nu = p + 0.5;
    p = nu_gp - 0.5;
    lambda = sqrt(2*nu_gp)/l_gp;
    Sw = 2*var_gp*sqrt(pi)*lambda^(2*p+1)*gamma(p+1)/gamma(p+1/2);
    
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
function [A, B, C, Sw] = convert_approximate(l, sigma, nu)
    error('Approximation for non-integer p not implemented yet.');
    A = [];
    B = [];
    C = [];
    Sw = [];
end
