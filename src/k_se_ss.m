function [A, B, C, Sw, Pinf] = k_se_ss(l, sigma, N)
% Taylor approximation of the SE covariance spectral density
%
% SYNOPSIS
%   [A, B, C, Sw, Pinf] = se2ss(l, sigma, N)
%
% DESCRIPTION
%   Calculates the linear state-space approximation of a temporal Gaussian 
%   process (GP) with squared exponential (SE) covariance function. In 
%   particular, if
%
%       f(t) ~ GP(m(t), k(t, t')),
%
%   where
%
%       k(t, t') = sigma^2*exp(-(t-t')^2/(2*l^2))
%
%   is the SE covariance kernel, then it calculates the matrices A, B, C
%   and spectral density Sw such that
%
%       dx = A*x*dt + B*w(t)
%       f = C*x
%
%   where Sw is the spectral density of w(t). Furthermore, Pinf is the
%   stationary covariance and solves the continuous Ricatti equation
%
%       A'*Pinv + Pinv*A + B*Sw*B' = 0.
%
%   The approximation is based on the Taylor series expansion of the SE
%   spectral density, see, for example [1].
%
% PARAMETERS
%   l   GP length scale (optional, default: 1).
%
%   sigma
%       GP covariance magnitude (standard deviation; optional, default: 1).
%
%   N   Approximation order (must be even; optional, default: 6).
%
% RETURNS
%   A, B, C, Sw, Pinf
%       Parameters of the linear state-space equivalent as described above.
%
% SEE ALSO
%   matern2ss
%
% REFERENCES
%   [1] J. Hartikainen and S. S?rkk?, "Kalman filtering and smoothing 
%       solutions to temporal Gaussian process regression models," in IEEE 
%       International Workshop on Machine Learning for Signal Processing 
%       (MLSP), pp. 379?384, August 2010.
%
% VERSION
%   2016-04-10
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(0, 3);
    if nargin == 0 || isempty(l)
        l = 1;
    end
    if nargin < 2 || isempty(sigma)
        sigma = 1;
    end
    if nargin < 3 || isempty(N)
        N = 6;
    end
    if mod(N, 2) ~= 0
        error('The approximation order N must be even (%d).', N);
    end

    %% Approximation
    n = 0:N;
    kappa = 1/(2*l^2);
    Sw = sigma^2*factorial(N)*(4*kappa)^N*sqrt(pi/kappa);
    
    c = zeros(1, 2*N+1);
    c(2*n+1) = factorial(N)*(-1).^n.*(4*kappa).^(N-n)./factorial(n);
    r = roots(fliplr(c));
    rminus = r(real(r) < 0);
    pminus = fliplr(poly(rminus));

    A = [
        zeros(N-1, 1), eye(N-1);
                   -pminus(1:N);
    ];
    B = [zeros(1, N-1), 1].';
    C = [1, zeros(1, N-1)];
    Pinf = care(A', zeros(N), B*Sw*B');
end
