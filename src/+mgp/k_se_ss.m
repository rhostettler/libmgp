function [A, B, C, Sw, Pinf] = k_se_ss(ell, sigma2, J)
% # State-space representation of the SE covariance function (Taylor app.)
% ## Usage
% * `[A, B, C, Sw, Pinf] = k_se_ss()`
% * `[A, B, C, Sw, Pinf] = k_se_ss(ell, sigma2, J)`
%
% ## Description
% Calculates the linear state-space approximation of a temporal Gaussian 
% process (GP) with squared exponential (SE) covariance function. In 
% particular, if
%
%   f(t) ~ GP(m(t), k(t, t')),
%
% where
%
%   k(t, t') = sigma2*exp(-(t-t')^2/(2*ell^2))
%
% is the SE covariance function, then it calculates the matrices A, B, C
% and spectral density Sw such that
%
%   dx = A*x*dt + B*dw(t)
%   f = C*x
%
% where Sw is the spectral density of w(t). Furthermore, Pinf is the
% stationary covariance and solves the continuous Lyapunov equation
%
%   A'*Pinf + Pinv*A + B*Sw*B' = 0.
%
% The approximation is based on the Taylor series expansion of the SE
% spectral density, see, for example [1].
%
% ## Input
% * `ell`: Length scale (default: 1).
% * `sigma2`: GP covariance magnitude (standard deviation; default: 1).
% * `J`: Approximation order (must be even; default: 6).
%
% ## Output
% * `A`, `B`, `C`, `Sw`, `Pinf`: Linear state-space system equivalent as 
%   described above.
%
% ## References
% 1. J. Hartikainen and S. Sarkka, "Kalman filtering and smoothing 
%    solutions to temporal Gaussian process regression models," in IEEE 
%    International Workshop on Machine Learning for Signal Processing 
%    (MLSP), pp. 379?384, August 2010.
%
% ## Author
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
    narginchk(0, 3);
    if nargin == 0 || isempty(ell)
        ell = 1;
    end
    if nargin < 2 || isempty(sigma2)
        sigma2 = 1;
    end
    if nargin < 3 || isempty(J)
        J = 6;
    end
    if mod(J, 2) ~= 0
        error('The approximation order J must be even (%d).', J);
    end

    %% Approximation
    j = 0:J;
    kappa = 1/(2*ell^2);
    Sw = sigma2*factorial(J)*(4*kappa)^J*sqrt(pi/kappa);
    
    c = zeros(1, 2*J+1);
    c(2*j+1) = factorial(J)*(-1).^j.*(4*kappa).^(J-j)./factorial(j);
    r = roots(fliplr(c));
    rminus = r(real(r) < 0);
    pminus = fliplr(poly(rminus));

    A = [
        zeros(J-1, 1), eye(J-1);
                   -pminus(1:J);
    ];
    B = [zeros(1, J-1), 1].';
    C = [1, zeros(1, J-1)];
    Pinf = care(A', zeros(J), B*Sw*B');
end
