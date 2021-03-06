function Kxx = k_matern(x1, x2, ell, sigma2, nu)
% # Matern covariance function
% ## Usage
% * `k = k_matern(x1, x2)`
% * `k = k_matern(x1, x2, ell, sigma2, nu)`
%
% ## Description
% Matern covariance function defined as
%                                   _                    _            _                    _ 
%                        2^(1-nu)  | (2*nu)^(1/2)*|x1-x2| |^nu       | (2*nu)^(1/2)*|x1-x2| |
%   k(x1, x2) = sigma2 * --------- | -------------------- |    * K_nu| -------------------  |
%                        Gamma(nu) |_        l           _|          |_        l           _|
%
% where Gamma(.) is the gamma function and K_nu the Bessel function of the
% second kind of order nu.
%
% ## Input
% * `x1`, `x2`: Input points. If `x1` and `x2` have more than one column, 
%   each column is treated as a pair. If one of the inputs has more than 
%   one column while the other one only has one, the latter is expanded to 
%   match the first one.
% * `ell`: Length scale (default: 1).
% * `sigma2`: Variance (default: 1).
% * `nu`: Order (default: 1.5).
%
% ## Output
% * `Kxx`: Covaraince between `x1` and `x2`.
% 
% ## Authors
% * 2017-present -- Roland Hostettler

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
% * Fix bug for nu > 5/2
% * Allow for diagonal ell matrices
% * Add input expansion

    %% Defaults
    narginchk(2, 5);
    if nargin < 3 || isempty(ell)
        ell = 1;
    end
    if nargin < 4 || isempty(sigma2)
        sigma2 = 1;
    end
    if nargin < 5 || isempty(nu)
        nu = 1.5;
    end
if 0
    [x1, x2] = expand_gp_inputs(x1, x2);
end
    
    %% Calculate Covariance
    % If x2 is empty, we assume that x1 is a matrix of euclidean distances
    % (used for fast computations)
    if ~isempty(x2)
        r = sqrt(sum((x1-x2).^2, 1));
    else
        r = x1;
    end
    switch nu
        case 0.5
            Kxx = sigma2*exp(-r/ell);
        
        case 1.5
            Kxx = sigma2*(1 + sqrt(3)*r/ell).*exp(-sqrt(3)*r/ell);
            
        case 2.5
            Kxx = sigma2*(1 + sqrt(5)*r/ell + 5*r.^2/(3*ell^2)).*exp(-sqrt(5)*r/ell);
            
        otherwise
            Kxx = sigma2*2^(1-nu)/gamma(nu)*(sqrt(2*nu)/ell*r).^nu.*besselk(nu, sqrt(2*nu)/ell*r);
            % TODO: This is not quite true.
            Kxx(r == 0) = sigma2;
    end
end
