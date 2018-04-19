function Kxx = gpk_matern(x1, x2, ell, sigma2, nu)
% Matern covariance function
% 
% USAGE
%   k = gpk_matern(x1, x2)
%   k = gpk_matern(x1, x2, ell, sigma2, nu)
%
% DESCRIPTION
%   Matern covariance function defined as
%                                        _                    _            _                    _ 
%                            2^(1-nu)  | (2*nu)^(1/2)*|x1-x2| |^nu       | (2*nu)^(1/2)*|x1-x2| |
%       k(x1, x2) = sigma2 * --------- | -------------------- |    * K_nu| -------------------  |
%                            Gamma(nu) |_        l           _|          |_        l           _|
%
%   where Gamma(.) is the gamma function and K_nu the Bessel function of
%   the second kind of order nu.
%
% PARAMETERS
%   x1, x2  Input points. If x1 and x2 have more than one column, each 
%           column is treated as a pair. If one of the inputs has more than
%           one column while the other one only has one, the latter is 
%           expanded to match the first one.
%   ell     Length scale (optional, default: 1)
%   sigma2  Variance (optional, default: 1)
%   nu      Order (optional, default: 1.5)
%
% RETURNS
%   Kxx     Covaraince between x1 and x2
% 
% AUTHOR
%   2017-12-21 -- Roland Hostettler <roland.hostettler@aalto.fi>

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
%   * Fix bug for nu > 5/2
%   * Implement nu = 1/2 (Ornstein-Uhlenbeck case)
%   * Allow for diagonal ell matrices

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
    r = sqrt(sum((x1-x2).^2, 1));
    switch nu
        case 1.5
            Kxx = sigma2*(1 + sqrt(3)*r/ell)*exp(-sqrt(3)*r/ell);
            
        case 2.5
            Kxx = sigma2*(1 + sqrt(5)*r/ell + 5*r.^2/(3*ell^2))*exp(-sqrt(5)*r/ell);
            
        otherwise
            Kxx = sigma2*2^(1-nu)/gamma(nu)*(sqrt(2*nu)/ell*r)^nu*besselk(nu, sqrt(2*nu)/ell*r);
            % TODO: This is not quite true.
            Kxx(r == 0) = sigma2;
    end
end
