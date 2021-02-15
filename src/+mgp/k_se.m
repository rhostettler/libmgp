function Kxx = k_se(x1, x2, ell, sigma2)
% # Squared exponential covariance function
% ## Usage
% * `Kxx = k_se(x1, x2)`
% * `Kxx = k_se(x1, x2, ell, sigma2)`
%
% ## Description
% Squared exponential covariance function given by
%
%   k(x1, x2) = sigma2*exp(-1/2*(|x1-x2|/elll)^2).
% 
% ## Input
% * `x1`, `x2`:  Input points. If x1 and x2 have more than one column, each
%   column is treated as a pair. If one of the inputs has more than one 
%   column while the other one only has one, the latter is expanded to 
%   match the first one.
% * `ell`: Length scale or matrix of length scales (default: 1).
% * `sigma2`: Variance (optional, default: 1).
%
% ## Output
% * `Kxx`: Covaraince between x1 and x2.
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

% TODO:
% * Allow for diagonal length scale matrix.
% * Add input expansion

    %% Defaults
    narginchk(2, 4);
    if nargin < 3 || isempty(ell)
        ell = 1;
    end
    if nargin < 4 || isempty(sigma2)
        sigma2 = 1;
    end
    
if 0
    [x1, x2] = expand_inputs(x1, x2);
end
    
    %% Calculate Covariance
    r = sqrt(sum((x1-x2).^2, 1));
    Kxx = sigma2*exp(-1/2*(r/ell).^2);
end
