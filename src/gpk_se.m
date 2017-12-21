function Kxx = gpk_se(x1, x2, ell, sigma2)
% Squared exponential covariance kernel
%
% USAGE
%   Kxx = gpk_se(x1, x2)
%   Kxx = gpk_se(x1, x2, ell, sigma2)
%
% DESCRIPTION
%   Squared exponential covariance function given by
%
%       k(x1, x2) = sigma2*exp(-1/2*(|x1-x2|/elll)^2).
% 
% PARAMETERS
%   x1, x2  Input points. If x1 and x2 have more than one column, each
%           column is treated as a pair. If one of the inputs has more than
%           one column while the other one only has one, the latter is
%           expanded to match the first one.
%   ell     Length scale or matrix of length scales (optional, default: 1)
%   sigma2  Variance (optional, default: 1)
%
% RETURNS
%   Kxx       Covaraince between x1 and x2
%
% VERSION
%   2017-12-21
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Does not work for other than scalar inputs yet.

    %% Defaults
    narginchk(2, 4);
    if nargin < 3 || isempty(ell)
        ell = 1;
    end
    if nargin < 4 || isempty(sigma2)
        sigma2 = 1;
    end
if 0
    [x1, x2] = expand_gp_inputs(x1, x2);
end
    
    %% Calculate Covariance
    r = sqrt(sum((x1-x2).^2, 1));
    Kxx = sigma2*exp(-1/2*(r/ell)^2);
end
