function k = k_se(x1, x2, l_gp, var_gp)
% Squared exponential covariance kernel
%
% SYNOPSIS
%   k = k_se(x1, x2)
%   k = k_se(x1, x2, l, var)
%
% DESCRIPTION
%   Squared exponential covariance function given by
%
%       k(x1, x2) = var*exp(-1/2*(|x1-x2|/l)^2).
% 
% PARAMETERS
%   x1, x2  Input points. If x1 and x2 have more than one column, each
%           column is treated as a pair. If one of the inputs has more than
%           one column while the other one only has one, the latter is
%           expanded to match the first one.
%
%   l       Length scale (optional, default: 1)
%
%   var     Variance (optional, default: 1)
%
% RETURNS
%   k       Covaraince between x1 and x2
%
% VERSION
%   2017-05-24
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(2, 4);
    if nargin < 3 || isempty(l_gp)
        l_gp = 1;
    end
    if nargin < 4 || isempty(var_gp)
        var_gp = 1;
    end
if 0
    [x1, x2] = expand_gp_inputs(x1, x2);
end
    
    %% Calculate Covariance
    r = sqrt(sum((x1-x2).^2, 1));
    k = var_gp*exp(-1/2*(r/l_gp)^2);
end
