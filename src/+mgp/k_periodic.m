function K = k_periodic(x1, x2, l_gp, var_gp)
% Periodic covariance function
%
% SYNOPSIS
%   K = k_periodic(x1, x2)
%   K = k_periodic(x1, x2, l_gp, var_gp)
%
% DESCRIPTION
%   Periodic covariance function of the form
%
%       k(x1, x2) = var_gp*exp(-2*sin(abs(x1-x2))^2/l_gp^2).
%
% PARAMETERS
%   x1, x2  Inputs
%
%   l_gp    Lenght scale (optional, default: pi/2)
%
%   var_gp  Variance magnitude (optional, default: 1)
%
% RETURNS
%   K       The covariance of x1 and x2
%
% VERSION
%   2017-06-05
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(2, 4);
    if nargin < 3 || isempty(l_gp)
        l_gp = pi/2;
    end
    if nargin < 4 || isempty(var_gp)
        var_gp = 1;
    end
if 0
    [x1, x2] = expand_gp_inputs(x1, x2);
end    

    %% Covariance
    r = abs(x1-x2);
    K = var_gp*exp(-2*sin(r/2).^2/l_gp^2);
end
