function dK = dk_periodic(x1, x2, l_gp, var_gp)
% Derivative of the periodic covariance function
% 
% SYNOPSIS
%   dK = dk_periodic(x1, x2)
%   dK = dk_periodic(x1, x2, l_gp, var_gp)
%
% DESCRIPTION
%   Derivative of the peridiac covariance function with respect to the
%   first input x1 given by
%
%       d k(x1, x2)       1
%       ----------- = - ------ sin(x1-x2) k(x1, x2)
%          d x1         l_gp^2
%
%   where k(x1, x2) is the periodic covariance kernel itself.
%
% PARAMETERS
%   x1, x2  Inputs
%
%   l_gp    Length scale (optional, default: pi/2)
%
%   var_gp  Covariance magntiude (optional, default: 1)
%
% RETURNS
%   dK      The derivative
%
% SEE ALSO
%   k_periodic 
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

    %% Derivative of Covariance
    k = k_periodic(x1, x2, l_gp, var_gp);
    dK = -1/l_gp^2*sin(x1-x2).*k;
end
