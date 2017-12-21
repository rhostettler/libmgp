function k = k_matern(x1, x2, l_gp, var_gp, nu_gp)
% Matern covariance function
% 
% SYNOPSIS
%   k = k_matern(x1, x2)
%   k = k_matern(x1, x2, l, var, nu)
%
% DESCRIPTION
%   Matern covariance function defined as
%                                        _                    _            _                    _ 
%                             2^(1-nu)  | (2*nu)^(1/2)*|x1-x2| |^nu       | (2*nu)^(1/2)*|x1-x2| |
%       k(x1, x2) = sigma^2 * --------- | -------------------- |    * K_nu| -------------------  |
%                             Gamma(nu) |_        l           _|          |_        l           _|
%
%   where Gamma(.) is the gamma function and K_nu the Bessel function of
%   the second kind of order nu.
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
%   nu      Order (optional, default: 1.5)
%
% RETURNS
%   k   Covaraince between x1 and x2
% 
% VERSION
%   2017-05-17
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

    %% Defaults
    narginchk(2, 5);
    if nargin < 3 || isempty(l_gp)
        l_gp = 1;
    end
    if nargin < 4 || isempty(var_gp)
        var_gp = 1;
    end
    if nargin < 5 || isempty(nu_gp)
        nu_gp = 1.5;
    end
if 0
    [x1, x2] = expand_gp_inputs(x1, x2);
end
    
    %% Calculate Covariance
    r = sqrt(sum((x1-x2).^2, 1));
    switch nu_gp
        case 1.5
            k = var_gp*(1 + sqrt(3)*r/l_gp)*exp(-sqrt(3)*r/l_gp);
            
        case 2.5
            k = var_gp*(1 + sqrt(5)*r/l_gp + 5*r.^2/(3*l_gp^2))*exp(-sqrt(5)*r/l_gp);
            
        otherwise
            k = var_gp*2^(1-nu_gp)/gamma(nu_gp)*(sqrt(2*nu_gp)/l_gp*r)^nu_gp*besselk(nu_gp, sqrt(2*nu_gp)/l_gp*r);
            % TODO: This is not quite true.
            k(r == 0) = var_gp;
    end
end
