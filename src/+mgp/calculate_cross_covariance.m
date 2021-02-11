function K12 = calculate_cross_covariance(x1, x2, k)
% Calculates the cross-covariance between two inputs
%
% SYNOPSIS
%   K12 = GP_CALCULATE_CROSS_COVARIANCE(x1, x2, k)
%
% DESCRIPTION
%   Calculates the cross-covariance between two variables x1 and x2 using
%   the covariance kernel k, that is, it calculates Cov{x1, x2} for each 
%   pair in x1 and x2.
%
% PARAMETERS
%
% RETURNS
%   K12     N1 times N2 
% 
% SEE ALSO
%   gp_calculate_covariance
%
% AUTHORS
%   2017-11-15 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
% * Merge with gp_calculate_covariance (use varargin there)

    narginchk(3, 3);
    N1 = size(x1, 2);
    N2 = size(x2, 2);
    K12 = zeros(N1, N2);
    for i = 1:N1
        for j = 1:N2
            K12(i, j) = k(x1(:, i), x2(:, j));
        end
    end
end
