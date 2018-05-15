function K = gp_calculate_covariance(x, k)
% Calculates the GP covariance matrix
%
% USAGE
%   K = GP_CALCULATE_COVARIANCE(x, k)
%
% DESCRIPTION
%   Calculates the covariance matrix for all columns in the input variable
%   x using the covariance kernel k.
%
% PARAMETERS
%   x   The Nx*N input vector
%   k   Function handle to the covariance function of the form k(x1, x2)
%       where x1 and x2 are Nx*1 vectors.
%
% RETURNS
%   K   The N*N covariance matrix
%
% AUTHORS
%   2017-05-24 -- Roland Hostettler <roland.hostettler@aalto.fi>

    narginchk(2, 2);
    N = size(x, 2);
    K = zeros(N, N);
    for i = 1:N
        for j = i:N
            K(i, j) = k(x(:, i), x(:, j));
        end
    end
    K = K + K' - diag(diag(K));
    K = (K+K')/2;
end
