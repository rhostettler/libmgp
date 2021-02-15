function K = calculate_covariance(x, varargin)
% # Calculates the GP covariance matrix
% ## Usage
% * `K = calculate_covariance(x)`
% * `K = calculate_covariance(x, k)`
% * `K = calculate_covariance(x1, x2)`
% * `K = calculate_covariance(x1, x2, k)`
%
% ## Description
% Calculates the covariance matrix for all columns in the input variable
% `x` using the covariance kernel `k`. If two inputs `x1` and `x2` are
% provided, the cross-covariance between `x1` and `x2` is calculated.
%
% ## Input
% * `x`: dx-times-N matrix of input values where each column is a
%   dx-dimensional input value.
% * `x1` and `x2`: dx-times-N1 and dx-times-N2 matrices of input values for
%   calculating the cross-covariance between the `x1`s and `x2`s.
% * `k`: Function handle of the covariance function of the form `k(x1, x2)`
%   (default: @k_se).
%
% ## Output
% * `K`: N-times-N or N1-times-N2 (cross-)covariance matrix.
%
% ## Authors
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

    %% Defaults
    narginchk(1, 3);
    x1 = x;
    if nargin < 2
        x2 = x;
        k = @k_se;
    end
    if nargin < 3
        if isa(varargin{1}, 'function_handle')
            x2 = x;
            k = varargin{1};
        else
            x2 = varargin{1};
            k = @gpk_se;
        end
    end
    if nargin == 3
        x2 = varargin{1};
        k = varargin{2};
    end
    
    %% Calculate covaraince matrix
    N1 = size(x1, 2);
    N2 = size(x2, 2);
    K = zeros(N1, N2);
    for i = 1:N1
        for j = i:N2
            K(i, j) = k(x1(:, i), x2(:, j));
        end
    end
    
    % TODO: We should do this for the N1 != N2 case
    if N1 == N2
        K = K + K' - diag(diag(K));
        K = (K+K')/2;
    end
end
