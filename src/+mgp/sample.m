function f = sample(x, varargin)
% # Sample from a Gaussian process
% ## Usage
% * `f = sample(x)`
% * `f = sample(x, k)`
% * `f = sample(x, m, k)`
% * `f = sample(x, m, k, M)`
%
% ## Description
% Samples a set of function values `f` from a Gaussian process with mean
% `m(x)` and covariance function `k(x1, x2)`.
%
% ## Input
% * `x`: dx*N matrix of N input values.
% * `m(x)`: Mean function (default: zero).
% * `k(x1, x2)`: Covariance function.
% * `
%
% ## Output
% * `f`: The sampled function values.
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
    narginchk(1, 4);
    N = size(x, 2);
    switch nargin
        case 1
            m = zeros(1, N);
            k = @mgp.k_se;
            M = 1;
        case 2
            m = zeros(1, N);
            k = varargin{1};
            M = 1;
        case 3
            if isa(varargin{2}, 'function_handle')
                m = varargin{1};
                k = varargin{2};
                M = 1;
            else
                m = zeros(1, N);
                k = varargin{1};
                M = varargin{2};
            end
        case 4
            m = varargin{1};
            k = varargin{2};
            M = varargin{3};
        otherwise
            % nop
    end
    
    if isa(m, 'function_handle')
        m = m(x);
    end
    
    %% Generate the data
    Kxx = mgp.calculate_covariance(x, k);
    Lxx = chol(Kxx).';
    f = ones(M, 1)*m + (Lxx*randn(N, M)).';
end
