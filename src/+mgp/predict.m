function [mp, Cp] = predict(xp, xt, yt, R, m, k)
% # Gaussian process prediction
% ## Usage
% * `[mp, Cp] = predict(xp, xt, yt)`
% * `[mp, Cp] = predict(xp, xt, yt, R, m, k)`
%
% ## Description
% Predicts the output of a Gaussian process f ~ GP(m(x), k(x, x') for the
% test input `xp`, given the training inputs `xt` and (noisy) output 
% measurements yt = f(xt) + r where r ~ N(0, R).
%
% ## Input
% * `xp`: dx-times-Np matrix of Np test inputs.
% * `xt`: dx-times-Nt matrix of Nt training inputs.
% * `yt`: 1-times-Nt vector of Nt (noisy) training outputs.
% * `R`: Measurement noise variance (default: 1).
% * `m`: Mean function m = @(x) ... (default: 0).
% * `k`: Covariance function k = @(x1, x2) ... (default: `k_se`).
%
% ## Output
%   mp      Predicted mean
%   Cp      Predicted covariance
%
% ## Authors
% 2018-present -- Roland Hostettler

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
% * Implement the possibility to take pre-calculated covariance matrices as
%   an input to make things more efficient. Should be combined with a
%   function estimate() or similar.
% * Handle case when we don't have any training data => return the prior

    %% Defaults
    narginchk(3, 6)
    Np = size(xp, 2);
    Nt = size(xt, 2);
    % Ny = size(yt, 1);
    if nargin < 4 || isempty(R)
        R = 1;
    end
    if nargin < 5 || isempty(m)
        m = 0;
    end
    if nargin < 6 || isempty(k)
        k = @mgp.k_se;
    end
        
    %% Prediction
    % Mean
    if isa(m, 'function_handle')
        mxp = m(xp);
        mxt = m(xt);
    else
        mxp = m*ones(1, Np);
        mxt = m*ones(1, Nt);
    end

    % Covariance
    K = mgp.calculate_covariance([xp, xt], k);
    Kpp = K(1:Np, 1:Np);
    Kpt = K(1:Np, Np+1:Np+Nt);
    Ktt = K(Np+1:Np+Nt, Np+1:Np+Nt);
    
    % Predict
    S = Ktt + R*eye(Nt);
    L = Kpt/S;
    mp = (mxp.' + L*(yt - mxt).').';
    Cp = Kpp - L*S*L';
end
