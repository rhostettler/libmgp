function [rho, fp, Sigmap] = predict_class_laplace(xp, xt, ft, k)
% # Binary Gaussian process class predictor using the Laplace approximation
% ## Usage
% * `rho = predict_class_laplace(xp, xt, ft)`
% * `[rho, fp, Sigmap] = predict_class_laplace(xp, xt, ft, k)`
%
% ## Description
% Predicts the binary class probability using the Laplace approximation and
% a latent GP.
%
% ## Input
% * `xp`: dx-times-Np matrix of test inputs.
% * `xt`: dx-times-Nt matrix of training inputs.
% * `ft`: 1-times-Nt vector of latent GP values estimated using the Laplace
%   approximation.
% * `k`: Covariance function (default: `k_se`).
%
% ## Output
% * `rho`: Predicted class probability.
% * `fp`, `Sigmap`: Predicted latent GP values and covariance.
%
% ## Authors
% 2021-present -- Roland Hostettler

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
% * Check that the class probability prediction is implemented correctly.
% * Carefully check the details regarding the prediction, it should also 
%   take the "measurement noise" into account (Sigma).
% * Consider merging with predict_class_ep.

    %% Defaults
    narginchk(3, 4);
    if nargin < 4 || isempty(k)
        k = @mgp.k_se;
    end
    
    %% Prediction
    [fp, Sigmap] = mgp.predict(xp, xt, ft, 0, [], k);
    rho = normcdf(fp./sqrt(1+diag(Sigmap).'));
end
