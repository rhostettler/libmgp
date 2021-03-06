function [rho, fp, Sigmap] = predict_class_ep(xp, xt, ft, k)
% # Binary Gaussian process class predictor using expectation propagation
% ## Usage
% * `rho = predict_class_ep(xp, xt, ft)
% * `[rho, fp, Sigmap] = predict_class_ep(xp, xt, ft, k)
%
% ## Description
% % Predicts the binary class probability using expecation propagation and
% a latent GP.
%
% ## Input
% * `xp`: dx-times-Np matrix of test inputs.
% * `xt`: dx-times-Nt matrix of training inputs.
% * `ft`: 1-times-Nt vector of latent GP values estimated using expectation
%   propagation.
% * `k`: Covariance function (default: `k_se`).
%
% ## Output
% * `rho`: Predicted class probability.
% * `fp`, `Sigmap`: Predicted latent GP values and covariance.
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

% TODO:
% * Remove legacy code and/or consider merging with predict_class_laplace
%   since they essentially are the same.
% * It should also take the "measurement noise" into account

    %% Defaults
    narginchk(3, 4);
    if nargin < 4 || isempty(k)
        k = @mgp.k_se;
    end

    %% Prediction
if 0
    Np = size(xp, 2);
    Nt = size(xt, 2);
    K = mgp.calculate_covariance([xp, xt], k);
    Kpp = K(1:Np, 1:Np);
    Kpt = K(1:Np, Np+1:Np+Nt);
    Ktt = K(Np+1:Np+Nt, Np+1:Np+Nt);

    nu = nu(:);
    S_tilde = diag(tau);
    B = eye(Nt) + sqrt(S_tilde)*Ktt*sqrt(S_tilde);
    L = chol(B, 'lower');
    z = sqrt(S_tilde)*(L\(L'\(sqrt(S_tilde)*Ktt*nu)));
    fp = Kpt*(nu - z);
    
    Sigmap = zeros(Np, Np);
    for i = 1:Np
        v = L\(sqrt(S_tilde)*Kpt(i, :)');
        Sigmap(i, i) = Kpp(i, i) - v'*v;
    end
end
    [fp, Sigmap] = mgp.predict(xp, xt, ft, 0, [], k);
    rho = normcdf(fp./sqrt(1+diag(Sigmap).'));
    
    %% Output
    rho = rho.';
    fp = fp.';
end
