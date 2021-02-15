function [S, dS] = k_se_psd(w, ell, sigma2)
% # Power spectral density of (1D) squared exponential covariance function
% ## Usage
% * `S = k_se_psd(w)`
% * `[S, dS] = k_se_psd(w, ell, sigma2)`
%
% ## Description
% Power spectral density of the squared exponential (SE) Gaussian process
% covariance function. Also returns a struct of the PSD's gradient with
% respect to the hyperparameters.
%
% ## Input
% * `w`: Angular frequency in rad/s.
% * `ell`: Length scale (default: 1).
% * `sigma2`: Magnitude (default: 1).
%
% ## Output
% * `S`: Power spectral density.
% * `dS`: Gradient of the power spectral density with respect to `ell` and
%   `sigma2`.
%
% ## Author
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
    if nargin < 2 || isempty(ell)
        ell = 1;
    end
    if nargin < 3 || isempty(sigma2)
        sigma2 = 1;
    end

    %% PSD
    S = log(sigma2) + 1/2*log(2*pi*ell.^2) - 1/2*(w*ell).^2;
    S = exp(S);
    
    %% Derivative of PSD w.r.t. parameters
    dSdell = 1/2*log(2*pi*ell.^2)-1/2*(w*ell).^2;
    dSdell = exp(dSdell);
    dSdsigma2 = sigma2*sqrt(2*pi)*exp(-1/2*(w*ell).^2).*( ...
        1 - ell*(w.^2.*ell) ...
    );
    dS = {dSdell; dSdsigma2};
end
