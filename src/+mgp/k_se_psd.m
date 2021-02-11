function [S, dS] = k_se_psd(w, ell, sigma2)
% Power Spectral Density of (1D) Squared Exponential Covariance Function
%
% USAGE
%   S = gpk_se_psd(w)
%   [S, dS] = gpk_se_psd(w, ell, sigma2)
%
% DESCRIPTION
%   Power spectral density of the squared exponential (SE) Gaussian process
%   covariance function. Also returns a struct of the PSD's derivatives
%   with respect to the hyperparameters.
%
% PARAMETERS
%   w       Circular frequence.
%   ell     Length scale (default: 1).
%   sigma2  Magnitude (default: 1).
%
% RETURNS
%   S       Power spectral density.
%   dS      Gradient of the power spectral density with respect to ell and
%           sigma2.
%
% AUTHOR
%   2017-12-21 -- Roland Hostettler <roland.hostettler@aalto.fi>

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
