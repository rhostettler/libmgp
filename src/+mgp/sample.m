function f = sample(x, k)
% Sample from a Gaussian Process
% ==============================
% Synopsis
% --------
% `f = gp_sample(x, k)`
%
% Description
% -----------
% Draws a set of samples `f` from a zero-mean Gaussian process with
% covariance function `k(x1, x2)`.
%
% Parameters
% ----------
% * `x`: Input values.
% * `k(x1, x2)`: Covariance function.
%
% Returns
% -------
% * `f`: The sampled function values.

% Changes
% -------
% 2017-08-10 - Roland Hostettler <roland.hostettler@aalto.fi>
% * Initial version

    narginchk(2, 2);
    N = size(x, 2);
    Kxx = gp_calculate_covariance(x, k);
    Lxx = chol(Kxx).';
    f = (Lxx*randn(N, 1)).';
end
