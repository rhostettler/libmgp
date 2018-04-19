function model = gp_model_smc(gpk_t, gpk_u, ut, up, py, ptheta, Ts, m0)
% State-space Gaussian process model with arbitrary likelihood
% 
% USAGE
%   model = GP_MODEL_SMC()
%   model = GP_MODEL_SMC(gpk_t, gpk_u, ut, up, py, ptheta, Ts, m0)
%
% DESCRIPTION
%   Implements a generic state-space representation for latent Gaussian
%   processes with arbitrary likelihoods. The model is useful together with
%   SMC-based methods, see [1] and the libsmc library. When Kalman filters 
%   are to be used, this model constructor is not useful, use gp_model_ss
%   instead.
%
%   Both temporal and spatio-temporal models can be constructed using this
%   function. For spatio-temporal models, the limitation right now is that
%   only separable covariance functions are accepted.
%
% PARAMETERS
%   gpk_t   Temporal covariance function (state-space representation; 
%           default: gpk_matern_ss).
%   gpk_u   Spatial covariance function (default: none).
%   ut      Inducing (training) points for the spatial variable. Only
%           required if gpk_u is not empty.
%   up      Prediction points for the spatial variable (default: []).
%   py      Measurement likelihood structure. Must follow the common pdf 
%           format (default: linear, Gaussian likelihood with unit 
%           variance).
%   ptheta  (Hyper)parameter prior, cell array with one prior for each
%           parameter in the order of the parameter vector (default: {}).
%           The pdf descriptions must follow the common pdf format.
%           Additionally, a field 'parameters' may be included that
%           includes the prior's parameters.
%   Ts      Sampling time (default: 1).
%   m0      Initial mean (default: zeros(Nx, 1)).
%
% RETURNS
%   model   The model structure suitable for use together with libsmc.
%
% REFERENCES
%   [1] R. Hostettler, S. Sarkka, S. J. Godsill, "Rao-Blackwellized
%       particle MCMC for parameter estimation in spatio-temporal Gaussian
%       processes," in 27th IEEE International Workshop on Machine Learning
%       for Signal Processing (MLSP), Tokyo, Japan, September 2017.
%
% AUTHOR
%   2018-04-19 -- Roland Hostettler <roland.hostettler@aalto.fi>

% Copyright (C) 2018 Roland Hostettler <roland.hostettler@aalto.fi>
% 
% This file is part of the libgp Matlab toolbox.
%
% libgp is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free 
% Software Foundation, either version 3 of the License, or (at your option)
% any later version.
% 
% libgp is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
% details.
% 
% You should have received a copy of the GNU General Public License along 
% with libgp. If not, see <http://www.gnu.org/licenses/>.

% TODO:
%   * Allow for non-zero initial states
%   * Use a generic CLGSS model from libsmc once that is fully implemented
%     rather than re-implementing everything
%   * Use a struct-generating function for normal pdfs

    %% Defaults
    narginchk(0, 8)
    if nargin < 1 || isempty(gpk_t)
        gpk_t = @gpk_matern_ss;
    end
    if nargin < 2
        gpk_u = [];
    end
    if nargin < 3
        ut = [];
    end
    if nargin < 4
        up = [];
    end
    if nargin < 5
        py = [];
    end    
    if nargin < 6 || isempty(ptheta)
        ptheta = {};
    end
    if nargin < 7 || isempty(Ts)
        Ts = 1;
    end
    if nargin < 8
        m0 = [];
    end
    
    if isempty(ut)
        Jt = 1;
    else
        Jt = size(ut, 2);
    end
    if isempty(up)
        Jp = 0;
    else  
        Jp = size(up, 2);
    end
    
    %% Create model structure
    % Get state-space representation
    [F, Q, C, P0, N0] = gp_model_ss(gpk_t, gpk_u, ut, up, Ts);
    
    % Calculate the indices to the nonlinear and linear state components.
    % Only the training (ut) points are nonlinear (these are the states
    % that enter the likelihood).
    Nx = Jt*N0 + Jp*N0;
    in = (1:N0:Jt*N0-1).';          % Indices of the nonlinear states
    il = (2:N0).'*ones(1, Jt);      % Indices of the linear states
    a = ones(N0-1, 1)*(0:N0:(Jt-1)*N0);
    il = a(:) + il(:);
    il = [il; (Jt*N0+1:Nx).'];
    
    % Default initial state
    if isempty(m0)
        m0 = zeros(Nx, 1);
    elseif size(m0, 1) ~= Nx
        error('Size of initial state does not match state dimension.');
    end
    
    % Default likelihood N(f, 1)
    if isempty(py)
        py = struct();
        py.fast = true;
        py.logpdf = @(y, s, t) logmvnpdf(y.', s.', eye(Jt)).';
    end
                       
    % Initialize model struct
    model = struct();
    model.in = in;
    model.il = il;

    % Initial state
    LP0 = chol(P0).';
    model.px0 = struct();
    model.px0.fast = true;
    model.px0.rand = @(M) m0*ones(1, M) + LP0*randn(Nx, M);
    model.px0.logpdf = @(x) logmvnpdf(x.', (m0*ones(1, size(x, 2))).', P0).';
    model.m0 = m0;
    model.P0 = P0;
    
    % Dynamic model
    LQ = chol(Q).';
    model.fn = @(s, t) F(in, in)*s;
    model.fl = @(s, t) F(il, in)*s;
    model.Fn = @(s, t) F(in, il);
    model.Fl = @(s, t) F(il, il);
    model.F = @(t) F;
    model.Q = @(s, t) Q;
    model.Qn = @(s, t) Q(in, in);
    model.Ql = @(s, t) Q(il, il);
    model.Qnl = @(s, t) Q(in, il);
    model.px = struct();
    model.px.fast = true;
    model.px.rand = @(x, t) F*x + LQ.'*randn(N0, size(x, 2));
    model.px.logpdf = @(xp, x, t) logmvnpdf(xp.', (F*x).', Q).';

    % Likelihood and parameter prior; these need to be defined outside (or
    % the default is a linear, Gaussian likelihood, see above.
    model.py = py;
    model.ptheta = ptheta;
end
