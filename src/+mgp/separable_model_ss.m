function [F, Q, C, m0, P0, dx] = separable_model_ss(k_t, k_u, ut, up, Ts, f0)
% # Discrete-time state-space model for separable spatio-temporal GPs
% ## Usage
% * `[F, Q, C, P0] = separable_model_ss()`
% * `[F, Q, C, m0, P0, dx] = separable_model_ss(k_t, k_u, ut, up, Ts, f0)`
%
% ## Description
% Generic discrete-time state-space representation for temporal and spatio-
% temporal Gaussian processes with separable covariance function. The model
% is of the form
%
%   x[n] = F x[n-1] + q[n]
%   f[n] = C x[n]
%
% where x[0] ~ N(m0, P0) and q[n] ~ N(0, Q).
%
% Note that if both training (ut) and prediction (up) locations are given,
% they are stacked such that the first 1, ..., dx*Nt states correspond to 
% the states of the training locations (and their derivatives) and the 
% states dx*Nt+1, ..., dx*Nt+dx*Np correspond to the test locations.
%
% ## Input
% * `k_t`: Temporal covariance function (state-space representation; 
%   default: `@k_matern_ss`).
% * `k_u`: Spatial covariance function (default: none).
% * `ut`: Inducing (training) points for the spatial variable. Only
%   required if `k_u` is not empty.
% * `up`: Prediction points for the spatial variable (default: []).
% * `Ts`: Sampling time (default: 1).
% * `f0`: Initial mean of the GP, that is, f(u, 0).
%
% ## Output
% * `F`, `Q`, `C`, `m0`, `P0`: Discrete-time linear state-space model 
%   parameters.
% * `dx`: For temporal models, this is the state dimension and for spatio-
%   temporal models, this is the dimension of the state for one spatial 
%    inducing points.
%
% ## References
% 1. R. Hostettler, S. Sarkka, S. J. Godsill, "Rao-Blackwellized particle 
%    MCMC for parameter estimation in spatio-temporal Gaussian processes,"
%    in 27th IEEE International Workshop on Machine Learning for Signal 
%    Processing (MLSP), Tokyo, Japan, September 2017.
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

    %% Defaults
    narginchk(0, 6)
    if nargin < 1 || isempty(k_t)
        k_t = @mgp.k_matern_ss;
    end
    if nargin < 2 || isempty(k_u)
        temporal = true;
    else
        temporal = false;
        if nargin < 3 || isempty(ut)
            error('Please specify points for spatio-temporal models.');
        end
        if nargin < 4
            up = [];
        end
    end
    if nargin < 5 || isempty(Ts)
        Ts = 1;
    end
    if nargin < 6
        f0 = [];
    end
        
    %% Convert System
    % Get state-space representation of the temporal covariance function
    [A, B, C, Sw, P0] = k_t();
    [F, Q] = lti_disc(A, B, Sw, Ts);
    Q = (Q+Q')/2;
    dx = size(F, 1);
    J = 1;
        
    % Add spatial part, if applicable
    if ~temporal
        % Generate a system for each point uj
        Nt = size(ut, 2);
        Np = size(up, 2);
        J =  Nt + Np;
        u = [ut, up];
            
        % State transition matrix: A block-diagonal matrix with the matrix
        % F on the diagonal
        F = kron(eye(J), F);
    
        % Measurement vector: A block vector similar to the above but with 
        % no measurements for the prediction points up.
        C = [kron(eye(Nt), C), zeros(Nt, Np*dx)];

        % The process noise covariance between the ujs' subsystems is given
        % by k(u, u')*Q. Hence, the Kronecker produc between the full Kuu
        % and Q yields the full large covariance. The same applies for the
        % covariance matrix of the initial state.
        Kuu = mgp.calculate_covariance(u, k_u);
        Q = kron(Kuu, Q);
        P0 = kron(Kuu, P0);
    end
    
    % Add m0
    m0 = zeros(J*dx, 1);
    if isempty(f0)
        f0 = zeros(J, 1);
    end
    m0(1:dx:J*dx) = f0;
end
