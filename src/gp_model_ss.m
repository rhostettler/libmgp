function [F, Q, C, P0, Nx] = gp_model_ss(gpk_t, gpk_u, ut, up, Ts)
% Discrete-time state-space model for separable spatio-temporal GPs
% 
% USAGE
%   [F, Q, C, P0] = GP_MODEL_SS()
%   [F, Q, C, P0, Nx] = GP_MODEL_SS(gpk_t, gpk_u, ut, up, Ts)
%
% DESCRIPTION
%   Generic discrete-time state-space representation for temporal and
%   spatio-temporal Gaussian processes with separable covariance function.
%   The model is of the form
%
%       x[n] = F x[n-1] + q[n]
%       f[n] = C x[n]
%
%   where x[0] ~ N(m0, P0) and q[n] ~ N(0, Q).
%
%   Note that if both training (ut) and prediction (up) locations are
%   given, they are stacked such that the first 1...Nx*Nt states correspond
%   to the states of the training locations (and their derivatives) and the
%   states Nx*Nt+1...Nx*Nt+Nx*Np correspond to the test locations.
%
% PARAMETERS
%   gpk_t   Temporal covariance function (state-space representation; 
%           default: gpk_matern_ss).
%   gpk_u   Spatial covariance function (default: none).
%   ut      Inducing (training) points for the spatial variable. Only
%           required if gpk_u is not empty.
%   up      Prediction points for the spatial variable (default: []).
%   Ts      Sampling time (default: 1).
%
% RETURNS
%   F, Q, C, P0
%           Discrete-time linear state-space model parameters.
%   Nx      For temporal models, this is the state dimension and for
%           spatio-temporal models, this is the dimension of the state for
%           one spatial inducing point.
%
% REFERENCES
%   [1] R. Hostettler, S. Sarkka, S. J. Godsill, "Rao-Blackwellized
%       particle MCMC for parameter estimation in spatio-temporal Gaussian
%       processes," in 27th IEEE International Workshop on Machine Learning
%       for Signal Processing (MLSP), Tokyo, Japan, September 2017.
%
% AUTHOR
%   2018-03-22 -- Roland Hostettler <roland.hostettler@aalto.fi>

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

    %% Defaults
    narginchk(0, 5)
    if nargin < 1 || isempty(gpk_t)
        gpk_t = @gpk_matern_ss;
    end
    if nargin < 2 || isempty(gpk_u)
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
        
    %% Convert System
    % Get state-space representation of the temporal covariance function
    [A, B, C, Sw, P0] = gpk_t();
    [F, Q] = lti_disc(A, B, Sw, Ts);
    Q = (Q+Q')/2;
    Nx = size(F, 1);
        
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
        C = [kron(eye(Nt), C), zeros(Nt, Np*Nx)];

        % The process noise covariance between the ujs' subsystems is given
        % by k(u, u')*Q. Hence, the Kronecker produc between the full Kuu
        % and Q yields the full large covariance. The same applies for the
        % covariance matrix of the initial state.
        Kuu = gp_calculate_covariance(u, gpk_u);
        Q = kron(Kuu, Q);
        P0 = kron(Kuu, P0);
    end
end
