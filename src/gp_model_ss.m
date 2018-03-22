function model = gp_model_ss(gpk_t, gpk_u, ut, up, py, ptheta, Ts)
% State-space Gaussian process model with arbitrary likelihood
% 
% USAGE
%   model = GP_MODEL_SS()
%   model = GP_MODEL_SS(gpk_t, gpk_u, ut, up, py, ptheta, Ts)
%
% DESCRIPTION
%   Implements a generic state-space representation for latent Gaussian
%   processes with arbitrary likelihoods. The model is useful together with
%   SMC-based methods, see [1]. When Kalman filters are to be used, this
%   model constructor is not useful.
%
%   Both temporal and spatio-temporal models can be constructed using this
%   routine. For spatio-temporal models, the limitation right now is that
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
%           parameter in the order of the parameter vector (defualt: {}).
%           The pdf descriptions must follow the common pdf format.
%   Ts      Sampling time (default: 1).
%
% RETURNS
%   model   The model structure.
%
% REFERENCES
%   [1] R. Hostettler, S. Sarkka, S. J. Godsill, "Rao-Blackwellized
%       particle MCMC for parameter estimation in spatio-temporal Gaussian
%       processes," in 27th IEEE International Workshop on Machine Learning
%       for Signal Processing (MLSP), Tokyo, Japan, September 2017.
%
% VERSION
%   2018-03-22
%
% AUTHOR
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
%   * Allow for non-zero initial states
%   * Use a generic CLGSS model once that is fully implemented rather than
%     re-implementing 
%   * Use a struct-generating function for normal pdfs

    %% Defaults
    narginchk(0, 7)
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
    if nargin < 5
        py = [];
    end
    if nargin < 6 || isempty(ptheta)
        ptheta = {};
    end
    if nargin < 7 || isempty(Ts)
        Ts = 1;
    end
        
    %% Convert System
    % Get state-space representation of the temporal covariance function
    [A, B, C, Sw, P0] = gpk_t();
    [F, Q] = lti_disc(A, B, Sw, Ts);
    Q = (Q+Q')/2;
    N0 = size(F, 1);
        
    % Add spatial part, if applicable
    if ~temporal
        % Generate a system for each point uj
        Jt = size(ut, 2);
        Jp = size(up, 2);
        J =  Jt + Jp;
        u = [ut, up];
    
        % State transition matrix: A block-diagonal matrix with the matrix
        % F on the diagonal
        F = kron(eye(J), F);
    
        % Measurement vector: A block vector similar to the above but with 
        % no measurements for the prediction points up.
        C = [kron(eye(Jt), C), zeros(Jt, Jp*N0)];

        % The process noise covariance between the ujs' subsystems is given
        % by k(u, u')*Q. Hence, the Kronecker produc between the full Kuu
        % and Q yields the full large covariance. The same applies for the
        % covariance matrix of the initial state.
        Kuu = gp_calculate_covariance(u, gpk_u);
        Q = kron(Kuu, Q);
        P0 = kron(Kuu, P0);
    else
        Jt = 1;
        Jp = 0;
    end
    
    % Default likelihood
    if isempty(py)
        py = struct();
        py.fast = true;
        py.logpdf = @(y, s, t) logmvnpdf(y.', s.', eye(Jt)).';
    end
    
    %% Create model structure
    % Calculate the indices to the nonlinear and linear state components.
    % Only the training (ut) points are nonlinear (these are the states
    % that enter the likelihood).
    Nx = Jt*N0 + Jp*N0;
    in = (1:N0:Jt*N0-1).';          % Indices of the nonlinear states
    il = (2:N0).'*ones(1, Jt);      % Indices of the linear states
    a = ones(N0-1, 1)*(0:N0:(Jt-1)*N0);
    il = a(:) + il(:);
    il = [il; (Jt*N0+1:Nx).'];
                       
    % Initialize model struct
    model = struct();
    model.in = in;
    model.il = il;

    % Initial state
    m0 = zeros(N0, 1);  
    LP0 = chol(P0).';
    model.px0 = struct();
    model.px0.fast = true;
    model.px0.rand = @(M) m0*ones(1, M) + LP0*randn(N0, M);
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
