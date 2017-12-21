function varargout = gp_k_quasiperiodic_ss(ell_p, sigma2, omega, ell_q, J, type)
% State space 
% 
 
    %% Defaults
    narginchk(0, 6);
    if nargin < 1 || isempty(ell_p)
        ell_p = 1;
    end
    if nargin < 2 || isempty(sigma2)
        sigma2 = 1;
    end
    if nargin < 3 || isempty(omega)
        omega = 2*pi;
    end
    if nargin < 4 || isempty(ell_q)
        ell_q = 1;
    end
    if nargin < 5 || isempty(J);
        J = 20;
    end
    if nargin < 6 || isempty(type)
        type = 'Continuous';
    end
    
    %% 
    lambda = 1/ell_q;
    
    % Spectral density
    j = 0:J;
    qj = 2*sigma2*exp(-1/ell_p^2)*besseli(j, 1/ell_p^2);
    qj(1) = qj(1)/2;                                        % DC component
    Pinf = blkdiag(qj(1), kron(diag(qj(2:end)), eye(2)));
    Q = 2*lambda*Pinf;
    
    % 
    C = [1, kron(ones(1, J), [1 0])];

    if strcmp(type, 'Continuous') || strcmp(type, 'continuous')
        A = zeros(2*J+1, 2*J+1);
        A(1, 1) = 1;
        for j = 1:J
            jj = 2*j:2*j+1;
            A(jj, jj) = [
                -lambda, -j*omega;
                j*omega,  -lambda;
            ];
        end
        B = eye(2*J+1);
        
        % Pack output
        varargout = cell([1, 5]);
        varargout{1} = A;
        varargout{2} = B;
        varargout{3} = Q;
        varargout{4} = C;
        varargout{5} = Pinf;
    elseif strcmp(type, 'Discrete') || strcmp(type, 'discrete')
        % Discretized process noise covariance
        Q = dt*Q;
        
        % Construct the state transition matrix including DC
        F = zeros(2*J+1, 2*J+1);
        F(1, 1) = 1; % DC component
        for j = 1:J
            jj = 2*j:2*j+1;
            F(jj, jj) = [
                cos(j*omega*dt), -sin(j*omega*dt);
                sin(j*omega*dt),  cos(j*omega*dt);
            ];
        end
    else
        error('Type must be continuous or discrete.');
    end
end
