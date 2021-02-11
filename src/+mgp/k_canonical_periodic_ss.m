function [A, B, C, Sw, Pinf] = k_canonical_periodic_ss(ell, sigma2, w0, J)
% # State Space Representation of the Canonical Periodic Covariance Kernel
% ## Usage
% [A, B, C, Sw, Pinf] = k_canonical_periodic_ss(ell, sigma2, w0, J)
%
% ## Description
%
% ## Parameters
% * `ell`:
% * `sigma2`: 
% * `w0`: 
% * `J`: Approximation order (default: 10).
%
% ## Returns
% * `A`:
%
% ## References
% [1] A. Solin and S. Särkkä
%
% ## Author(s)
% 2017-12-06 -- Roland Hostettler <roland.hostettler@aalto.fi>

% TODO:
% * Add documentation

    %% Defaults
    narginchk(0, 4);
    if nargin < 1 || isempty(ell)
        ell = 1;
    end
    if nargin < 2 || isempty(sigma2)
        sigma2 = 1;
    end
    if nargin < 3 || isempty(w0)
        w0 = 2*pi;
    end
    if nargin < 4 || isempty(J)
        J = 10;
    end

    %% Construct state space model
    % Drift matrix
    A = zeros(2*J+1, 2*J+1);
    %A(1, 1) = 1;
    for j = 1:J
        jj = 2*j+(0:1);
        A(jj, jj) = [
               0, -j*w0;
            j*w0,     0;
        ];
    end

    % Initial state covariance
    j = 0:J;
    qj = 2*sigma2*exp(-1/ell^2)*besseli(j, 1/ell^2);
    qj(1) = qj(1)/2;
    Pinf = blkdiag(qj(1), kron(diag(qj(2:J+1)), eye(2)));
    
    % Diffusion
    B = eye(2*J+1);
    Sw = zeros(2*J+1, 2*J+1);
    
    % Output
    C = [1, kron(ones(1, J), [1, 0])];
end
