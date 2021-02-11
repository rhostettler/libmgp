function [f, Sigma, lpy] = train_classifier_laplace(x, y, k, par)
% Binary Gaussian process classifier training using Laplace approximation


% TODO:
%   * Convergence criterion
%   * Parameter handling

    narginchk(3, 4);
    if nargin < 4
        par = [];
    end
    def = struct(...
        'J', 10, ...        % Maximum no. of iterations
        'epsilon', 1e-3 ... % Convergence tolerance (in norm of posterior mean change)
    );
    par = parchk(par, def);
    
    %% Initialize
    % Calculate covaraince matrix
    K = gp_calculate_covariance(x, k);

    % Preallocate
    y = y(:);
    N = size(y, 1);
    f = zeros(N, 1);

    done = false;
    j = 0;
    while ~done
        fp = f;
        j = j+1;
        
        Nyf = normpdf(y.*f);
        Cyf = normcdf(y.*f);
        H = -Nyf.^2./Cyf.^2 - y.*f.*Nyf./Cyf;
        W = -diag(H);
        B = eye(N) + sqrt(W)*K*sqrt(W);
        L = chol(B, 'lower');

        g = y.*Nyf./Cyf;
        b = W*f + g;
        v = L\(sqrt(W)*K);
        a = b - sqrt(W)*(L'\(v*b));
        f = K*a;
        Sigma = K - v'*v;
        
%         norm(f-fp)/N
        done = (j >= par.J);
    end
    
    % Calculate approximate log marg likelihood
    lpy = -1/2*a'*f + sum(log(normcdf(y.*f))) - trace(log(L));
    f = f.';
end
