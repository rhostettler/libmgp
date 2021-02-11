function [mp, Cp] = predict(xp, xt, yt, R, m, k)
% Gaussian process prediction
%
% SYNOPSIS
%   [mp, Cp] = gp_predict(xp, xt, yt, R, m, k)
%
% DESCRIPTION
%   Predicts the output of a Gaussian process f ~ GP(m(x), k(x, x') for the
%   test input xp, given the training inputs xt and (noisy) output 
%   measurements yt = f(xt) + r where r ~ N(0, R).
%
% PARAMETERS
%   xp      NxMp matrix of Mp test input
%
%   xt      NxMt matrix of Mt training inputs
%
%   yt      1xMt vector of Mt (noisy) training measurements
%
%   R       Measurement noise variance (optional, default: 1)
%
%   m       GP mean function m = @(x) ... (optional, default: 0)
%
%   k       GP covariance function k = @(x1, x2) ... (optional, default:
%           k_se)
%
% RETURNS
%   mp      Predicted mean
%
%   Cp      Predicted covariance
%
% VERSION
%   2017-08-10
%
% AUTHORS
%   Roland Hostettler <roland.hostettler@aalto.fi>

% TODO
%   * Handle case when we don't have any training data => return the prior

    %% Parameters & defaults
    narginchk(3, 6)
    NTest = size(xp, 2);
    NTrain = size(xt, 2);
    Ny = size(yt, 1);
    if nargin < 4 || isempty(R)
        R = 1;
    end
    if nargin < 5 || isempty(m)
        m = @(x) zeros(Ny, size(x, 2));
    end
    if nargin < 6 || isempty(k)
        k = @(x1, x2) k_se(x1, x2);
    end
    
    %% Make Prediction
    % Mean
    Mxp = m(xp);
    Mxt = m(xt);

    % Covariance
    K = gp_calculate_covariance([xp, xt], k);
    Kxpxp = K(1:NTest, 1:NTest);
    Kxpxt = K(1:NTest, NTest+1:NTest+NTrain);
    Kxtxt = K(NTest+1:NTest+NTrain, NTest+1:NTest+NTrain);
    
    % Predict
    S = Kxtxt + R*eye(NTrain);
    L = Kxpxt/S;
    mp = (Mxp.' + L*(yt - Mxt).').';
    Cp = Kxpxp - L*S*L';
end
