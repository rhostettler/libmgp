function [rho, fp, Cp] = gpc_predict_ep(xp, xt, yt, k, nu, tau)
% Binary Gaussian process class predictor using Expectation Propagation


    NTest = size(xp, 2);
    NTrain = size(xt, 2);
    
    K = gp_calculate_covariance([xp, xt], k);
    Kxpxp = K(1:NTest, 1:NTest);
    Kxpxt = K(1:NTest, NTest+1:NTest+NTrain);
    Kxtxt = K(NTest+1:NTest+NTrain, NTest+1:NTest+NTrain);

    S_tilde = diag(tau);
    B = eye(NTrain) + sqrt(S_tilde)*Kxtxt*sqrt(S_tilde);
    L = chol(B, 'lower');
    z = sqrt(S_tilde)*(L\(L'\(sqrt(S_tilde)*Kxtxt*nu)));
    fp = Kxpxt*(nu - z);
    
    Cp = zeros(NTest, NTest);
    for i = 1:NTest
        v = L\(sqrt(S_tilde)*Kxpxt(i, :)');
        Cp(i, i) = Kxpxp(i, i) - v'*v;
    end
    rho = normcdf(fp./sqrt(1+diag(Cp)));
end
