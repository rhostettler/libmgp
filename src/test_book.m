



clear variables;
addpath ~/Projects/Misc/gpkit

x = [
    -6 + 0.8*randn(20, 1);
     0 + 0.8*randn(30, 1);
     2 + 0.8*randn(10, 1);
].';
y = [
       ones(20, 1);
    -1*ones(30, 1);
       ones(10, 1);
].';


ell = 2.6;
sigma = 7^2;
k = @(x1, x2) k_se(x1, x2, ell, sigma);

f_laplace = gpc_train_laplace(x, y, k);
[nu_ep, tau_ep, ~, f_ep] = gpc_train_ep(x, y, k);

xp = -9:0.1:4;
[rhop_ep, fp_ep] = gpc_predict_ep(xp, x, y, k, nu_ep, tau_ep);

fp_laplace = gp_predict(xp, x, f_laplace, 0, [], k);
rhop_laplace = normcdf(fp_laplace);

%%
figure(1); clf();
plot(x, f_laplace, '.'); hold on;
plot(x, f_ep, '.');
legend('Laplace', 'EP');
title('Posterior Mean');

figure(2); clf();
plot(xp, fp_laplace); hold on;
plot(xp, fp_ep);
legend('Laplace', 'EP');
title('Predicted Latent Function Values');

figure(3); clf();
plot(xp, rhop_laplace); hold on;
plot(xp, rhop_ep);
legend('Laplace', 'EP');
plot(x(y == 1), y(y == 1), 'xg');
plot(x(y == -1), y(y == -1), 'og');
title('Predicted Class Probabilities');
xlim([-9, 4]);
