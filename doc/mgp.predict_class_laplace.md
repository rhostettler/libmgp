# Binary Gaussian process class predictor using the Laplace approximation
## Usage
* `rho = predict_class_laplace(xp, xt, ft)`
* `[rho, fp, Sigmap] = predict_class_laplace(xp, xt, ft, k)`
 
## Description
Predicts the binary class probability using the Laplace approximation and
a latent GP.
 
## Input
* `xp`: dx-times-Np matrix of test inputs.
* `xt`: dx-times-Nt matrix of training inputs.
* `ft`: 1-times-Nt vector of latent GP values estimated using the Laplace
  approximation.
* `k`: Covariance function (default: `k_se`).
 
## Output
* `rho`: Predicted class probability.
* `fp`, `Sigmap`: Predicted latent GP values and covariance.
 
## Authors
2021-present -- Roland Hostettler
