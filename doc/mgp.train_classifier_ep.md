# Binary Gaussian process classifier training using expectation propagation
## Usage
* `[nu, tau] = train_classifier_ep(y, k)`
* `[nu, tau, lpy, mu, Sigma] = train_classifier_ep(y, k, epsilon, J)`
 
## Description
Training of binary Gaussian process classifiers using the expectation
propagation.
 
## Input
* `x`: dx-times-N matrix of training input values.
* `y`: 1-times-N vector of training class labels (-1 / +1).
* `k`: Covariance function (default: `k_se`).
* `epsilon`: Convergence tolerance (in relative marginal loglikelihood 
  change; default: 1e-3).
* `J`: Maximum number of iterations before stopping (default: 10).
 
## Output
* `nu`, `tau`: Trained parameters.
* `lpy`:  Marginal loglikelihood.
* `mu`, `Sigma`: Latent GP posterior mean and covariance.
 
## Authors
2017-present -- Roland Hostettler
