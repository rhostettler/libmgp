# Binary Gaussian process classifier training using Laplace approximation
## Usage
* `[f, Sigma, lpy] = train_classifier_laplace(x, y)`
* `[f, Sigma, lpy] = train_classifier_laplace(x, y, k, epsilon, J)`
 
## Description
Training of binary Gaussian process classifiers using the Laplace
approximation.
 
## Input
* `x`: dx-times-N matrix of training input values.
* `y`: 1-times-N vector of training class labels (-1 / +1).
* `k`: Covariance function (default: `k_se`).
* `epsilon`: Convergence tolerance (in relative marginal loglikelihood 
  change; default: 1e-3).
* `J`: Maximum number of iterations before stopping (default: 10).

## Output
* `f`, `Sigma`: Latent GP function values and variance.
* `lpy`: Marginal log-likelihood.
 
## Authors
2017-present -- Roland Hostettler
