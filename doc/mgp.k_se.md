# Squared exponential covariance function
## Usage
* `Kxx = k_se(x1, x2)`
* `Kxx = k_se(x1, x2, ell, sigma2)`
 
## Description
Squared exponential covariance function given by
 
  k(x1, x2) = sigma2*exp(-1/2*(|x1-x2|/elll)^2).

## Input
* `x1`, `x2`:  Input points. If x1 and x2 have more than one column, each
  column is treated as a pair. If one of the inputs has more than one 
  column while the other one only has one, the latter is expanded to 
  match the first one.
* `ell`: Length scale or matrix of length scales (default: 1).
* `sigma2`: Variance (optional, default: 1).
 
## Output
* `Kxx`: Covaraince between x1 and x2.
 
## Author
2017-present -- Roland Hostettler
