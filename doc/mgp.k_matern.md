# Matern covariance function
## Usage
* `k = k_matern(x1, x2)`
* `k = k_matern(x1, x2, ell, sigma2, nu)`
 
## Description
Matern covariance function defined as
                                  _                    _            _                    _ 
                       2^(1-nu)  | (2*nu)^(1/2)*|x1-x2| |^nu       | (2*nu)^(1/2)*|x1-x2| |
  k(x1, x2) = sigma2 * --------- | -------------------- |    * K_nu| -------------------  |
                       Gamma(nu) |_        l           _|          |_        l           _|
 
where Gamma(.) is the gamma function and K_nu the Bessel function of the
second kind of order nu.
 
## Input
* `x1`, `x2`: Input points. If `x1` and `x2` have more than one column, 
  each column is treated as a pair. If one of the inputs has more than 
  one column while the other one only has one, the latter is expanded to 
  match the first one.
* `ell`: Length scale (default: 1).
* `sigma2`: Variance (default: 1).
* `nu`: Order (default: 1.5).
 
## Output
* `Kxx`: Covaraince between `x1` and `x2`.

## Authors
* 2017-present -- Roland Hostettler
