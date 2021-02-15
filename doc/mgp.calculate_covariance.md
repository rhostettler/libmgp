# Calculates the GP covariance matrix
## Usage
* `K = calculate_covariance(x)`
* `K = calculate_covariance(x, k)`
* `K = calculate_covariance(x1, x2)`
* `K = calculate_covariance(x1, x2, k)`
 
## Description
Calculates the covariance matrix for all columns in the input variable
`x` using the covariance kernel `k`. If two inputs `x1` and `x2` are
provided, the cross-covariance between `x1` and `x2` is calculated.
 
## Input
* `x`: dx-times-N matrix of input values where each column is a
  dx-dimensional input value.
* `x1` and `x2`: dx-times-N1 and dx-times-N2 matrices of input values for
  calculating the cross-covariance between the `x1`s and `x2`s.
* `k`: Function handle of the covariance function of the form `k(x1, x2)`
  (default: @k_se).
 
## Output
* `K`: N-times-N or N1-times-N2 (cross-)covariance matrix.
 
## Authors
2017-present -- Roland Hostettler
