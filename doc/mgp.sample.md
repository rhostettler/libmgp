# Sample from a Gaussian process
## Usage
* `f = sample(x)`
* `f = sample(x, k)`
* `f = sample(x, m, k)`
 
## Description
Samples a set of function values `f` from a Gaussian process with mean
`m(x)` and covariance function `k(x1, x2)`.
 
## Input
* `x`: dx*N matrix of N input values.
* `m(x)`: Mean function (default: zero).
* `k(x1, x2)`: Covariance function.
 
## Output
* `f`: The sampled function values.

## Authors
2017-present -- Roland Hostettler
