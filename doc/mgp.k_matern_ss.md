# State-space representation of the Matern covariance function
## Usage
* `[A, B, C, Sw, Pinf] = gpk_matern_ss(l, sigma2, nu)`
 
## Description
Calculates the linear state-space representation of a one-dimensional 
Gaussian process with Matern covariance function. If
 
  f ~ GP(0, k(t, t')),
 
with the covariance function k(t, t') given by
                                 _                   _            _                   _                                              
                      2^(1-nu)  | (2*nu)^(1/2)*|t-t'| |^nu       | (2*nu)^(1/2)*|t-t'| |
  k(t, t') = sigma2 * --------- | ------------------- |    * K_nu| ------------------- |,
                      Gamma(nu) |_        l          _|          |_        l          _|
 
this can be written as a state-space system of the form
 
  dx = A*x*dt + B*dw(t),
  f = C*x,
 
where Sw is the spectral density of w(t). Furthermore, Pinf is the 
stationary covariance and solves the continuous Ricatti equation
 
  A'*Pinf + Pinv*A + B*Sw*B' = 0.
 
The conversion is exact for positive integer values of p = nu - 0.5, 
see [1]. For non-integer values of p, the approximation in [2] is used.
 
## Input
* `ell`, `sigma2`, and `nu`: Parameters of the covariance function.
 
## Output
* `A`, `B`, `C`, `Sw`, and `Pinf`: Parameters of the linear state-space 
  equivalent as described above.
 
## See Also
* `k_se_ss`.
 
## References
1. J. Hartikainen and S. Särrkkä, "Kalman filtering and smoothing 
   solutions to temporal Gaussian process regression models," in IEEE 
   International Workshop on Machine Learning for Signal Processing 
   (MLSP), pp. 379-384, August 2010.
 
2. T. Karvonen and S. Särkkä, "Approximate state-space Gaussian 
   processes via spectral transformation," in 26th IEEE International 
   Workshop on Machine Learning for Signal Processing (MLSP), 
   September 2016.
 
## Authors
2017-present -- Roland Hostettler
