# State-space representation of the SE covariance function (Taylor app.)
## Usage
* `[A, B, C, Sw, Pinf] = k_se_ss()`
* `[A, B, C, Sw, Pinf] = k_se_ss(ell, sigma2, J)`
 
## Description
Calculates the linear state-space approximation of a temporal Gaussian 
process (GP) with squared exponential (SE) covariance function. In 
particular, if
 
  f(t) ~ GP(m(t), k(t, t')),
 
where
 
  k(t, t') = sigma2*exp(-(t-t')^2/(2*ell^2))
 
is the SE covariance function, then it calculates the matrices A, B, C
and spectral density Sw such that
 
  dx = A*x*dt + B*dw(t)
  f = C*x
 
where Sw is the spectral density of w(t). Furthermore, Pinf is the
stationary covariance and solves the continuous Lyapunov equation
 
  A'*Pinf + Pinv*A + B*Sw*B' = 0.
 
The approximation is based on the Taylor series expansion of the SE
spectral density, see, for example [1].
 
## Input
* `ell`: Length scale (default: 1).
* `sigma2`: GP covariance magnitude (standard deviation; default: 1).
* `J`: Approximation order (must be even; default: 6).
 
## Output
* `A`, `B`, `C`, `Sw`, `Pinf`: Linear state-space system equivalent as 
  described above.
 
## References
1. J. Hartikainen and S. Sarkka, "Kalman filtering and smoothing 
   solutions to temporal Gaussian process regression models," in IEEE 
   International Workshop on Machine Learning for Signal Processing 
   (MLSP), pp. 379?384, August 2010.
 
## Author
2017-present -- Roland Hostettler
