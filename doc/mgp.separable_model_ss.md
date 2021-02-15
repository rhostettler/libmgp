# Discrete-time state-space model for separable spatio-temporal GPs
## Usage
* `[F, Q, C, P0] = separable_model_ss()`
* `[F, Q, C, m0, P0, dx] = separable_model_ss(k_t, k_u, ut, up, Ts, f0)`
 
## Description
Generic discrete-time state-space representation for temporal and spatio-
temporal Gaussian processes with separable covariance function. The model
is of the form
 
  x[n] = F x[n-1] + q[n]
  f[n] = C x[n]
 
where x[0] ~ N(m0, P0) and q[n] ~ N(0, Q).
 
Note that if both training (ut) and prediction (up) locations are given,
they are stacked such that the first 1, ..., dx*Nt states correspond to 
the states of the training locations (and their derivatives) and the 
states dx*Nt+1, ..., dx*Nt+dx*Np correspond to the test locations.
 
## Input
* `k_t`: Temporal covariance function (state-space representation; 
  default: `@k_matern_ss`).
* `k_u`: Spatial covariance function (default: none).
* `ut`: Inducing (training) points for the spatial variable. Only
  required if `k_u` is not empty.
* `up`: Prediction points for the spatial variable (default: []).
* `Ts`: Sampling time (default: 1).
* `f0`: Initial mean of the GP, that is, f(u, 0).
 
## Output
* `F`, `Q`, `C`, `m0`, `P0`: Discrete-time linear state-space model 
  parameters.
* `dx`: For temporal models, this is the state dimension and for spatio-
  temporal models, this is the dimension of the state for one spatial 
   inducing points.
 
## References
1. R. Hostettler, S. Sarkka, S. J. Godsill, "Rao-Blackwellized particle 
   MCMC for parameter estimation in spatio-temporal Gaussian processes,"
   in 27th IEEE International Workshop on Machine Learning for Signal 
   Processing (MLSP), Tokyo, Japan, September 2017.
 
## Authors
2018-present -- Roland Hostettler
