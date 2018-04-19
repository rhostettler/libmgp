Gaussian Process Library
========================
This is a simple Gaussian process (GP) library for Matlab. It provides a bunch of simple functions to do basic GP regression and classification, mostly aimed for research purposes. The library is by no means complete or stable and is most likely not suitable for generic purposes. If you are looking for a stable GP library, have a look at (for example) [GPStuff](http://research.cs.aalto.fi/pml/software/gpstuff/).


Naming Conventions
------------------
This library follows a couple of naming conventions. These are:

* `gp_XXX`: Generic function (e.g. `gp_calculate_covariance()` or `gp_predict()`),
* `gpk_XXX`: Covariance functions (e.g. `gpk_se()` or `gpk_matern()`),
* `gpc_XXX`: Functions related to GP classification (e.g. `gpc_train_ep()` or `gpc_predict_ep()`).

Functions not following this convention are most likely broken.


Covariance Functions
--------------------
Covariance kernels follow the general interface `Kxx = gpk_xxx(x1, x2, theta1, theta2, ...)`, where `theta1`, `theta2`, etc. denote the kernels parameters. If omitted, the kernel parameters will be set to a default value (`1` in most cases, see the Matlab help text for the respective covariance function). In addition to the standard formulation of the covariance function, there are also functions for the spectral density (generally of the form `Sw = gpk_xxx_psd(w, theta1, theta2, ...)`) and state-space representations (of the form `[A, B, C, Sw, Pinf] = gpk_xxx_ss(theta1, theta2, ...)`) for stationary covariance functions.

The following covariance functions are implemented:

* Squared exponential covariance function:
    * `gpk_se(x1, x2, ell, sigma2)`: Covariance function,
    * `gpk_se_psd(w, ell, sigma2)`: Power spectral density,
    * `gpk_se_ss(ell, sigma2, N)`: State-space representation.
* Matérn covariance function:
    * `gpk_matern(x1, x2, ell, sigma2, nu)`: Covariance function,
    * `gpk_matern_ss(ell, sigma2, nu)`: State-space representation.


State-Space Representations
---------------------------
One of the main purposes of the library is to provide tools for state-space representations of temporal GPs. Hence, there are a couple of functions that help to build this kind of models. These are:

* `gp_model_ss()`: A constructor for discrete-time state-space representations of GPs with separable covariance functions.


Examples
--------
There are a bunch of examples implemented (see the folder `examples`). These are:

* `example_gpc_toy.m`: Gaussian process classification example using the Laplace and Expectation Propagation approximations (re-implementation of the example in Section 3.7.2 of Rasmussen and Williams (2006)).
* `example_matern_ss.m`: Example of using the Matern state space formulation for a spatio-temporal GP regression problem.


TODO
----
[ ] Gather all gp_xxxx files
[ ] Merge duplicates
[ ] Allow for diagonal length scale matrix in gpk_se.
[ ] Fix minor bugs in gpk_matern.
[ ] Adopt the naming convention gpk_xx


Status
------
Here's an overview of the status of the different files:

*examples:*
* `example_gpc_toy.m`
* `example_matern_ss.m`
* `example_model_ss.m`
* `example_se_ss.m`

*src:*
* `dk_periodic.m`
* `gp_calculate_cross_covariance.m`
* `gp_calculate_log_posterior.m`
* `gpc_predict_ep.m`
* `gpc_train_laplace.m`
* `gpk_matern_ss.m`
* `gpk_se.m`
* `gpk_se_ss.m`
* *OK* `gp_model_ss.m`
* `gp_sample.m`
* `k_canonical_periodic_ss.m`
* `k_quasiperiodic_ss.m`
* `gp_calculate_covariance.m`
* `gp_calculate_loglikelihood_ss.m`
* `gp_calculate_logposterior_ss.m`
* `gpc_train_ep.m`
* `gpk_matern.m`
* `gp_k_quasiperiodic_ss.m`
* `gpk_se_psd.m`
* `gp_model_smc.m`
* `gp_predict.m`
* `k_bessel.m`
* `k_periodic.m`

