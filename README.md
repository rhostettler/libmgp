## Gaussian Process Library
This is a simple Gaussian process (GP) library for Matlab. It provides a 
collection of functions to do basic GP regression and classification, 
mostly aimed for research purposes. The library is by no means complete or 
stable and is most likely not suitable for generic purposes. If you are 
looking for a stable, general-purpose GP library, have a look at (for 
example) [GPStuff](http://research.cs.aalto.fi/pml/software/gpstuff/) or 
similar.


## Usage
the library is organized as a Matlab namespace. To use it in your code, add
the `src` path to your Matlab path (`addpath ...`) and use the functions 
using `mgp.XXX`, for example, `mgp.sample()`. You may also import the whole
library, but this may cause naming conflicts.


## Covariance functions
Covariance functions follow the general interface
`Kxx = k_xxx(x1, x2, theta1, theta2, ...)`, where `theta1`, `theta2`, etc. 
are the kernels parameters. If omitted, the parameters will be set to a 
default value (`1` in most cases, see the Matlab help text for the
respective covariance function). In addition to the standard formulation of
the covariance function, there are also functions for the spectral density
(generally of the form `Sww = k_xxx_psd(w, theta1, theta2, ...)`) and 
state-space representations (of the form 
`[A, B, C, Sw, Pinf] = k_xxx_ss(theta1, theta2, ...)`) for stationary 
covariance functions.


## State-space Representations
One of the main purposes of the library is to provide tools for state-space
representations of temporal GPs. Hence, there are a couple of functions 
that help to build this kind of models:

* `separable_model_ss()`: A constructor for discrete-time state-space 
  representations of GPs with separable covariance functions,
* `model_smc()`: A constructor for discrete-time state-space 
  representations with arbitrary likelihoods that returns a model `struct` 
  suitable for the inference methods in [libsmc](https://github.com/rhostettler/libsmc).


## Examples
There are a bunch of examples implemented (see the folder `examples`):

* `example_gpc_toy.m`: Gaussian process classification example using the 
  Laplace and Expectation Propagation approximations (re-implementation of 
  the example in Section 3.7.2 of Rasmussen and Williams (2006)),
* `example_se_ss.m`: Example of using Taylor series approximation-based 
   state-space formulation for a squared exponential covariance function in
   a timeseries example,
* `example_matern_ss.m`: Example of using the Matern state-space 
  formulation for a separable spatio-temporal regression problem.