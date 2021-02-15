# Power spectral density of (1D) squared exponential covariance function
## Usage
* `S = k_se_psd(w)`
* `[S, dS] = k_se_psd(w, ell, sigma2)`
 
## Description
Power spectral density of the squared exponential (SE) Gaussian process
covariance function. Also returns a struct of the PSD's gradient with
respect to the hyperparameters.
 
## Input
* `w`: Angular frequency in rad/s.
* `ell`: Length scale (default: 1).
* `sigma2`: Magnitude (default: 1).
 
## Output
* `S`: Power spectral density.
* `dS`: Gradient of the power spectral density with respect to `ell` and
  `sigma2`.
 
## Author
2017-present -- Roland Hostettler
