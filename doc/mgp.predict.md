# Gaussian process prediction
## Usage
* `[mp, Cp] = predict(xp, xt, yt)`
* `[mp, Cp] = predict(xp, xt, yt, R, m, k)`
 
## Description
Predicts the output of a Gaussian process f ~ GP(m(x), k(x, x') for the
test input `xp`, given the training inputs `xt` and (noisy) output 
measurements yt = f(xt) + r where r ~ N(0, R).
 
## Input
* `xp`: dx-times-Np matrix of Np test inputs.
* `xt`: dx-times-Nt matrix of Nt training inputs.
* `yt`: 1-times-Nt vector of Nt (noisy) training outputs.
* `R`: Measurement noise variance (default: 1).
* `m`: Mean function m = @(x) ... (default: 0).
* `k`: Covariance function k = @(x1, x2) ... (default: `k_se`).
 
## Output
  mp      Predicted mean
  Cp      Predicted covariance
 
## Authors
2018-present -- Roland Hostettler
