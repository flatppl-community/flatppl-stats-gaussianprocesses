# flatppl-stats-gaussianprocesses

[Function reference](docs/reference.md)

Gaussian process covariances, marginalized Gaussian regression and conditional
prediction in FlatPPL. Library version **0.1**.

```flatppl
flatppl_compat = "0.1"
gp = load_module("main.flatppl")
x = [-1.0, 0.0, 1.0]
distance = gp.pairwise_distance(x, x)
covariance = gp.squared_exponential(distance, 1.2, 0.8)
observations = gp.gaussian_regression(zeros(3), covariance, 0.2, 1e-8)
```

The [regression example](examples/playground.flatppl) infers three covariance and
noise parameters with proper priors and an explicit prior, kernel, likelihood
and posterior. The latent function is integrated out. Conditional prediction
keeps hyperparameters fixed; averaging over their posterior is a separate step.

Squared-exponential, Matern 3/2, Matern 5/2 and one-dimensional periodic
covariances are included. Observation noise is a standard deviation; numerical
jitter is a separate variance. Every call supplies both explicitly.

Run the independent scalar Gaussian checks with Python 3.10+ and Node 24+:

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js python tests/check_math.py
```

[The runtime fixture](tests/runtime.flatppl) and the independent `runtime_values`
function in [the checker](tests/check_math.py) support value and gradient checks,
including coincident coordinates.
