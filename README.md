# flatppl-stats-gaussianprocesses

[Function reference](docs/reference.md) · [Live example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-gaussianprocesses%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)

Gaussian process covariances, marginalized Gaussian regression, and conditional prediction for FlatPPL. Library version **0.1**.

```flatppl
flatppl_compat = "0.1"
gp = load_module("main.flatppl")
x = [-1.0, 0.0, 1.0]
distance = gp.pairwise_distance(x, x)
covariance = gp.squared_exponential(distance, 1.2, 0.8)
observations = gp.gaussian_regression(zeros(3), covariance, 0.2, 1e-8)
```

Covariances include squared-exponential, Matern 3/2, Matern 5/2, and one-dimensional periodic forms. Observation noise is a standard deviation. Numerical jitter is a separate variance. Conditional prediction fixes the hyperparameters.

The [regression example](examples/playground.flatppl) infers three covariance and noise parameters with proper priors. It integrates out the latent function.

See the reference for definitions, domains, numerical limits, and validation commands.
