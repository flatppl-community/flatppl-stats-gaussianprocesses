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

Covariances include squared-exponential, Matérn 3/2, Matérn 5/2, and one-dimensional periodic kernels.

For $n$ training points, the observation law is $N(\mu,C)$ with
$C=K+\mathrm{diag}(\sigma_1^2+j,\ldots,\sigma_n^2+j)$ and $K$ of shape $n\times n$.
Prediction takes an $n\times m$ cross-covariance and returns a mean vector of length $m$
and an $m\times m$ covariance. Observation and prediction laws require positive-definite covariance.

The [regression example](examples/playground.flatppl) infers three covariance and noise parameters with proper priors. It integrates out the latent function.
