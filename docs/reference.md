# Gaussian process reference

Library version **0.1** · [Package overview](../README.md)

Load `main.flatppl` with `load_module`. All arguments are explicit; the library
chooses no prior, mean function, observation noise or numerical jitter.

| Function | Purpose |
| --- | --- |
| [`pairwise_distance`](#pairwise_distance) | Distances between scalar coordinates |
| [`squared_exponential`](#squared_exponential) | Squared-exponential covariance |
| [`matern32`](#matern32) | Matern 3/2 covariance |
| [`matern52`](#matern52) | Matern 5/2 covariance |
| [`periodic`](#periodic) | One-dimensional periodic covariance |
| [`observation_covariance`](#observation_covariance) | Add observation variance and explicit jitter |
| [`gaussian_regression`](#gaussian_regression) | Marginal Gaussian observation law |
| [`conditional_moments`](#conditional_moments) | Joint conditional mean and covariance |
| [`predictive_law`](#predictive_law) | Latent or noisy joint prediction law |

## Covariance functions

### `pairwise_distance`

`pairwise_distance(x, z)` takes nonempty vectors of scalar coordinates, of lengths
$n$ and $m$, and returns an $n\times m$ matrix with entries $|x_i-z_j|$. This
convenience function is for one-dimensional inputs. Coordinates must use one
consistent input unit $U$.

The covariance functions act elementwise on a scalar or array argument. A
covariance matrix must be a rank-two array such as `rowstack([[1, 0], [0, 1]])`,
rather than a vector of vectors. Their scalar parameters are:

| Parameter | Domain | Units |
| --- | --- | --- |
| `amplitude` $a$ | $a>0$ | response units $Y$ |
| SE/Matern `length_scale` $\ell$ | $\ell>0$ | input units $U$ |
| periodic `length_scale` $\ell_p$ | $\ell_p>0$ | dimensionless |
| `period` $p$ | $p>0$ | input units $U$ |

Each result has covariance units $Y^2$. Distances are finite and nonnegative.
The SE and Matern functions also accept Euclidean distances computed externally
between points in higher-dimensional Euclidean space, with consistently scaled
coordinates. Arbitrary distance matrices need not define valid covariances.

### `squared_exponential`

`squared_exponential(distance, amplitude, length_scale)` gives, for distance $d$:

$$k(d)=a^2\exp\left(-\frac{d^2}{2\ell^2}\right).$$

This is also called the exponentiated-quadratic covariance.

### `matern32`

`matern32(distance, amplitude, length_scale)` uses $r=\sqrt{3}d/\ell$:

$$k(d)=a^2(1+r)e^{-r}.$$

### `matern52`

`matern52(distance, amplitude, length_scale)` uses $r=\sqrt{5}d/\ell$:

$$k(d)=a^2\left(1+r+\frac{r^2}{3}\right)e^{-r}.$$

### `periodic`

`periodic(separation, amplitude, length_scale, period)` uses a signed or absolute
**one-dimensional separation** $h=x-z$:

$$k(h)=a^2\exp\left(-\frac{2\sin^2(\pi h/p)}{\ell_p^2}\right).$$

This is periodic in the original input coordinate, with period $p$. Applying
this expression to arbitrary radial distances in several dimensions does not
in general define a positive semidefinite covariance.

All four kernels satisfy $k(0)=a^2$ and have zero first derivative with respect
to separation at coincident points. Their amplitude derivative there is $2a$;
their length-scale derivative there is zero. The periodic kernel also has zero
period derivative at coincidence.

## Marginalized regression

### `observation_covariance`

`observation_covariance(covariance, noise_sd, jitter)` takes a symmetric latent
covariance $K$ of shape $n\times n$, a scalar observation standard deviation
or a vector of $n$ standard deviations, and a scalar numerical variance $j$:

$$C=K+\mathrm{diag}(\sigma_1^2+j,\ldots,\sigma_n^2+j).$$

`noise_sd` is nonnegative and has units $Y$. `jitter` is nonnegative and has
units $Y^2$. Scalar noise is repeated on the diagonal. Jitter is numerical
regularization, but adding it changes the supplied Gaussian law; it is never
silently chosen or estimated as observation noise. Supply zero when no jitter
is wanted. In particular, neither noise nor jitter is added to cross-covariances.

### `gaussian_regression`

`gaussian_regression(mean, covariance, noise_sd, jitter)` returns
`MvNormal(mean, observation_covariance(covariance, noise_sd, jitter))`.
The `mean` is a vector of length $n>0$ in units $Y$, and $C$ must be positive
definite. This is the normalized observation law obtained by integrating
$f\sim N(m,K)$ out of the Gaussian observation model. No latent function
values or priors are introduced by the library.

## Conditional prediction

### `conditional_moments`

`conditional_moments(observed, mean_train, covariance_train, mean_test, cross_covariance, covariance_test)`
returns a record with fields `mean` and `covariance`. This is the **joint latent
conditional distribution at fixed hyperparameters**, with arguments:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `observed` $y$ | length $n$ | observed training responses |
| `mean_train` $m$ | length $n$ | prior mean at training inputs |
| `covariance_train` $C$ | $n\times n$ | training covariance including noise and jitter |
| `mean_test` $m_*$ | length $m$ | prior mean at prediction inputs |
| `cross_covariance` $B$ | $n\times m$ | latent training-to-prediction covariance |
| `covariance_test` $K_*$ | $m\times m$ | latent prior prediction covariance |

Both $n$ and $m$ are positive. Means and responses have units $Y$; all
covariances have units $Y^2$. The supplied blocks must describe a consistent
joint Gaussian model, and $C$ must be positive definite. The result is

$$m_c=m_*+B^TC^{-1}(y-m),\qquad S_c=K_*-B^TC^{-1}B.$$

The implementation factors $C=LL^T$ once and uses triangular solves:
$V=L^{-1}B$, $m_c=m_*+V^TL^{-1}(y-m)$ and $S_c=K_*-V^TV$.
One matrix solve handles all columns of $B$. No matrix inverse is constructed.
The result retains dependence between prediction locations.

### `predictive_law`

`predictive_law(moments, noise_sd, jitter)` returns the normalized joint law

$$N\left(m_c,S_c+\mathrm{diag}(\sigma_{*,1}^2+j_*,\ldots,\sigma_{*,m}^2+j_*)\right).$$

Use `noise_sd = 0` for the latent function, or supply the prediction observation
standard deviations for future noisy responses. Prediction jitter is a separate,
explicit variance, with the same convention as training jitter. This law is
conditional on the supplied hyperparameters; posterior prediction additionally
averages over their posterior.

The final covariance must be positive definite for `MvNormal`. Noiseless duplicate
prediction locations can give valid singular conditional moments without defining
a nonsingular `predictive_law`. Numerical cancellation can also affect a small
conditional variance. The library neither clips variances nor inserts jitter.

```flatppl
flatppl_compat = "0.1"
gp = load_module("../main.flatppl")
train = [0.0, 1.0]
test = [0.4, 1.5]
training = gp.squared_exponential(gp.pairwise_distance(train, train), 1.2, 0.8)
cross_covariance = gp.squared_exponential(gp.pairwise_distance(train, test), 1.2, 0.8)
prediction = gp.squared_exponential(gp.pairwise_distance(test, test), 1.2, 0.8)
moments = gp.conditional_moments([0.7, -0.2], zeros(2),
    gp.observation_covariance(training, 0.25, 1e-8), zeros(2), cross_covariance, prediction)
latent = gp.predictive_law(moments, 0.0, 1e-8)
future_observations = gp.predictive_law(moments, 0.25, 1e-8)
```

The formulas follow Stan's [Gaussian process guide](https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html),
especially “GP with a normal outcome” and “Analytical form of joint predictive
inference”, and the [covariance function definitions](https://mc-stan.org/docs/functions-reference/matrix_operations.html#gaussian-process-covariance-functions)
in its function reference. The tests independently recover SE and Matern
covariances by integrating smoothing-filter autocorrelations, factor Gaussian
densities into scalar conditional densities and obtain predictive moments by
successive scalar conditioning.

## Examples and validation

The [live regression example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-gaussianprocesses%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post) infers three covariance and
noise parameters with proper priors and an explicit prior, kernel, likelihood
and posterior. The latent function is integrated out. Conditional prediction
keeps hyperparameters fixed; averaging over their posterior is a separate step.

Run the independent scalar Gaussian checks with Python 3.10+ and Node 24+:

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js python tests/check_math.py
```

[The runtime fixture](../tests/runtime.flatppl) and the independent `runtime_values`
function in [the checker](../tests/check_math.py) support value and gradient checks,
including coincident coordinates.
