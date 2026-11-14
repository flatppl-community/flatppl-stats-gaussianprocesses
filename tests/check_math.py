"""Check GP covariances and conditioning using scalar Gaussian elimination."""

import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_POINTS = [
    [math.log(1.2), math.log(0.8), math.log(0.25), math.log(2.5), 0.4],
    [math.log(0.7), math.log(1.4), math.log(0.4), math.log(1.8), 0.0],
    [math.log(1.6), math.log(0.5), math.log(0.15), math.log(3.1), 1.0],
]


def simpson(function, lower, upper, count=8192):
    step = (upper - lower) / count
    total = function(lower) + function(upper)
    total += 4 * sum(function(lower + i * step) for i in range(1, count, 2))
    total += 2 * sum(function(lower + i * step) for i in range(2, count, 2))
    return total * step / 3


def filter_covariance(name, distance, amplitude, length_scale):
    """Covariance as an impulse-response autocorrelation, not a kernel formula."""
    if name == "squared_exponential":
        impulse = lambda t: math.exp(-((t / length_scale) ** 2))
        integral = simpson(
            lambda t: impulse(t) * impulse(t + distance),
            -8 * length_scale,
            8 * length_scale,
        )
        variance = math.sqrt(math.pi / 2) * length_scale
    else:
        order = 2 if name == "matern32" else 3
        rate = math.sqrt(2 * order - 1) / length_scale
        impulse = lambda t: t ** (order - 1) * math.exp(-rate * t)
        integral = simpson(lambda t: impulse(t) * impulse(t + distance), 0, 24 / rate)
        variance = math.factorial(2 * order - 2) / (2 * rate) ** (2 * order - 1)
    return amplitude**2 * integral / variance


def covariance(name, distance, amplitude, length_scale, period=1.0):
    if name == "squared_exponential":
        return amplitude**2 * math.exp(-(distance**2) / (2 * length_scale**2))
    if name == "periodic":
        return amplitude**2 * math.exp(
            -(1 - math.cos(2 * math.pi * distance / period)) / length_scale**2
        )
    r = math.sqrt(3 if name == "matern32" else 5) * distance / length_scale
    polynomial = 1 + r if name == "matern32" else 1 + r + r * r / 3
    return amplitude**2 * polynomial * math.exp(-r)


def normal_log(value, mean, variance):
    return -0.5 * (math.log(2 * math.pi * variance) + (value - mean) ** 2 / variance)


def gaussian_log(values, means, cov):
    """Factor p(y) as successive scalar conditional densities (no inverse)."""
    residual = [value - mean for value, mean in zip(values, means, strict=True)]
    matrix = [row[:] for row in cov]
    score = 0.0
    for i in range(len(values)):
        variance = matrix[i][i]
        score += normal_log(residual[i], 0, variance)
        for j in range(i + 1, len(values)):
            coefficient = matrix[j][i] / variance
            residual[j] -= coefficient * residual[i]
            for k in range(i + 1, len(values)):
                matrix[j][k] -= coefficient * matrix[i][k]
    return score


def conditional_two(observed, mean_train, training, mean_test, cross, test):
    """Condition first on y1, then on the residual y2 (scalar rank-one updates)."""
    residual_first = observed[0] - mean_train[0]
    variance_second = training[1][1] - training[1][0] ** 2 / training[0][0]
    residual_second = (
        observed[1] - mean_train[1] - training[1][0] / training[0][0] * residual_first
    )
    remaining_cross = [
        cross[1][j] - training[1][0] / training[0][0] * cross[0][j]
        for j in range(len(mean_test))
    ]
    mean = [
        mean_test[j]
        + cross[0][j] / training[0][0] * residual_first
        + remaining_cross[j] / variance_second * residual_second
        for j in range(len(mean_test))
    ]
    cov = [
        [
            test[j][k]
            - cross[0][j] * cross[0][k] / training[0][0]
            - remaining_cross[j] * remaining_cross[k] / variance_second
            for k in range(len(mean_test))
        ]
        for j in range(len(mean_test))
    ]
    return mean, cov


def runtime_values(x):
    amplitude, length_scale, noise_sd, period = map(math.exp, x[:4])
    kernel = lambda d: covariance("squared_exponential", d, amplitude, length_scale)
    train, test = [0.0, 1.0], [x[4], 1.5]
    training = [
        [
            kernel(abs(a - b)) + (noise_sd**2 + 0.0001 if i == j else 0)
            for j, b in enumerate(train)
        ]
        for i, a in enumerate(train)
    ]
    cross = [[kernel(abs(a - b)) for b in test] for a in train]
    prior = [[kernel(abs(a - b)) for b in test] for a in test]
    mean, cov = conditional_two(
        [0.7, -0.2], [0.1, -0.1], training, [0.2, 0.3], cross, prior
    )
    latent = [
        [cov[i][j] + (0.0001 if i == j else 0) for j in range(2)] for i in range(2)
    ]
    noisy = [
        [latent[i][j] + (noise_sd**2 if i == j else 0) for j in range(2)]
        for i in range(2)
    ]
    names = ["squared_exponential", "matern32", "matern52", "periodic"]
    return [
        *[amplitude**2] * 4,
        *[covariance(name, 0.7, amplitude, length_scale, period) for name in names],
        *[amplitude**2] * 4,
        gaussian_log([0.7, -0.2], [0.1, -0.1], training),
        *mean,
        cov[0][0],
        cov[0][1],
        cov[1][1],
        gaussian_log([0.4, -0.1], mean, latent),
        gaussian_log([0.4, -0.1], mean, noisy),
        covariance("matern32", abs(x[4]), amplitude, length_scale),
        covariance("matern52", abs(x[4]), amplitude, length_scale),
    ]


def posterior(point):
    amplitude, length_scale, noise_sd = map(math.exp, point)
    train = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]
    cov = [
        [
            covariance("squared_exponential", abs(a - b), amplitude, length_scale)
            + (noise_sd**2 + 1e-8 if i == j else 0)
            for j, b in enumerate(train)
        ]
        for i, a in enumerate(train)
    ]
    return gaussian_log([-0.8, -0.7, -0.2, 0.4, 0.8, 0.7], [0.0] * 6, cov) + sum(
        normal_log(value, mean, 0.25)
        for value, mean in zip(point, [0, 0, math.log(0.2)], strict=True)
    )


def decode(value):
    if isinstance(value, dict):
        if "data" in value:
            data = [decode(x) for x in value["data"].values()]
            return data[0] if value.get("shape") == [] else data
        return {key: decode(x) for key, x in value.items()}
    if isinstance(value, list):
        return [decode(x) for x in value]
    return float(value) if isinstance(value, str) else value


def evaluate(request):
    result = subprocess.run(
        ["node", str(ROOT / "tests/evaluate.cjs")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return decode(json.loads(result.stdout))


def close(actual, expected, tolerance=3e-10):
    assert math.isfinite(actual) and math.isclose(
        actual, expected, rel_tol=tolerance, abs_tol=tolerance
    ), (actual, expected)


def main():
    expressions, expected = [], []
    for name in ["squared_exponential", "matern32", "matern52", "periodic"]:
        for distance in [0, 1e-8, 0.7, 4.0]:
            args = f"{distance},1.3,0.8" + (",2.5" if name == "periodic" else "")
            expressions.append(f"a.{name}({args})")
            expected.append(covariance(name, distance, 1.3, 0.8, 2.5))
    # Exact analytic values pin diagonal amplitude, period and length-scale conventions.
    expressions += [
        "a.periodic(2.5,1.3,0.8,2.5)",
        "a.squared_exponential(0.8,1.3,0.8)",
        "a.observation_covariance(rowstack([[2,0.3],[0.3,2]]),[0.2,0.4],0.001)[2,2]",
    ]
    expected += [1.3**2, 1.3**2 / math.sqrt(math.e), 2 + 0.4**2 + 0.001]
    # A Gaussian smoothing filter and causal polynomial-exponential filters
    # independently recover SE and both Matern covariances by quadrature.
    for name in ["squared_exponential", "matern32", "matern52"]:
        expressions.append(f"a.{name}(0.7,1.3,0.8)")
        expected.append(filter_covariance(name, 0.7, 1.3, 0.8))
    for actual, reference in zip(evaluate(expressions), expected, strict=True):
        close(actual, reference)

    # Rectangular cross-covariance fixes the orientation and one solve per column.
    observed, train_mean = [0.8, -0.1], [0, 0.2]
    training = [[2, 0.3], [0.3, 1]]
    cross = [[0.4, 0.2, 0.1], [-0.1, 0.3, 0.2]]
    test = [[1, 0.2, 0.1], [0.2, 1, 0.3], [0.1, 0.3, 1]]
    mean, cov = conditional_two(
        observed, train_mean, training, [0.1, -0.2, 0.3], cross, test
    )
    source = 'flatppl_compat="0.1"\na=load_module("../main.flatppl")\n'
    source += "c=a.conditional_moments([0.8,-0.1],[0,0.2],rowstack([[2,0.3],[0.3,1]]),[0.1,-0.2,0.3],rowstack([[0.4,0.2,0.1],[-0.1,0.3,0.2]]),rowstack([[1,0.2,0.1],[0.2,1,0.3],[0.1,0.3,1]]))"
    actual = evaluate({"source": source, "bindings": ["c"]})[0]
    for value, reference in zip(actual["mean"], mean, strict=True):
        close(value, reference)
    for value, reference in zip(
        actual["covariance"], [value for row in cov for value in row], strict=True
    ):
        close(value, reference)

    # A noiseless observation fixes duplicate latent predictions exactly; moments
    # exist even though no positive-definite predictive law exists without noise.
    singular = evaluate(
        [
            "a.conditional_moments([0.7],[0],rowstack([[1]]),[0,0],rowstack([[1,1]]),rowstack([[1,1],[1,1]]))"
        ]
    )[0]
    for value in singular["mean"]:
        close(value, 0.7)
    for value in singular["covariance"]:
        close(value, 0)

    fixture = (ROOT / "tests/runtime.flatppl").read_text()
    prefix, output_vector = fixture.split("outputs = [\n")
    # Probe the scalar outputs separately: the JS materialiser cannot currently
    # assemble a fixed value vector containing logdensityof(MvNormal(...), ...).
    output_expressions = [
        line.strip().removesuffix(",") for line in output_vector.splitlines()[:-1]
    ]
    output_names = [f"value{i}" for i in range(len(output_expressions))]
    scalar_fixture = prefix + "\n".join(
        f"{name}={expression}"
        for name, expression in zip(output_names, output_expressions, strict=True)
    )
    for point in RUNTIME_POINTS:
        source = scalar_fixture.replace(
            "x = elementof(cartpow(reals, 5))", f"x = {json.dumps(point)}"
        )
        values = evaluate({"source": source, "bindings": output_names})
        for actual, reference in zip(values, runtime_values(point), strict=True):
            close(actual, reference)

    points = [
        [0, 0, math.log(0.2)],
        [0.2, -0.2, math.log(0.3)],
        [-0.3, 0.4, math.log(0.15)],
    ]
    source = 'flatppl_compat="0.1"\ne=load_module("../examples/playground.flatppl")\n'
    source += "\n".join(
        f"p{i}=logdensityof(e.Pi_post,record(log_amplitude={p[0]},log_length_scale={p[1]},log_noise_sd={p[2]}))"
        for i, p in enumerate(points)
    )
    scores = evaluate({"source": source, "bindings": ["p0", "p1", "p2"]})
    for score, point in zip(scores, points, strict=True):
        close(score - scores[0], posterior(point) - posterior(points[0]))
    print(
        json.dumps(
            {
                "kernel_checks": len(expressions),
                "runtime_points": len(RUNTIME_POINTS),
                "runtime_outputs": len(runtime_values(RUNTIME_POINTS[0])),
                "posterior_points": len(points),
            }
        )
    )


if __name__ == "__main__":
    main()
