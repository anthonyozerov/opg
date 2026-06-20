"""
Core statistical tools shared by the analysis scripts.

The central quantity is the Birge ratio. Given many measurements of the same
quantity, each with its own claimed error bar, calibrated error bars make the
measurements scatter around their average by about as much as the bars predict,
giving a Birge ratio near 1. A Birge ratio above 1 means the measurements scatter
more than the bars predict, which is overprecision.

Throughout, `c` is the true population Birge ratio (the amount of overprecision,
which the data only let us estimate). See STATISTICS_PLAN.md for the full
statistical reasoning.
"""

import numpy as np
from scipy.stats import chi2, norm

# Candidate values of c used when building a confidence interval by test inversion
# (see invert_ci below). Runs from well below 1 (under-confidence) up past the
# largest overprecision we simulate.
CANDIDATE_C_GRID = np.linspace(0.2, 8.0, 200)
# Kept under the old name too, since other modules import DEFAULT_C0_GRID.
DEFAULT_C0_GRID = CANDIDATE_C_GRID


def birge_ratio(value, error):
    """Compute the weighted mean of some measurements and their Birge ratio.

    Returns (weighted_mean, birge) where `birge` is the Birge ratio itself (the
    scatter relative to the error bars), not an inflated error bar. A value near 1
    means the error bars match the scatter; well above 1 means overprecision.
    """
    n = len(value)
    weights = 1 / error**2
    weighted_mean = np.sum(value * weights) / np.sum(weights)
    birge = np.sqrt(np.sum((value - weighted_mean)**2 * weights) / (n - 1))
    return weighted_mean, birge


def birge_ratio_conf_p(birge, n, coverage=0.6827):
    """Exact confidence interval and p-value for the Birge ratio.

    When error bars are calibrated, the squared Birge ratio times its degrees of
    freedom follows a chi-squared distribution. That gives both:
      * a confidence interval for the true Birge ratio c, and
      * a one-sided p-value testing c = 1 (calibrated) against c > 1 (overprecise).
        A small p-value is evidence of overprecision.

    `coverage` is the confidence level (pass coverage=0.95 for a 95% CI; the
    default 0.6827 is the 1-sigma band).

    Returns (interval, p_value), where interval is a 2-element array.
    """
    tail_prob = (1 - coverage) / 2

    # Translate the observed Birge ratio into a chi-squared value.
    chi2_observed = birge**2 * (n - 1)

    # Invert the chi-squared distribution to get the confidence interval.
    chi2_interval = chi2.ppf(np.array([1 - tail_prob, tail_prob]), df=n - 1)
    birge_interval = np.sqrt(chi2_observed / chi2_interval)

    # One-sided p-value: probability of a Birge ratio this large if c were 1.
    p_value = chi2.sf(chi2_observed, df=n - 1)

    return birge_interval, p_value


# ---------------------------------------------------------------------------
# The simulation engine (STATISTICS_PLAN.md section 3)
# ---------------------------------------------------------------------------
# Several statistics have no closed form, so we estimate their behaviour by
# simulation: generate many datasets with calibrated error bars (c = 1) and see
# how the statistics scatter. We simulate only once and reuse it for every
# candidate c and every statistic:
#
#   * The Birge ratio scales linearly: a draw giving birge B at c = 1 gives c * B
#     at any other c.
#   * The pairwise z-scores z_ij = (x_i - x_j) / sqrt(sigma_i^2 + sigma_j^2) also
#     scale linearly: z at c = 1 becomes c * z. The other two statistics are just
#     counts of how many |z| fall below various thresholds, recovered at any c by
#     re-thresholding the same stored z-scores.
#
# So we store each simulated dataset's Birge ratio and sorted |z| values, and
# recover everything else by rescaling. One caveat: a single z has the same
# distribution regardless of the error bars, but the joint distribution of all the
# z's depends on the error-bar pattern (sigma), so always run the engine with the
# real sigma of the dataset being analysed.

def parametric_null_sim(sigma, B, rng):
    """Simulate B calibrated (c = 1) datasets with the given error bars sigma.

    Returns
    -------
    birge_draws : (B,) array
        The Birge ratio of each simulated dataset.
    abs_z_sorted : (B, n_pairs) float32 array
        For each simulated dataset, the absolute pairwise z-scores |z_ij|, sorted
        ascending. Sorting lets us later count how many fall below a threshold t
        with a binary search instead of a loop.
    """
    sigma = np.asarray(sigma, dtype=float)
    n = len(sigma)
    weights = 1.0 / sigma**2
    total_weight = weights.sum()

    # Every unordered pair (i, j) of measurements; sigma_pair is the combined
    # error bar used to standardize the difference between that pair.
    index_a, index_b = np.triu_indices(n, k=1)
    sigma_pair = np.sqrt(sigma[index_a]**2 + sigma[index_b]**2)
    n_pairs = len(index_a)

    birge_draws = np.empty(B)
    abs_z_sorted = np.empty((B, n_pairs), dtype=np.float32)

    # Work in chunks of simulated datasets so we never hold too much in memory.
    chunk = max(1, min(B, 4000))
    for start in range(0, B, chunk):
        m = min(chunk, B - start)
        # Generate m datasets of n measurements each. With calibrated error bars
        # and a true mean of 0, each measurement is normal noise * sigma.
        simulated_values = rng.standard_normal((m, n)) * sigma  # shape (m, n)
        weighted_mean = (simulated_values * weights).sum(axis=1) / total_weight
        residuals = simulated_values - weighted_mean[:, None]
        birge_draws[start:start + m] = np.sqrt(
            (residuals**2 * weights).sum(axis=1) / (n - 1))

        # Pairwise z-scores for each simulated dataset, then sort each row.
        z = (simulated_values[:, index_a] - simulated_values[:, index_b]) / sigma_pair
        np.abs(z, out=z)
        z.sort(axis=1)
        abs_z_sorted[start:start + m] = z.astype(np.float32)

    return birge_draws, abs_z_sorted


def _first_crossing(x, y, target):
    """Find the x where a curve y(x) first crosses a horizontal line `target`.

    Used to read a confidence-interval endpoint off a curve. We walk along the
    grid, find the first place where y - target changes sign, and interpolate a
    straight line between the two surrounding points to pin down the crossing.
    Returns nan if the curve never crosses `target`.
    """
    difference = np.asarray(y, dtype=float) - target
    sign = np.sign(difference)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) == 0:
        return np.nan
    i = crossings[0]
    y0, y1 = y[i], y[i + 1]
    if y1 == y0:
        return x[i]
    return x[i] + (x[i + 1] - x[i]) * (target - y0) / (y1 - y0)


def invert_ci(candidate_c_grid, q_low, q_high, observed):
    """Turn a family of simulated reference bands into a confidence interval for c.

    Test inversion: for each candidate true value c, simulation gives a band
    [q_low(c), q_high(c)] that the statistic falls in 95% of the time. The 95%
    confidence interval is every candidate c whose band contains the observed
    statistic. Because the band shifts steadily as c grows, that set of candidates
    is one continuous interval; its ends are where the observed value crosses the
    lower-quantile and upper-quantile curves.

    Works whether the statistic grows or shrinks with c.
    """
    c_end_1 = _first_crossing(candidate_c_grid, q_low, observed)
    c_end_2 = _first_crossing(candidate_c_grid, q_high, observed)

    # If a curve is never crossed, the interval is open on that side; fall back to
    # the edge of the grid so the CI is still reported (and visibly pinned to the
    # boundary) rather than silently disappearing.
    if np.isnan(c_end_1):
        c_end_1 = candidate_c_grid[0] if observed >= q_low[0] else candidate_c_grid[-1]
    if np.isnan(c_end_2):
        c_end_2 = candidate_c_grid[-1] if observed <= q_high[-1] else candidate_c_grid[0]

    return (c_end_1, c_end_2) if c_end_1 <= c_end_2 else (c_end_2, c_end_1)


# ---------------------------------------------------------------------------
# Birge-ratio jackknife (STATISTICS_PLAN.md section 4.1)
# ---------------------------------------------------------------------------
# The jackknife gets a confidence interval without a closed form: recompute the
# statistic many times, each time leaving one measurement out, and measure how
# much the answer varies. The Birge ratio's distribution is skewed, so a plain
# jackknife gives intervals that are slightly too narrow. The arithmetic can
# optionally run on a more symmetric transformed scale (e.g. logs) and transform
# the interval back. The transform used is chosen empirically in
# check_jackknife.py; the final choice is the identity (no transform).

def _scale_funcs(scale):
    """Return (transform, inverse_transform, null_value) for a chosen scale.

    `null_value` is the transformed location of the well-calibrated case c = 1,
    used as the reference point for the hypothesis test.
    """
    if scale == "log":
        return np.log, np.exp, 0.0
    if scale == "sq":
        return (lambda x: x**2), (lambda t: np.sqrt(np.clip(t, 0.0, None))), 1.0
    if scale == "identity":
        return (lambda x: x), (lambda t: t), 1.0
    raise ValueError(f"unknown scale: {scale!r}")


def birge_jackknife(value, error, scale="identity", coverage=0.95, alpha_test=0.005):
    """Jackknife confidence interval and one-sided test for the Birge ratio.

    Leaves out one measurement at a time to gauge how much the Birge ratio varies,
    builds a confidence interval from that variation, and tests c = 1 (calibrated)
    against c > 1 (overprecision).

    Returns (ci_low, ci_high, p_value) on the ordinary Birge-ratio scale.
    """
    value = np.asarray(value, dtype=float)
    error = np.asarray(error, dtype=float)
    n = len(value)

    _, birge_full = birge_ratio(value, error)

    # Recompute the Birge ratio n times, each time dropping one measurement.
    birge_leave_one_out = np.empty(n)
    keep_all = np.ones(n, dtype=bool)
    for i in range(n):
        keep = keep_all.copy()
        keep[i] = False
        _, birge_leave_one_out[i] = birge_ratio(value[keep], error[keep])

    transform, inverse_transform, null_value = _scale_funcs(scale)
    theta_full = transform(birge_full)
    theta_leave_one_out = transform(birge_leave_one_out)

    # Pseudovalues are the standard jackknife building block: each one estimates
    # the statistic with the influence of a single point amplified. Their average
    # and spread give the estimate and its standard error.
    pseudovalues = n * theta_full - (n - 1) * theta_leave_one_out
    pseudo_mean = pseudovalues.mean()
    pseudo_var = np.sum((pseudovalues - pseudo_mean)**2) / (n - 1)
    standard_error = np.sqrt(pseudo_var) / np.sqrt(n)

    z_for_ci = norm.ppf((1 + coverage) / 2)
    ci_transformed = np.array([pseudo_mean - z_for_ci * standard_error,
                               pseudo_mean + z_for_ci * standard_error])
    ci = np.sort(inverse_transform(ci_transformed))

    # If every leave-one-out gives the same answer there is no variation to
    # measure; treat that as no evidence of overprecision (p = 0.5).
    z = 0.0 if standard_error == 0 else (pseudo_mean - null_value) / standard_error
    p_value = norm.sf(z)

    return float(ci[0]), float(ci[1]), float(p_value)
