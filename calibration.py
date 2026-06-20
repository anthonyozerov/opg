"""
Calibration curve and two related overprecision statistics.

A calibration curve asks: when measurements claim, say, 90% confidence, are they
right about 90% of the time? We cannot see each true value, but we can compare
every pair of measurements. If two calibrated measurements disagree, the size of
their disagreement (relative to their combined error bars) follows a known normal
distribution. Comparing the observed pattern to that ideal gives the curve.

This file provides three summaries of the same question -- are the error bars
calibrated, or too small (overprecision)? -- each with two ways to get a
confidence interval and a p-value:

  1. Calibration area     -- the gap between the calibration curve and the
                             perfectly-calibrated diagonal. Zero means perfect;
                             positive means overprecision.
  2. Within-1-sigma share -- the fraction of measurement pairs that agree to within
                             1 standard error. With calibrated error bars this is
                             about 68%; overprecision pushes it lower.
  3. (The Birge ratio itself lives in methods.py.)

The two routes to a confidence interval are jackknife (leave one measurement out)
and parametric (compare against simulated calibrated datasets). Run as a script,
this file draws the calibration curve for one example dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from preprocess import preprocess
from methods import parametric_null_sim, invert_ci, DEFAULT_C0_GRID

DATASET_NUM = 500

# The horizontal axis of the calibration curve: nominal confidence levels from
# 0 to 1. The observed coverage is measured at each one.
CONFIDENCES = np.linspace(0, 1, 99)

# The within-1-sigma statistic is one point on the curve: a z-score of 1, i.e. the
# confidence level 2*Phi(1) - 1 ~= 0.6827, the value expected if the error bars are
# perfectly calibrated (c = 1).
P_NULL = 2 * norm.cdf(1.0) - 1  # ~0.6827


def pairwise_z_stats(values, sigma):
    """For every pair of measurements, how big is their disagreement?

    Returns the absolute z-score of each pair -- the difference between the two
    measurements divided by their combined error bar -- plus the index arrays
    telling which two measurements each pair came from.
    """
    index_a, index_b = np.triu_indices(len(values), k=1)
    difference = values[index_a] - values[index_b]
    sigma_pair = np.sqrt(sigma[index_a]**2 + sigma[index_b]**2)
    return np.abs(difference / sigma_pair), index_a, index_b


def calibration_area_from_z(z_stats, confidences):
    """The signed area between the calibration curve and the ideal diagonal.

    For each nominal confidence level, the matching z-score threshold is found,
    and we measure what fraction of pairs fall under it (the observed coverage).
    The area is the integral of (nominal - observed): positive when observed
    coverage runs below nominal, which is the signature of overprecision.
    """
    thresholds = norm.ppf((1 + confidences) / 2)
    observed = np.mean(z_stats[:, None] <= thresholds[None, :], axis=0)
    area = np.trapezoid(confidences - observed, confidences)
    return area, observed


def calibration_and_ci(values, sigma, confidences=CONFIDENCES):
    """Calibration area with a jackknife 95% confidence interval and p-value.

    Returns (area, ci_low, ci_high, observed, p_value). The interval and p-value
    come from the jackknife: recompute the area with each measurement left out and
    measure how much it varies. The test is area = 0 (calibrated) vs. area > 0
    (overprecision).
    """
    z_stats, index_a, index_b = pairwise_z_stats(values, sigma)
    n = len(values)
    n_pairs = len(z_stats)

    thresholds = norm.ppf((1 + confidences) / 2)
    # below[k, j] is True if pair k's z-score is under threshold j.
    below = z_stats[:, None] <= thresholds[None, :]  # (n_pairs, n_conf)

    full_count = below.sum(axis=0)  # number of pairs under each threshold
    observed = full_count / n_pairs
    area = np.trapezoid(confidences - observed, confidences)

    # Jackknife: redo the area n times, each time dropping one measurement. Rather
    # than recompute from scratch, we subtract off the pairs that involve the
    # dropped measurement. Each measurement i appears in exactly (n - 1) pairs.
    removed_count = np.zeros((n, len(confidences)))
    np.add.at(removed_count, index_a, below)
    np.add.at(removed_count, index_b, below)

    leftout_n_pairs = n_pairs - (n - 1)
    leftout_observed = (full_count[None, :] - removed_count) / leftout_n_pairs
    leftout_areas = np.trapezoid(confidences[None, :] - leftout_observed,
                                 confidences, axis=1)

    # Standard jackknife pseudovalues -> estimate, standard error, interval, test.
    pseudo_areas = n * area - (n - 1) * leftout_areas
    pseudo_area = np.mean(pseudo_areas)
    pseudo_var = np.sum((pseudo_areas - pseudo_area)**2) / (n - 1)
    pseudo_se = np.sqrt(pseudo_var)
    standard_error_of_mean = pseudo_se / np.sqrt(n)

    ci_low = pseudo_area - 1.96 * standard_error_of_mean
    ci_high = pseudo_area + 1.96 * standard_error_of_mean

    # One-sided p-value: how far above 0 is the area, in standard errors?
    p_value = norm.cdf(-(pseudo_area - 0) / standard_error_of_mean)
    return pseudo_area, ci_low, ci_high, observed, p_value


def calibration_and_ci_boot(values, sigma, confidences=CONFIDENCES, B=2000, rng=None,
                            skip_ci=False):
    """Calibration area with a BOOTSTRAP 95% CI (an older cross-check method).

    Instead of the jackknife, this resamples whole measurements with replacement
    B times and recomputes the area each time. Used only by check_bootstrap.py.
    Returns (area, ci_low, ci_high, observed, p_value).
    """
    if rng is None:
        rng = np.random.default_rng()

    z_stats, index_a, index_b = pairwise_z_stats(values, sigma)
    n = len(values)
    total_pairs_boot = n * (n - 1) // 2

    thresholds = norm.ppf((1 + confidences) / 2)
    below = z_stats[:, None] <= thresholds[None, :]  # (n_pairs, n_conf)
    observed = below.mean(axis=0)
    area = np.trapezoid(confidences - observed, confidences)

    if skip_ci:
        return area, None, None, None, None

    # For each bootstrap resample, weight the original pairs by how many times each
    # of their two measurements was drawn. A measurement drawn against itself gives
    # a z-score of 0 (always under every threshold), counted here as degenerate.
    boot_areas = np.empty(B)
    for b in range(B):
        drawn = rng.integers(0, n, size=n)
        counts = np.bincount(drawn, minlength=n)
        pair_weights = counts[index_a] * counts[index_b]  # (n_pairs,)
        degenerate = int(np.sum(counts * (counts - 1) // 2))
        boot_observed = (pair_weights @ below + degenerate) / total_pairs_boot
        boot_areas[b] = float(np.trapezoid(confidences - boot_observed, confidences))

    # Basic bootstrap interval: reflect the bootstrap spread around the estimate.
    boot_q = np.quantile(boot_areas, [0.025, 0.975])
    ci_low = 2 * area - boot_q[1]
    ci_high = 2 * area - boot_q[0]

    # One-sided p-value: re-center the bootstrap areas on 0 (calibrated case) and ask
    # how often they reach the observed area.
    p_value = np.mean((boot_areas - np.mean(boot_areas)) >= area)

    return area, ci_low, ci_high, observed, p_value


# ---------------------------------------------------------------------------
# What each statistic equals for a given true overprecision c.
# Both functions are monotone in c, so a
# confidence interval found on the c scale maps cleanly onto the statistic scale.
# The areas use the same CONFIDENCES grid as the estimator so the comparison is
# consistent.
# ---------------------------------------------------------------------------

def area_of_c(c, confidences=CONFIDENCES):
    """The calibration area we expect if the true overprecision is c.

    Larger c (more overprecision) gives a larger area; c = 1 gives ~0.
    """
    z_alpha = norm.ppf((1 + confidences) / 2)
    observed_true = 2 * norm.cdf(z_alpha / c) - 1
    return float(np.trapezoid(confidences - observed_true, confidences))


def proportion_of_c(c):
    """The within-1-sigma share we expect if the true overprecision is c.

    Equals 2*Phi(1/c) - 1. Larger c gives a smaller share; c = 1 gives ~0.6827.
    """
    return float(2 * norm.cdf(1.0 / c) - 1)


# ---------------------------------------------------------------------------
# Within-1-sigma share: jackknife CI + one-sided test.
# This is just the calibration curve read at the
# single point z = 1, so it reuses the same leave-one-out idea as
# calibration_and_ci, specialized to one threshold.
# ---------------------------------------------------------------------------

def proportion_and_ci(values, sigma, scale="identity", coverage=0.95):
    """Within-1-sigma share with a jackknife 95% CI and p-value.

    The test is share = P_NULL (~0.6827, calibrated) vs. share < P_NULL
    (overprecision). `scale` is 'identity' (no transform) or 'logit' (a transform
    that can stabilize the variance); the final choice is settled empirically in
    check_jackknife.py.

    Returns (share, ci_low, ci_high, p_value).
    """
    z_stats, index_a, index_b = pairwise_z_stats(values, sigma)
    n = len(values)
    n_pairs = len(z_stats)

    below = (z_stats <= 1.0).astype(float)
    full_count = below.sum()
    share = full_count / n_pairs

    # Pairs containing measurement i that agree to within 1 sigma.
    removed = np.zeros(n)
    np.add.at(removed, index_a, below)
    np.add.at(removed, index_b, below)

    leftout_n_pairs = n_pairs - (n - 1)
    leftout_share = (full_count - removed) / leftout_n_pairs  # (n,)

    if scale == "identity":
        transform = lambda x: x
        inverse_transform = lambda t: t
    elif scale == "logit":
        clip = lambda x: np.clip(x, 1e-6, 1 - 1e-6)
        transform = lambda x: np.log(clip(x) / (1 - clip(x)))
        inverse_transform = lambda t: 1.0 / (1.0 + np.exp(-t))
    else:
        raise ValueError(f"unknown scale: {scale!r}")

    null_value = transform(P_NULL)
    pseudovalues = n * transform(share) - (n - 1) * transform(leftout_share)
    pseudo_mean = pseudovalues.mean()
    pseudo_var = np.sum((pseudovalues - pseudo_mean) ** 2) / (n - 1)
    standard_error_of_mean = np.sqrt(pseudo_var) / np.sqrt(n)

    z_for_ci = norm.ppf((1 + coverage) / 2)
    ci = np.sort(inverse_transform(np.array([
        pseudo_mean - z_for_ci * standard_error_of_mean,
        pseudo_mean + z_for_ci * standard_error_of_mean])))

    # No variation across leave-one-out (e.g. every pair on the same side of z = 1
    # at very small n) -> treat as no evidence of overprecision.
    z = (0.0 if standard_error_of_mean == 0
         else (pseudo_mean - null_value) / standard_error_of_mean)
    p_value = norm.cdf(z)  # lower tail: a small share is evidence of overprecision

    return float(share), float(ci[0]), float(ci[1]), float(p_value)


# ---------------------------------------------------------------------------
# Parametric (simulation-based) CI + test for the area and the share.
# Given the sorted |z| from each
# simulated calibrated dataset (parametric_null_sim), a draw's statistic at candidate
# c is the same calculation applied to {c * |z|}, i.e. counting |z| below
# (threshold / c). That lets one simulation serve every candidate c.
# ---------------------------------------------------------------------------

def _proportions_over_grid(abs_z_sorted, candidate_c_grid):
    """Within-1-sigma share of each simulated dataset at each candidate c.

    Returns a (B, C) array (B simulated datasets, C candidate c values).
    """
    B, n_pairs = abs_z_sorted.shape
    # At candidate c, within 1 sigma means the rescaled |z| is below 1, i.e. the
    # stored |z| is below 1/c. searchsorted counts those quickly in sorted data.
    thresholds = 1.0 / np.asarray(candidate_c_grid, dtype=float)
    out = np.empty((B, len(thresholds)))
    for b in range(B):
        out[b] = np.searchsorted(abs_z_sorted[b], thresholds, side="right")
    return out / n_pairs


def _areas_over_grid(abs_z_sorted, candidate_c_grid, confidences):
    """Calibration area of each simulated dataset at each candidate c.

    Returns a (B, C) array.
    """
    B, n_pairs = abs_z_sorted.shape
    candidate_c_grid = np.asarray(candidate_c_grid, dtype=float)
    z_alpha = norm.ppf((1 + confidences) / 2)  # (m,)
    C, m = len(candidate_c_grid), len(confidences)
    # At candidate c, observed coverage = fraction of stored |z| below z_alpha / c.
    thresholds = (z_alpha[None, :] / candidate_c_grid[:, None]).ravel()  # (C*m,)
    base = np.trapezoid(confidences, confidences)
    out = np.empty((B, C))
    for b in range(B):
        count = np.searchsorted(abs_z_sorted[b], thresholds, side="right").reshape(C, m)
        observed = count / n_pairs
        out[b] = base - np.trapezoid(observed, confidences, axis=1)
    return out


def _parametric_ci_test(stat_obs, stat_c1_draws, grid_draws, candidate_c_grid,
                        map_to_stat, coverage):
    """Shared engine: a p-value (test at c = 1) plus a CI mapped to the statistic.

    `stat_c1_draws`  : (B,) simulated values of the statistic at c = 1 (the full
                       set, so the 0.005 tail of the test is well resolved).
    `grid_draws`     : (B_sub, C) simulated values at each candidate c (used for
                       the interval's quantile bands).
    `map_to_stat`    : a function turning a c value into the statistic's value.

    Callers feed appropriately-signed inputs so that overprecision is always the
    UPPER tail of the statistic here.
    """
    # A safe Monte-Carlo p-value (Phipson & Smyth 2010): the (1 + count)/(B + 1)
    # form is guaranteed not to understate the tail, unlike the raw fraction.
    B = len(stat_c1_draws)
    p_value = float((1 + np.sum(stat_c1_draws >= stat_obs)) / (B + 1))

    tail = (1 - coverage) / 2
    q_low = np.quantile(grid_draws, tail, axis=0)
    q_high = np.quantile(grid_draws, 1 - tail, axis=0)
    c_lo, c_hi = invert_ci(candidate_c_grid, q_low, q_high, stat_obs)
    stat_lo, stat_hi = sorted([map_to_stat(c_lo), map_to_stat(c_hi)])
    return p_value, stat_lo, stat_hi


def _run_engine(sigma, B, rng, abs_z_sorted, ci_subsample):
    """Run the simulation engine if needed, and pick the CI subsample.

    Returns (full_set, subsample). Passing a precomputed abs_z_sorted lets several
    statistics share a single (expensive) engine run.
    """
    if abs_z_sorted is None:
        _, abs_z_sorted = parametric_null_sim(sigma, B, rng)
    B_full = abs_z_sorted.shape[0]
    subsample = abs_z_sorted if ci_subsample >= B_full else abs_z_sorted[:ci_subsample]
    return abs_z_sorted, subsample


def area_parametric(values, sigma, B=20000, rng=None, confidences=CONFIDENCES,
                    c0_grid=None, coverage=0.95, abs_z_sorted=None,
                    ci_subsample=8000):
    """Calibration area with a parametric (simulation-based) 95% CI and p-value.

    Compares the observed area against many simulated calibrated datasets. The p-value
    is the fraction of simulated areas at least as large as observed; the CI comes
    from inverting the simulated quantile bands across candidate c. Pass
    `abs_z_sorted` (from parametric_null_sim) to reuse one engine run.

    Returns (area, ci_low, ci_high, p_value).
    """
    if rng is None:
        rng = np.random.default_rng()
    if c0_grid is None:
        c0_grid = DEFAULT_C0_GRID

    z_stats, _, _ = pairwise_z_stats(values, sigma)
    area_obs, _ = calibration_area_from_z(z_stats, confidences)

    full, subsample = _run_engine(sigma, B, rng, abs_z_sorted, ci_subsample)
    areas_c1 = _areas_over_grid(full, np.array([1.0]), confidences)[:, 0]
    grid_draws = _areas_over_grid(subsample, c0_grid, confidences)

    p_value, ci_low, ci_high = _parametric_ci_test(
        area_obs, areas_c1, grid_draws, c0_grid,
        lambda c: area_of_c(c, confidences), coverage)
    return float(area_obs), ci_low, ci_high, p_value


def proportion_parametric(values, sigma, B=20000, rng=None, c0_grid=None,
                          coverage=0.95, abs_z_sorted=None, ci_subsample=8000):
    """Within-1-sigma share with a parametric 95% CI and p-value.

    Overprecision LOWERS the share, so the test is a lower-tail test. To reuse the
    shared upper-tail machinery in _parametric_ci_test we flip the sign of the
    share (and of its c-curve), run it, then flip the reported interval back.

    Returns (share, ci_low, ci_high, p_value).
    """
    if rng is None:
        rng = np.random.default_rng()
    if c0_grid is None:
        c0_grid = DEFAULT_C0_GRID

    z_stats, _, _ = pairwise_z_stats(values, sigma)
    share_obs = float(np.mean(z_stats <= 1.0))

    full, subsample = _run_engine(sigma, B, rng, abs_z_sorted, ci_subsample)
    shares_c1 = _proportions_over_grid(full, np.array([1.0]))[:, 0]
    grid_draws = _proportions_over_grid(subsample, c0_grid)

    p_value, neg_lo, neg_hi = _parametric_ci_test(
        -share_obs, -shares_c1, -grid_draws, c0_grid,
        lambda c: -proportion_of_c(c), coverage)
    ci_low, ci_high = -neg_hi, -neg_lo
    return share_obs, ci_low, ci_high, p_value


if __name__ == "__main__":
    # Load one example dataset, clean it, and draw its calibration curve.
    df = pd.read_csv(f"sim-G/dataset_{DATASET_NUM:04d}.csv")
    preprocess(df)

    use = df["use"]
    values = df["value_processed"][use].values
    sigma = df["error_processed"][use].values

    miscalibration, ci_low, ci_high, observed, _ = calibration_and_ci(values, sigma)

    print(f"Dataset {DATASET_NUM:04d}")
    print(f"Miscalibration area: {miscalibration:.4f}  95% CI: ({ci_low:.4f}, {ci_high:.4f})")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1,
            label="Perfect calibration")
    ax.plot(CONFIDENCES, observed, color='red', linewidth=2, label="Observed coverage")
    ax.fill_between(CONFIDENCES, CONFIDENCES, observed, alpha=0.2,
                    label=f"Area = {miscalibration:.4f}", color='grey')
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Observed coverage")
    ax.set_title(f"Calibration curve; dataset {DATASET_NUM:04d}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"calibration_{DATASET_NUM:04d}.png", dpi=150)
    plt.show()
    print("Saved calibration curve.")
