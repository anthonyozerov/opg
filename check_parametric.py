"""
Validate the three PARAMETRIC (simulation-based) overprecision tests + intervals:
  * Birge ratio              -- exact chi-squared formula (methods.birge_ratio_conf_p)
  * calibration area         -- simulation engine (calibration.area_parametric)
  * within-1-sigma share     -- simulation engine (calibration.proportion_parametric)

It also runs a cross-check: the simulation engine should reproduce the exact
chi-squared Birge-ratio test. If the "birge (engine)" row in the table tracks the
"birge (chi2)" row, that confirms the engine is trustworthy on a case where we
already know the right answer.

The parametric tests are exact apart from simulation noise (controlled by B). One
engine run of honest (c = 1) datasets, conditional on the fixed error-bar pattern,
serves all three statistics: its Birge ratios drive the cross-check, and its
sorted |z| values drive the area and the share by rescaling.
(STATISTICS_PLAN.md sections 3, 4.)
"""

import numpy as np

from methods import (
    birge_ratio, birge_ratio_conf_p, parametric_null_sim,
    invert_ci, DEFAULT_C0_GRID,
)
from calibration import (
    _areas_over_grid, _proportions_over_grid, area_of_c, proportion_of_c,
    calibration_area_from_z, pairwise_z_stats, CONFIDENCES,
)
from simulate import generate_measurements, TRUE_VALUE_MEAN
from check_common import report, ALPHA, dist_arg_parser

RNG = np.random.default_rng(0)

N = 20            # measurements per simulated dataset
B_REF = 20000     # reference honest datasets (enough to resolve the 0.005 tail)
CI_SUB = 8000     # how many of those to reuse for the confidence-interval bands
N_TYPE1 = 20000   # trials at c = 1 (to measure the tiny type-I rate)
N_OTHER = 2000    # trials at each c > 1 (to measure power)
C_GRID = [1.0, 1.25, 1.5, 2.0, 3.0]

# Which noise model generates the DATA being tested. Rebound from the command line
# in __main__ before the worker pool forks. The reference simulation built below
# is ALWAYS the textbook independent-normal model -- that is the null hypothesis
# the test inverts. Choosing --dist t or --dist corr only changes the data fed to
# the fixed test, exposing how its error rate inflates under misspecification.
DIST, DF, CORR = "normal", 3.0, 0.1

# The error-bar pattern (sigma) is fixed and the same for every trial, because the
# years (hence sigma) depend only on n, not on c or the noise draw.
_values, SIGMA, _years = generate_measurements(N, TRUE_VALUE_MEAN, 1.0, RNG)

# Build the reference simulation ONCE and precompute everything the trials need.
print(f"Building reference null (B={B_REF}, n={N})...")
_birge_draws, _abs_z = parametric_null_sim(SIGMA, B_REF, RNG)

# Sorted reference distributions of each statistic at c = 1, for the p-values.
BIRGE_REF_SORTED = np.sort(_birge_draws)
AREAS_C1_SORTED = np.sort(_areas_over_grid(_abs_z, np.array([1.0]), CONFIDENCES)[:, 0])
SHARES_C1_SORTED = np.sort(_proportions_over_grid(_abs_z, np.array([1.0]))[:, 0])

# Reference quantile bands across the candidate-c grid, for the confidence
# intervals (these only need the smaller CI_SUB subsample).
_area_grid = _areas_over_grid(_abs_z[:CI_SUB], DEFAULT_C0_GRID, CONFIDENCES)
AREA_QLO = np.quantile(_area_grid, 0.025, axis=0)
AREA_QHI = np.quantile(_area_grid, 0.975, axis=0)
_share_grid = _proportions_over_grid(_abs_z[:CI_SUB], DEFAULT_C0_GRID)
SHARE_QLO = np.quantile(_share_grid, 0.025, axis=0)
SHARE_QHI = np.quantile(_share_grid, 0.975, axis=0)


def _mc_p_upper(sorted_reference, observed):
    """Safe upper-tail p-value: fraction of reference values >= observed."""
    ge = len(sorted_reference) - np.searchsorted(sorted_reference, observed, side="left")
    return (1 + ge) / (len(sorted_reference) + 1)


def _mc_p_lower(sorted_reference, observed):
    """Safe lower-tail p-value: fraction of reference values <= observed."""
    le = np.searchsorted(sorted_reference, observed, side="right")
    return (1 + le) / (len(sorted_reference) + 1)


def _dataset(seed, c):
    """One simulated dataset at true overprecision c, with the chosen noise model."""
    return generate_measurements(N, TRUE_VALUE_MEAN, c, np.random.default_rng(seed),
                                 dist=DIST, df=DF, corr=CORR)


def trial_birge_chi2(args):
    seed, c = args
    values, sigma, _ = _dataset(seed, c)
    _, birge = birge_ratio(values, sigma)
    ci, p = birge_ratio_conf_p(birge, len(values), coverage=0.95)
    return float(min(ci)), float(max(ci)), float(p)


def trial_birge_engine(args):
    seed, c = args
    values, sigma, _ = _dataset(seed, c)
    _, birge = birge_ratio(values, sigma)
    ci, _ = birge_ratio_conf_p(birge, len(values), coverage=0.95)  # same CI as the chi2 test
    # ...but use the simulation engine for the p-value, to cross-check it.
    return float(min(ci)), float(max(ci)), float(_mc_p_upper(BIRGE_REF_SORTED, birge))


def trial_area(args):
    seed, c = args
    values, sigma, _ = _dataset(seed, c)
    z, _, _ = pairwise_z_stats(values, sigma)
    area, _ = calibration_area_from_z(z, CONFIDENCES)
    p = _mc_p_upper(AREAS_C1_SORTED, area)
    c_lo, c_hi = invert_ci(DEFAULT_C0_GRID, AREA_QLO, AREA_QHI, area)
    lo, hi = sorted([area_of_c(c_lo), area_of_c(c_hi)])
    return lo, hi, float(p)


def trial_proportion(args):
    seed, c = args
    values, sigma, _ = _dataset(seed, c)
    z, _, _ = pairwise_z_stats(values, sigma)
    share = float(np.mean(z <= 1.0))
    p = _mc_p_lower(SHARES_C1_SORTED, share)
    c_lo, c_hi = invert_ci(DEFAULT_C0_GRID, SHARE_QLO, SHARE_QHI, share)
    lo, hi = sorted([proportion_of_c(c_lo), proportion_of_c(c_hi)])
    return lo, hi, float(p)


CHECKS = [
    ("birge (chi2)",   trial_birge_chi2,   lambda c: c),
    ("birge (engine)", trial_birge_engine, lambda c: c),
    ("area",           trial_area,         area_of_c),
    ("proportion",     trial_proportion,   proportion_of_c),
]


if __name__ == "__main__":
    args = dist_arg_parser("Parametric overprecision validation").parse_args()
    DIST, DF, CORR = args.dist, args.df, args.corr
    print(f"Parametric overprecision validation (n={N}, alpha={ALPHA}, B={B_REF}, "
          f"dist={DIST}" + (f", df={DF}" if DIST == "t" else "")
          + (f", corr={CORR}" if DIST == "corr" else "") + ")")
    report(CHECKS, C_GRID, N_TYPE1, N_OTHER, RNG)
