"""
Validate the three jackknife (leave-one-out) overprecision tests + intervals:
  * Birge ratio              -- methods.birge_jackknife
  * calibration area         -- calibration.calibration_and_ci
  * within-1-sigma share     -- calibration.proportion_and_ci

Unlike the parametric tests, the jackknife is not guaranteed exact in small samples
(it relies on an approximation that improves with sample size), so its 0.005 error
rate has to be checked empirically here. The candidate variance-stabilizing scales
are kept as separate rows so the frozen final choices (Birge ratio = identity,
share = identity) rest on evidence.
"""

import numpy as np
from functools import partial

from methods import birge_jackknife
from calibration import calibration_and_ci, proportion_and_ci, area_of_c, proportion_of_c
from simulate import generate_measurements, TRUE_VALUE_MEAN
from check_common import report, ALPHA, dist_arg_parser

RNG = np.random.default_rng(0)

N = 20            # measurements per simulated dataset
N_TYPE1 = 20000   # trials at c = 1 (to measure the tiny type-I rate)
N_OTHER = 2000    # trials at each c > 1 (to measure power)
C_GRID = [0.9, 1.0, 1.25, 1.5, 2.0, 3.0]

# Which noise model generates the data being tested. Rebound from the command line
# in __main__ before the worker pool forks. Defaults to the textbook model.
DIST, DF, CORR = "normal", 3.0, 0.1


def _dataset(seed, c):
    """One simulated dataset at true overprecision c, with the chosen noise model."""
    return generate_measurements(N, TRUE_VALUE_MEAN, c, np.random.default_rng(seed),
                                 dist=DIST, df=DF, corr=CORR)


def trial_birge(args, scale):
    seed, c = args
    values, sigma, _ = _dataset(seed, c)
    lo, hi, p = birge_jackknife(values, sigma, scale=scale)
    return lo, hi, p


def trial_proportion(args, scale):
    seed, c = args
    values, sigma, _ = _dataset(seed, c)
    _, lo, hi, p = proportion_and_ci(values, sigma, scale=scale)
    return lo, hi, p


def trial_area(args):
    seed, c = args
    values, sigma, _ = _dataset(seed, c)
    _, lo, hi, _, p = calibration_and_ci(values, sigma)
    return lo, hi, p


CHECKS = [
    ("birge (identity)", partial(trial_birge, scale="identity"), lambda c: c),
    ("birge (log)",      partial(trial_birge, scale="log"),      lambda c: c),
    ("birge (sq)",       partial(trial_birge, scale="sq"),       lambda c: c),
    ("area",             trial_area,                             area_of_c),
    ("prop (identity)",  partial(trial_proportion, scale="identity"), proportion_of_c),
    ("prop (logit)",     partial(trial_proportion, scale="logit"),    proportion_of_c),
]


if __name__ == "__main__":
    args = dist_arg_parser("Jackknife overprecision validation").parse_args()
    DIST, DF, CORR = args.dist, args.df, args.corr
    print(f"Jackknife overprecision validation (n={N}, alpha={ALPHA}, "
          f"dist={DIST}" + (f", df={DF}" if DIST == "t" else "")
          + (f", corr={CORR}" if DIST == "corr" else "") + ")")
    report(CHECKS, C_GRID, N_TYPE1, N_OTHER, RNG)
