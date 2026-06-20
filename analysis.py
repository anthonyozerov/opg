"""
Analyse one dataset for signs of overprecision, and print a results table.

For a single simulated dataset this prints all three overprecision statistics
side by side -- the Birge ratio, the calibration-curve area, and the within-1-
sigma pairwise share -- and for each one shows both ways of getting a confidence
interval and a p-value (the parametric simulation route and the jackknife
leave-one-out route).

How to read the output:
  * Each statistic has a known calibrated value (Birge ratio = 1, area = 0, share
    ~= 0.6827). Departures in the overprecision direction (Birge ratio above 1,
    area above 0, share below 0.6827) indicate over-confidence.
  * The 95% confidence interval is a plausible range for the true value.
  * The p-value tests calibrated error bars against overprecision. A small p-value
    (flagged p < 0.005 with a *) is evidence of overprecision.

See STATISTICS_PLAN.md for the statistical details.
"""

import pandas as pd
import numpy as np

from preprocess import preprocess
from methods import (
    birge_ratio,
    birge_ratio_conf_p,
    birge_jackknife,
    parametric_null_sim,
    DEFAULT_C0_GRID,
)
from calibration import (
    calibration_and_ci,
    proportion_and_ci,
    area_parametric,
    proportion_parametric,
    P_NULL,
)

DATASET_NUM = 399
B = 20000             # number of simulated calibrated datasets for the parametric route
ALPHA_TEST = 0.005    # p-value threshold we flag as significant
RNG = np.random.default_rng(0)


def print_result_row(name, estimate, ci, p_value):
    """Print one line of the results table, flagging significant p-values."""
    flag = " *" if p_value < ALPHA_TEST else ""
    print(f"  {name:<24} est={estimate:+.4f}  95% CI=({ci[0]:+.4f}, {ci[1]:+.4f})  "
          f"p={p_value:.4f}{flag}")


if __name__ == "__main__":
    # Load one dataset and clean it down to usable G values + standard errors.
    df = pd.read_csv(f"sim-G/dataset_{DATASET_NUM:04d}.csv")
    preprocess(df)

    values = np.array(df.loc[df["use"], "value_processed"])
    sigma = np.array(df.loc[df["use"], "error_processed"])
    n = len(values)

    # Run the simulation engine once and reuse it for both pairwise statistics
    # (calibration area and within-1-sigma share) to avoid simulating twice.
    _, abs_z_sorted = parametric_null_sim(sigma, B, RNG)

    weighted_mean, birge = birge_ratio(values, sigma)
    print(f"Dataset {DATASET_NUM:04d}: n={n}, weighted mean = {weighted_mean:.6f}")
    print(f"(overprecision direction: Birge ratio > 1, area > 0, share < {P_NULL:.4f}; "
          f"* marks p < {ALPHA_TEST})\n")

    # --- Birge ratio ---
    birge_ci_param, birge_p_param = birge_ratio_conf_p(birge, n, coverage=0.95)
    birge_ci_lo_jack, birge_ci_hi_jack, birge_p_jack = birge_jackknife(
        values, sigma, scale="identity")
    print(f"Birge ratio = {birge:.4f}")
    print_result_row("  parametric (chi2)", birge, sorted(birge_ci_param), birge_p_param)
    print_result_row("  jackknife (identity)", birge,
                     (birge_ci_lo_jack, birge_ci_hi_jack), birge_p_jack)

    # --- Calibration area ---
    area_jack, area_lo_jack, area_hi_jack, _, area_p_jack = calibration_and_ci(values, sigma)
    area_param, area_lo_param, area_hi_param, area_p_param = area_parametric(
        values, sigma, abs_z_sorted=abs_z_sorted)
    print(f"\nCalibration area = {area_param:.4f}")
    print_result_row("  parametric", area_param,
                     (area_lo_param, area_hi_param), area_p_param)
    print_result_row("  jackknife", area_jack,
                     (area_lo_jack, area_hi_jack), area_p_jack)

    # --- Within-1-sigma share ---
    share_obs, share_lo_jack, share_hi_jack, share_p_jack = proportion_and_ci(
        values, sigma, scale="identity")
    share_param, share_lo_param, share_hi_param, share_p_param = proportion_parametric(
        values, sigma, abs_z_sorted=abs_z_sorted)
    print(f"\nWithin-1-sigma share = {share_obs:.4f}  (calibrated value {P_NULL:.4f})")
    print_result_row("  parametric", share_param,
                     (share_lo_param, share_hi_param), share_p_param)
    print_result_row("  jackknife", share_obs,
                     (share_lo_jack, share_hi_jack), share_p_jack)
