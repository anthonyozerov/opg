"""
Simulate historical measurements of the gravitational constant G.

Each simulated dataset covers measurements of G between 1700 and 2020, built from
one randomly chosen true value of G and one randomly chosen amount of
over-confidence (TRUE_BIRGE, explained below). Realistic noise produces 75
measurements, and we compute running averages of the most recent measurements as
metrologists do.

The goal is not to model the real history of physics but to create test data with
a known correct answer, so the statistical methods (in analysis.py /
calibration.py) can be checked for whether they detect over-confidence when it is
present.

Running this script writes two folders:
  sim-G/             one CSV per dataset: the measurements + the averages
  sim-G-parameters/  one CSV per dataset: the true values used to make it
"""

import numpy as np
import pandas as pd
import os

# How many datasets to generate.
N_DATASETS = 1000

# The true value of G is drawn fresh for each dataset from a normal (bell-curve)
# distribution centered here. (Units are 1e-11; we keep the 1e-11 implicit.)
TRUE_VALUE_MEAN = 6.67430
TRUE_VALUE_STD = 0.00015

# Each dataset contains this many fake measurements, spread across these years.
N_MEASUREMENTS = 75
YEAR_START = 1700
YEAR_END = 2020

# Nominal uncertainty is the error bar a scientist reports. As instruments
# improved, the reported error bar shrinks from NOM_ERR_START (1700) to
# NOM_ERR_END (2020).
NOM_ERR_START = 0.1
NOM_ERR_END = 0.00010

# Running averages are computed every AVG_YEAR_STEP years, starting in this year.
AVG_YEAR_START = 1930
AVG_YEAR_STEP = 5

# Real published values are reported in different forms, which we mimic. A value
# may be reported as G or as a density rho (G = 36.797 / rho). An error bar may be
# a standard error, a probable error, or an average deviation. These constants
# encode each form; preprocess.py decodes them back to plain G + standard errors.
VALUE_TYPES = ["G", "rho"]
ERROR_TYPES = ["probable error", "average deviation", "standard"]
# To turn a standard error into each reported form, multiply by these factors.
# (preprocess.py divides by the same factor to undo the encoding.)
ERROR_TYPE_FACTORS = {"probable error": 0.6745, "average deviation": 0.79788, "standard": 1.0}

RHO_CONSTANT = 36.797  # the constant relating density to G: G = RHO_CONSTANT / rho


def generate_years(n, year_start, year_end):
    """Pick n measurement years between year_start and year_end.

    The years are not evenly spaced: they bunch up toward the recent end, because
    measurements of G became more frequent over time. The formula below maps an
    evenly spaced helper variable t (0..1) onto years using 1 - (1 - t)**2, a
    curve that stretches early gaps and compresses recent ones.
    """
    t = np.linspace(0, 1, n)
    years = year_start + (year_end - year_start) * (1 - (1 - t) ** 2)
    return np.sort(years)


def nominal_errors(years):
    """The reported error bar for a measurement taken in each given year.

    Uncertainty shrinks geometrically (by a constant percentage per year, not a
    constant amount), done by interpolating in log-space between the 1700 and 2020
    values and exponentiating back.
    """
    fraction_of_span = (years - YEAR_START) / (YEAR_END - YEAR_START)
    log_start = np.log(NOM_ERR_START)
    log_end = np.log(NOM_ERR_END)
    return np.exp(log_start + fraction_of_span * (log_end - log_start))


def birge_weighted_average(values, nominal_errors):
    """Combine several measurements into one estimate with an error bar.

    The Birge-ratio weighted average used by metrologists:
      1. Weight each measurement by 1 / error^2 (precise measurements count more).
      2. The weighted mean is the estimate.
      3. If the measurements disagree more than their own error bars predict, that
         indicates unreported error, so the final error bar is inflated by the
         Birge ratio = sqrt(chi-squared / degrees of freedom). A Birge ratio near 1
         means the error bars are consistent with the scatter.
    """
    weights = 1.0 / nominal_errors**2
    total_weight = weights.sum()
    weighted_mean = (weights * values).sum() / total_weight

    # Error bar assuming the reported error bars are perfectly trustworthy.
    internal_error = 1.0 / np.sqrt(total_weight)

    # How much the measurements actually scatter, relative to their error bars.
    residuals = values - weighted_mean
    chi_squared = (weights * residuals**2).sum()
    degrees_of_freedom = len(values) - 1
    birge_ratio = np.sqrt(chi_squared / degrees_of_freedom) if degrees_of_freedom > 0 else 1.0

    # Inflate the error bar if the measurements scatter more than expected.
    external_error = internal_error * birge_ratio
    return weighted_mean, external_error


def generate_measurements(n, true_value, true_birge, rng, dist="normal", df=3.0, corr=0.1):
    """Create n measurements of the underlying quantity for one dataset.

    Returns (measured_values, nominal_errors, years):
      measured_values -- the noisy measured values
      nominal_errors  -- the error bar reported for each measurement
      years           -- when each measurement was taken

    Each measurement reports an error bar (nominal_errors), but its actual scatter
    is true_birge times larger. So true_birge > 1 means collective over-confidence
    (real errors bigger than claimed), the effect the later analysis detects.

    `dist` chooses the shape of the noise. The default, "normal", is the model the
    statistical tests assume. The other two keep the same overall noise size but
    violate an assumption, to measure how the tests behave under misspecification:
      * "normal" -- independent normal noise (the assumed model; the default, and
                    bit-for-bit identical to the original code).
      * "t"      -- heavier tails (more extreme outliers), same overall spread.
                    Tests robustness to non-normality. Requires df > 2.
      * "corr"   -- normal noise with measurements correlated rather than
                    independent. Tests robustness to dependence between
                    measurements.

    Only the noise shape changes; the reported error bars are untouched, so these
    options isolate the effect of misspecification on the fixed tests.
    """
    years = generate_years(n, YEAR_START, YEAR_END)
    nominal_errors_ = nominal_errors(years)
    actual_errors = nominal_errors_ * true_birge

    if dist == "normal":
        # Early return keeps the default draw byte-for-byte identical to the
        # original code, so previously generated sim-G/ files stay reproducible.
        measured_values = true_value + rng.normal(0, actual_errors)
        return measured_values, nominal_errors_, years

    if dist == "t":
        if df <= 2:
            raise ValueError("t distribution needs df > 2 for finite variance")
        # A Student-t variable is more spread out than a standard normal, so we
        # divide by its known standard deviation to put it back on unit scale.
        # That way only the tail shape is changed, not the overall spread.
        unit_noise = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2.0))
    elif dist == "corr":
        if not 0.0 <= corr < 1.0:
            raise ValueError("corr must be in [0, 1)")
        # Make every pair of measurements share a common random component plus its
        # own private noise. Mixing them in this proportion gives each pair a
        # correlation of exactly `corr` while keeping each measurement's spread 1.
        common_shock = rng.standard_normal()
        private_noise = rng.standard_normal(n)
        unit_noise = np.sqrt(corr) * common_shock + np.sqrt(1.0 - corr) * private_noise
    else:
        raise ValueError(f"unknown dist: {dist!r}")

    measured_values = true_value + unit_noise * actual_errors
    return measured_values, nominal_errors_, years


if __name__ == "__main__":
    # A seeded random-number generator makes the whole run reproducible: re-running
    # this script produces exactly the same datasets.
    RNG = np.random.default_rng(42)

    os.makedirs("sim-G", exist_ok=True)
    os.makedirs("sim-G-parameters", exist_ok=True)

    for dataset_index in range(N_DATASETS):
        # Draw the ground truth for this dataset: the true value of G and how
        # over-confident the measurements are (true_birge).
        true_value = RNG.normal(TRUE_VALUE_MEAN, TRUE_VALUE_STD)
        true_birge = RNG.uniform(1, 4)

        measured_values, nominal_errs, years = generate_measurements(
            N_MEASUREMENTS, true_value, true_birge, RNG)

        # Give each measurement a random reporting form (see the constants above):
        # some reported as G vs. density, some with different error types.
        value_types = RNG.choice(VALUE_TYPES, size=N_MEASUREMENTS)
        error_types = RNG.choice(ERROR_TYPES, size=N_MEASUREMENTS)
        error_factors = np.array([ERROR_TYPE_FACTORS[e] for e in error_types])

        # Encode the clean G values into the chosen reporting form. Density (rho)
        # is the reciprocal relation, and its error bar transforms with it.
        rho_values = RHO_CONSTANT / measured_values
        nominal_errs_rho = nominal_errs * rho_values / measured_values

        stored_values = np.where(value_types == "G", measured_values, rho_values)
        stored_errors = np.where(value_types == "G", nominal_errs, nominal_errs_rho) * error_factors

        measurements_df = pd.DataFrame({
            "year": years,
            "value": stored_values,
            "value_type": value_types,
            "error": stored_errors,
            "error_type": error_types,
            "doc_type": np.nan,  # NaN here marks a raw measurement (vs. "M" averages)
        })

        # How many of the most recent measurements each running average uses. This
        # is fixed within a dataset but varies between datasets (8 to 14).
        n_recent = int(RNG.integers(8, 15))  # 8 to 14 inclusive

        # Compute a running estimate every 5 years from 1930 onward, each time
        # combining the n_recent most recent measurements available by then.
        average_years = list(range(AVG_YEAR_START, YEAR_END + 1, AVG_YEAR_STEP))
        average_rows = []
        for year in average_years:
            measurements_so_far = np.where(years <= year)[0]
            if len(measurements_so_far) < n_recent:
                continue
            recent = measurements_so_far[-n_recent:]
            avg_value, avg_error = birge_weighted_average(
                measured_values[recent], nominal_errs[recent])
            average_rows.append({
                "year": year,
                "value": avg_value,
                "value_type": "G",
                "error": avg_error,
                "error_type": "standard",
                "doc_type": "M",  # "M" marks an averaged row, not a raw measurement
            })

        # Store the raw measurements and the averages together in one CSV.
        combined_df = pd.concat([measurements_df, pd.DataFrame(average_rows)],
                                ignore_index=True)
        combined_df.to_csv(f"sim-G/dataset_{dataset_index:04d}.csv", index=False)

        # Separately record the ground-truth parameters used for this dataset, so
        # the analysis scripts can later check their answers against the truth.
        pd.DataFrame([{
            "dataset": dataset_index,
            "TRUE_G": true_value,
            "TRUE_BIRGE": true_birge,
            "year_start": YEAR_START,
            "year_end": YEAR_END,
            "n_measurements": N_MEASUREMENTS,
            "nom_err_start": NOM_ERR_START,
            "nom_err_end": NOM_ERR_END,
            "n_recent": n_recent,
        }]).to_csv(f"sim-G-parameters/dataset_{dataset_index:04d}.csv", index=False)

        if (dataset_index + 1) % 100 == 0:
            print(f"  {dataset_index + 1}/{N_DATASETS} datasets done")

    print("Done.")
