# Overprecision in measurements of fundamental constants

This project studies whether reported error bars on measurements of fundamental
constants are calibrated, or whether they are systematically too small
(over-confident). Reporting more precision than the data support is called
overprecision.

The motivating case is the gravitational constant G, whose historical
measurements scatter more than their stated error bars predict, but the methods
here apply to any fundamental constant with repeated independent measurements.

Rather than analyse real historical data directly, we first build a simulator
that produces fake measurement histories with a known true answer. This lets us
check whether the statistical methods detect overprecision when it is present and
do not flag it when it is absent, before applying them to real data and
preregistering an analysis.

The statistics reduce to one question: given some numbers and their stated error
bars, do they scatter more than the error bars say they should?

## The Birge ratio and c

If a set of measurements is well calibrated, they should scatter around their
average by about as much as their error bars predict. The Birge ratio measures the
actual scatter relative to the predicted scatter:

- Birge ratio ≈ 1: error bars are consistent with the scatter.
- Birge ratio > 1: measurements scatter more than claimed (overprecision).
- Birge ratio < 1: measurements scatter less than claimed (conservativeness).

In the code, `c` is the true underlying Birge ratio: the actual amount of
overprecision, which the data only let us estimate. `TRUE_BIRGE` is the value of
`c` used to generate a given simulated dataset.

We summarise overprecision three related ways, which should agree:

1. Birge ratio — the scatter measure above. Null hypothesis: 1.
2. Calibration area — how far the stated-confidence vs. actual-hit-rate curve sits
   below the ideal. Null hypothesis: 0.
3. Within-1-sigma share — the fraction of measurement pairs that agree to within
   one combined error bar. Null hypothesis: ≈ 0.6827.

Each is reported with a 95% confidence interval and a one-sided p-value testing
calibrated vs. overprecise (small p is evidence of overprecision; we flag
p < 0.005).

## Data pipeline

1. `simulate.py` — generates 1000 fake measurement histories of G. Each uses one
   random true value of G and one random overprecision level (`TRUE_BIRGE`), then
   produces 75 noisy measurements plus running averages. Writes the `sim-G/` and
   `sim-G-parameters/` folders.
2. `preprocess.py` — cleans each dataset: converts every measurement to a plain G
   value with a plain standard error, and marks which rows are usable.

## Analysis

3. `methods.py` — the core statistical tools: the Birge ratio, its exact
   confidence interval/test, the simulation engine (`parametric_null_sim`), and the
   jackknife (leave-one-out) method.
4. `calibration.py` — the calibration curve and the other two statistics
   (calibration area, within-1-sigma share), each with both a jackknife and a
   simulation-based interval/test. Run directly, it draws a calibration curve for
   one example dataset.
5. `analysis.py` — the report for a single dataset: prints all three statistics ×
   both methods side by side. Run this first to see the project in action.

## Validation

6. `check_common.py` — shared scaffolding for the validation scripts.
7. `check_parametric.py` — checks the simulation-based methods by running them on
   thousands of datasets with a known answer.
8. `check_jackknife.py` — the same, for the jackknife methods.
9. `check_bootstrap.py` — an older, area-only cross-check using the bootstrap.

## How to run it

Everything uses the `opg` conda environment:

```bash
conda activate opg

python analysis.py        # analyse one example dataset (dataset 399 by default)
python calibration.py     # draw a calibration curve (dataset 500 by default)
python simulate.py        # regenerate all 1000 simulated datasets (~minutes)

# Validation runs (these simulate many datasets and take longer):
python check_parametric.py
python check_jackknife.py
python check_bootstrap.py
```

To analyse a different dataset, change `DATASET_NUM` near the top of `analysis.py`
or `calibration.py`.

## Reading analysis.py's output

```
Dataset 0399: n=75, weighted mean G = 6.674195
Birge ratio = 2.3539
    parametric (chi2)      est=+2.3539  95% CI=(+2.0281, +2.8052)  p=0.0000 *
    jackknife (identity)   est=+2.3539  95% CI=(+2.0040, +2.7124)  p=0.0000 *
...
```

Each statistic is shown with its estimate, a 95% confidence interval, and a
p-value. A trailing `*` flags p < 0.005. Here the Birge ratio of 2.35 (above 1,
CI excludes 1) indicates this dataset's measurements were substantially
over-confident, which is expected because it was generated with `TRUE_BIRGE`
greater than 1.
