# Overprecision in measurements of G — start here

This project asks a simple question with fiddly statistics behind it:

> When scientists report a measurement together with an error bar, are those
> error bars **honest**, or are scientists systematically **over-confident**
> (reporting error bars that are too small)?

We study this using the gravitational constant **G**, which has a long history of
measurements whose error bars famously don't quite line up with each other. This
phenomenon — claiming more precision than the data support — is called
**overprecision**.

Rather than analyse the real historical data directly, we first build a
**simulator** that produces fake measurement histories where we *know* the right
answer. That lets us check whether our statistical methods actually detect
overprecision when it's present, and don't cry wolf when it isn't, before we trust
them on the real data and preregister an analysis.

There is no specialist physics here on the statistics side: every method reduces
to "given some numbers and their claimed error bars, do they scatter more than the
error bars say they should?"

---

## Key idea: the Birge ratio and "c"

If a set of measurements is honest, they should scatter around their average by
about as much as their own error bars predict. The **Birge ratio** measures the
actual scatter relative to the predicted scatter:

- **Birge ratio ≈ 1** → error bars are trustworthy.
- **Birge ratio > 1** → measurements scatter *more* than claimed = overprecision.
- **Birge ratio < 1** → measurements scatter *less* than claimed = conservativeness.

Throughout the code, the symbol **`c`** means the *true* underlying Birge ratio —
the real amount of overprecision, which the data only let us estimate.
**`TRUE_BIRGE`** is the specific value of `c` baked into a given simulated dataset.

We summarise overprecision three different but related ways (they should agree):

1. **Birge ratio** — the scatter measure above. Null hypothesis: 1.
2. **Calibration area** — how far the "claimed confidence vs. actual hit rate"
   curve sits below the ideal. Null hypothesis: 0.
3. **Within-1-sigma share** — the fraction of measurement *pairs* that agree to
   within one combined error bar. Null hypothesis: ≈ 0.6827.

For each, we report a **95% confidence interval** (a plausible range for the true
value) and a **p-value** testing "honest" vs. "overprecise" (small p = evidence of
overprecision; we flag p < 0.005).

---

### Data pipeline

1. **`simulate.py`** — invents 1000 fake measurement histories of G. Each uses one
   random true value of G and one random overprecision level (`TRUE_BIRGE`), then
   produces 75 noisy "published" measurements plus running averages. Writes the
   `sim-G/` and `sim-G-parameters/` folders.
2. **`preprocess.py`** — cleans up each dataset: converts every measurement to a
   plain G value with a plain standard error, and marks which rows are usable.

### Analysis

3. **`methods.py`** — the core statistical tools: the Birge ratio, its exact
   confidence interval/test, the "simulation engine" (`parametric_null_sim`), and
   the jackknife (leave-one-out) method.
4. **`calibration.py`** — the calibration curve and the other two statistics
   (calibration area, within-1-sigma share), each with both a jackknife and a
   simulation-based interval/test. Run directly, it draws a calibration curve for
   one example dataset.
5. **`analysis.py`** — the main report for a single dataset: prints all three
   statistics × both methods side by side. **This is the best file to run first to
   see the project in action.**

### Validation

6. **`check_common.py`** — shared scaffolding for the validation scripts.
7. **`check_parametric.py`** — checks the simulation-based methods by running them
   on thousands of datasets with a known answer.
8. **`check_jackknife.py`** — the same, for the jackknife methods.
9. **`check_bootstrap.py`** — an older, area-only cross-check using a third method
   (the bootstrap).

---

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

### Reading `analysis.py`'s output

```
Dataset 0399: n=75, weighted mean G = 6.674195
Birge ratio = 2.3539
    parametric (chi2)      est=+2.3539  95% CI=(+2.0281, +2.8052)  p=0.0000 *
    jackknife (identity)   est=+2.3539  95% CI=(+2.0040, +2.7124)  p=0.0000 *
...
```

Each statistic is shown with its estimate, a 95% confidence interval, and a
p-value. A trailing `*` flags p < 0.005 — strong evidence of overprecision. Here
the Birge ratio of 2.35 (well above 1, CI excludes 1) says this dataset's
"scientists" were substantially over-confident — which is expected, because this
simulated dataset was built with `TRUE_BIRGE` greater than 1.
