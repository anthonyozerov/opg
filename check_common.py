"""
Shared machinery for the validation scripts (check_parametric.py, check_jackknife.py).

These scripts answer a "do our methods actually work?" question by brute force:
generate thousands of datasets where we KNOW the true overprecision c, run a
method on each, and tally how often it gets the right answer. Two things are
checked:
  * Coverage  -- does the 95% confidence interval contain the true value about
                 95% of the time?
  * Type-I / power -- when error bars are honest (c = 1), does the test wrongly
                 cry "overprecision!" only as often as it should (the type-I rate,
                 here 0.005)? And when there IS overprecision (c > 1), how often
                 does it correctly catch it (the power)?

A "check" is described by a triple (label, trial_fn, true_of_c):
  * label      -- a short name for the table;
  * trial_fn((seed, c)) -> (ci_low, ci_high, p_value) -- runs one dataset;
  * true_of_c(c) -- the statistic's true value at data-generating c (the target
                    the confidence interval is supposed to cover).

`report` runs every check across a grid of c values and prints the table.

Two technical notes (see STATISTICS_PLAN.md section 5):
  * The test level is alpha = 0.005, so detecting a 0.005 error rate reliably
    needs a LOT of trials (N_TYPE1).
  * The trials run in parallel worker processes. We deliberately use the "fork"
    start method so workers inherit any heavy precomputed data (like a reference
    simulation) instead of rebuilding it each. trial_fn must be a top-level
    function so it can be sent to the workers (a functools.partial of one is fine).
"""

import argparse
import numpy as np
import multiprocessing as mp

CTX = mp.get_context("fork")
ALPHA = 0.005


def dist_arg_parser(description):
    """Command-line options for choosing the data-generating noise model.

    This lets a validation run swap the textbook (independent-normal) data for a
    deliberately mismatched alternative WITHOUT changing the method under test:
      --dist t      heavy-tailed noise (same overall spread)
      --dist corr   correlated noise
    The method stays the same; only the simulated data changes, so the resulting
    table shows how well the fixed method holds up when its assumptions are
    violated. (See REVIEW_NOTES.md.)
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dist", choices=["normal", "t", "corr"], default="normal",
                   help="error distribution for the simulated data (default: normal)")
    p.add_argument("--df", type=float, default=3.0,
                   help="degrees of freedom when --dist t (default: 3)")
    p.add_argument("--corr", type=float, default=0.1,
                   help="pairwise correlation when --dist corr (default: 0.1)")
    return p


def run_trials(trial_fn, c, n_trials, rng, chunksize=64):
    """Run n_trials datasets at true overprecision c, in parallel.

    Returns an (n_trials, 3) array whose rows are (ci_low, ci_high, p_value).
    """
    seeds = rng.integers(0, 2**63, n_trials)
    args = [(int(s), c) for s in seeds]
    with CTX.Pool() as pool:
        results = list(pool.imap(trial_fn, args, chunksize=chunksize))
    return np.asarray(results, dtype=float)


def summarize(results, true_value):
    """Reduce many trial outcomes to (coverage, rejection_rate, mean_ci_width)."""
    ci_low, ci_high, p_value = results[:, 0], results[:, 1], results[:, 2]
    coverage = float(np.mean((ci_low <= true_value) & (true_value <= ci_high)))
    rejection_rate = float(np.mean(p_value < ALPHA))
    mean_width = float(np.mean(ci_high - ci_low))
    return coverage, rejection_rate, mean_width


def report(checks, c_grid, n_type1, n_other, rng):
    """Run every check across the c-grid and print the coverage/power table."""
    print(f"{'check':<17}{'c':>6}{'true':>9}{'cover95':>10}"
          f"{'rej(p<.005)':>13}{'width':>10}")
    for label, trial_fn, true_of_c in checks:
        for c in c_grid:
            # The honest case (c = 1) needs many more trials to pin down the tiny
            # 0.005 type-I rate; other c values just measure power.
            n_trials = n_type1 if c == 1.0 else n_other
            results = run_trials(trial_fn, c, n_trials, rng)
            coverage, rejection_rate, width = summarize(results, true_of_c(c))
            tag = "type-I" if c == 1.0 else "power"
            print(f"{label:<17}{c:>6.2f}{true_of_c(c):>9.4f}{coverage:>10.3f}"
                  f"{rejection_rate:>13.4f}{width:>10.4f}  ({tag}, N={n_trials})")
        print()
