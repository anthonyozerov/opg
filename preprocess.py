"""
Clean up the raw measurements so they are ready for analysis.

simulate.py stores each measurement in whatever "flavour" a scientist might have
reported it (as G or as density rho; with a standard, probable, or average-
deviation error bar). This script reverses that encoding so every usable row has:
  * value_processed -- the measurement expressed as plain G
  * error_processed -- its error bar expressed as a plain standard error
  * use             -- True if the row is a real measurement we should analyse

Rows we skip (use = False): the running-average rows (doc_type "M"), any rows
marked superseded, and any row whose flavour we don't recognise.
"""

import pandas as pd
import numpy as np


def rho_to_G(rho, err):
    """Convert a density measurement (rho) and its error bar into G."""
    G = 36.797 / rho
    # When you take a reciprocal, the relative error stays the same, so the
    # absolute error scales by G / rho.
    return G, G * err / rho


def c_air_to_c(c_air, err):
    """Convert a speed-of-light-in-air value to vacuum (kept for real-data use)."""
    c = c_air * 1.0002926
    return c, c * err / c_air


def preprocess(df):
    """Add value_processed, error_processed, and use columns to df, in place."""
    values = np.array(df["value"])
    value_type = np.array(df["value_type"])
    errors = np.array(df["error"])
    error_type = np.array(df["error_type"])
    doc_type = np.array(df["doc_type"], dtype=object)
    # Real datasets may have a "superseded" column; simulated ones don't, so
    # default everyone to "not superseded".
    superseded = (np.array(df["superseded"]) if "superseded" in df.columns
                  else np.full(len(values), False))

    values_processed = np.full_like(values, np.nan)
    errors_processed = np.full_like(values, np.nan)
    use = np.zeros(len(values), dtype=bool)

    for i in range(len(values)):
        value = values[i]
        error = errors[i]

        # Skip rows that are not raw, usable measurements.
        if doc_type[i] in ['M', 'R']:  # averaged ("M") or review ("R") rows
            continue
        if superseded[i] == 'Y':
            continue
        assert np.isnan(doc_type[i])  # at this point only raw rows should remain

        # Step 1: turn whatever error flavour was reported into a standard error.
        if error_type[i] == "probable error":
            error = error / 0.6745
        elif error_type[i] == "average deviation":
            error = error / 0.79788
        elif error_type[i] == "standard":
            pass  # already a standard error
        else:
            print(f"can't handle error type: {error_type[i]}, skipping")
            continue

        # Step 2: turn whatever value flavour was reported into plain G.
        if value_type[i] in ["G", "c"]:
            values_processed[i], errors_processed[i] = value, error
        elif value_type[i] == "rho":
            values_processed[i], errors_processed[i] = rho_to_G(value, error)
        elif value_type[i] == "c_air":
            values_processed[i], errors_processed[i] = c_air_to_c(value, error)
        else:
            print(f"Unknown value type: {value_type[i]}, skipping")
            continue

        use[i] = True

    df["value_processed"] = values_processed
    df["error_processed"] = errors_processed
    df["use"] = use
