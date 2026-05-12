#!/usr/bin/env python3
"""
fit_thermal_models.py

Fit deformation models to one observed InSAR point time series, both with and
without a thermal term.

Input:
  - CSV with DYYYYMMDD displacement columns
  - optional point_id
  - optional temperature CSV
  - otherwise downloads temperature using Meteostat
  - optional temperature predictor mode: hourly, daily mean, rolling means, and lagged temperature

Models:
  1) linear only
  2) linear + thermal
  3) quadratic only
  4) quadratic + thermal
  5) exponential only
  6) exponential + thermal

Outputs:
  - observed_point_timeseries_fitted.csv
  - model_comparison.csv
  - thermal_improvement_summary.csv
  - estimated_parameters_all_models.csv
  - 01_observed_timeseries.png
  - 02_model_comparison.png
  - 03_thermal_correction_best_model.png
  - 04_residuals_linear_no_thermal.png
  - 04_residuals_linear_thermal.png
  - 05_residuals_quadratic_no_thermal.png
  - 05_residuals_quadratic_thermal.png
  - 06_residuals_exponential_no_thermal.png
  - 06_residuals_exponential_thermal.png
  - 07_temperature.png
  - 08_thermal_vs_no_thermal.png
  - 09_fits_and_corrected_subplot.png
  - 10_detrended_thermal_residual_check.png
"""

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def parse_args():
    examples = """
Examples
--------
1) Fit one selected point using downloaded Meteostat temperature:
   fit_thermal_models.py \\
     --csv test_data/RitzCarlton/TSX_036_20170923_20251008_N2592W08012_N2592W08012_N2592W08012_N2592W08012.csv \\
     --point-id 18582 \\
     --temperature-mode rolling_3day \\
     --out results/RitzCarlton_point18582_rolling3day

2) Fit one selected point using hourly temperature:
   fit_thermal_models.py \\
     --csv test_data/RitzCarlton/TSX_036_20170923_20251008_N2592W08012_N2592W08012_N2592W08012_N2592W08012.csv \\
     --point-id 18582 \\
     --temperature-mode hourly \\
     --out results/RitzCarlton_point18582_hourly

3) Fit one selected point using a user-provided temperature CSV:
   fit_thermal_models.py \\
     --csv test_data/RitzCarlton/TSX_036_20170923_20251008_N2592W08012_N2592W08012_N2592W08012_N2592W08012.csv \\
     --point-id 18582 \\
     --temperature-csv temperature_table.csv \\
     --temperature-mode daily_mean \\
     --out results/RitzCarlton_point18582_daily_temperature_csv

4) Automatically select the most negative velocity point above a coherence threshold:
   fit_thermal_models.py \\
     --csv test_data/PorscheDesignTower/TSX_036_20170923_20251008_N2595W08012_N2595W08012_N2595W08012_N2595W08012.csv \\
     --select-most-negative \\
     --min-coherence 0.2 \\
     --temperature-mode rolling_3day \\
     --out results/Porsche_most_negative_rolling3day

5) Test a temperature lag:
   fit_thermal_models.py \\
     --csv test_data/RitzCarlton/TSX_036_20170923_20251008_N2592W08012_N2592W08012_N2592W08012_N2592W08012.csv \\
     --point-id 18582 \\
     --temperature-mode rolling_7day \\
     --temperature-lag-days 1 \\
     --out results/RitzCarlton_point18582_rolling7day_lag1

Notes
-----
- The script fits six models: linear, linear+thermal, quadratic, quadratic+thermal,
  exponential, and exponential+thermal.
- The thermal predictor is temperature anomaly, not raw temperature.
- rolling_3day, rolling_7day, and rolling_14day use trailing daily rolling means.
- If --temperature-csv is not provided, temperature is downloaded using Meteostat.
"""

    p = argparse.ArgumentParser(
        description="Fit deformation and thermal models to one InSAR displacement point.",
        epilog=examples,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("--csv", required=True, help="Input CSV.")
    p.add_argument("--point-id", type=int, default=None, help="Point ID to fit.")
    p.add_argument("--select-most-negative", action="store_true",
                   help="If no point ID is given, select the most negative velocity point.")
    p.add_argument("--min-coherence", type=float, default=None,
                   help="Optional minimum coherence for automatic point selection.")
    p.add_argument("--out", default="observed_thermal_fit", help="Output directory.")

    p.add_argument("--lat", type=float, default=None,
                   help="Latitude for temperature download. Default: selected point Y.")
    p.add_argument("--lon", type=float, default=None,
                   help="Longitude for temperature download. Default: selected point X.")
    p.add_argument("--hour", type=int, default=18,
                   help="Local acquisition hour used for temperature matching.")
    p.add_argument("--temperature-csv", default=None,
                   help="Optional temperature CSV with date/datetime and temperature_C/temp_C column.")

    p.add_argument("--temperature-mode",
                   choices=["hourly", "daily_mean", "rolling_3day", "rolling_7day", "rolling_14day"],
                   default="hourly",
                   help="Temperature predictor used for the thermal term.")

    p.add_argument("--temperature-lag-days", type=float, default=0.0,
                   help="Lag applied to temperature before matching acquisitions. "
                        "Example: 1 means use temperature from one day before acquisition.")

    p.add_argument("--exp-tau-upper", type=float, default=50.0,
                   help="Upper bound for exponential tau in years.")

    return p.parse_args()


def find_dcols(columns):
    """
    - Find SAR acquisition date columns,
    - make sure they are in time order,
    - and return them. (D20170923, D20171004, D20171108,...)
    """
    dcols = [c for c in columns if c.startswith("D") and len(c) == 9 and c[1:].isdigit()]
    dcols = sorted(dcols)
    if not dcols:
        raise ValueError("No DYYYYMMDD displacement columns found.")
    return dcols


def load_point(csv_path, point_id=None, select_most_negative=False, min_coherence=None):
    """
    - Read InSAR CSV,
    - Select one point/pixel,
    - Extract that point's displacement time series
    - Prepare the time and displacement arrays for model fitting.
    """
    df = pd.read_csv(csv_path)
    dcols = find_dcols(df.columns)

    if point_id is not None:
        if "point_id" not in df.columns:
            raise ValueError("CSV has no point_id column, but --point-id was provided.")
        rows = df[df["point_id"] == point_id]
        if rows.empty:
            raise ValueError(f"point_id {point_id} not found in CSV.")
        row = rows.iloc[0].copy()
    else:
        candidates = df.copy()

        if min_coherence is not None and "coherence" in candidates.columns:
            candidates = candidates[candidates["coherence"] >= min_coherence]
            if candidates.empty:
                raise ValueError("No points remain after applying --min-coherence.")

        if select_most_negative:
            if "velocity" not in candidates.columns:
                raise ValueError("CSV has no velocity column; cannot use --select-most-negative.")
            row = candidates.loc[candidates["velocity"].idxmin()].copy()
        else:
            row = candidates.iloc[0].copy()

    dates = pd.to_datetime([c[1:] for c in dcols], format="%Y%m%d")
    y = row[dcols].astype(float).to_numpy()

    valid = np.isfinite(y)
    dates = dates[valid]
    y = y[valid]

    if len(y) < 10:
        raise ValueError("Too few valid displacement observations for fitting.")

    y = y - y[0]
    t_years = (dates - dates[0]).days.to_numpy() / 365.25

    point = {
        "point_id": int(row["point_id"]) if "point_id" in row.index else None,
        "X": float(row["X"]) if "X" in row.index else np.nan,
        "Y": float(row["Y"]) if "Y" in row.index else np.nan,
        "velocity": float(row["velocity"]) if "velocity" in row.index else np.nan,
        "coherence": float(row["coherence"]) if "coherence" in row.index else np.nan,
        "omega": float(row["omega"]) if "omega" in row.index else np.nan,
        "st_consist": float(row["st_consist"]) if "st_consist" in row.index else np.nan,
        "dem_error": float(row["dem_error"]) if "dem_error" in row.index else np.nan,
        "dem": float(row["dem"]) if "dem" in row.index else np.nan,
    }

    return dates, t_years, y, point


def make_datetimes(dates, hour):
    """
    Take each SAR acquisition date and add a specific hour,
    match temperature at that time.
    """
    return pd.to_datetime([pd.Timestamp(d.date()) + pd.Timedelta(hours=hour) for d in dates])


def load_temperature_series_from_csv(path):
    """
    Load a temperature time series data from user-provided CSV file.

    CSV can be hourly or daily. Function only requires:
      - one time column: datetime, date, acquisition_datetime, or acquisition_date
      - one temperature column: temperature_C, temp_C, temp, or T_C
    """
    temp_df = pd.read_csv(path)

    time_col = None
    for candidate in ["datetime", "date", "acquisition_datetime", "acquisition_date"]:
        if candidate in temp_df.columns:
            time_col = candidate
            break

    temp_col = None
    for candidate in ["temperature_C", "temp_C", "temp", "T_C", "temperature", "air_temperature", "air_temp_C"]:
        if candidate in temp_df.columns:
            temp_col = candidate
            break

    if time_col is None or temp_col is None:
        raise ValueError(
            "Temperature CSV must contain a date/datetime column and a temperature column "
            "(temperature_C, temp_C, temp, T_C, temperature, air_temperature, air_temp_C)."
        )

    temp_df[time_col] = pd.to_datetime(temp_df[time_col])
    temp_series = temp_df.set_index(time_col)[temp_col].astype(float).sort_index()
    temp_series = temp_series[~temp_series.index.duplicated(keep="first")]
    temp_series = temp_series.interpolate(method="time").ffill().bfill()

    if temp_series.empty:
        raise RuntimeError("Temperature CSV produced an empty temperature series.")

    return temp_series


def download_temperature_series(datetimes, lat, lon, lag_days=0.0):
    """
    If user does not provide a temperature CSV,
    download temperature time series using Meteostat for the location of the selected InSAR point.

    Extra buffer is added so lagged and rolling temperature predictors can be computed.
    """
    try:
        from meteostat import Point, Hourly
    except ImportError as e:
        raise ImportError("Meteostat is not installed. Run:\n  pip install meteostat\n") from e

    # Add enough temporal buffer for rolling windows and lagged temperatures.
    rolling_buffer_days = 20
    lag_buffer_days = int(np.ceil(abs(lag_days))) + 2

    start = datetimes.min().to_pydatetime() - timedelta(days=rolling_buffer_days + lag_buffer_days)
    end = datetimes.max().to_pydatetime() + timedelta(days=2)

    location = Point(lat, lon)
    data = Hourly(location, start, end).fetch()

    if data.empty or "temp" not in data.columns:
        raise RuntimeError("No hourly temperature data returned by Meteostat.")

    temp_series = data["temp"].astype(float).copy()
    temp_series = temp_series[~temp_series.index.duplicated(keep="first")]
    temp_series = temp_series.sort_index()
    temp_series = temp_series.interpolate(method="time").ffill().bfill()

    return temp_series


def match_temperature_predictor(temp_series, datetimes, mode="hourly", lag_days=0.0):
    """
    Convert a temperature time series into the predictor used by the thermal model.

    Modes
    -----
    hourly:
        Nearest temperature at acquisition time within ±3 hours, after applying lag.
        If strict hourly matching fails, falls back to daily mean matching.

    daily_mean:
        Daily mean temperature for the lagged acquisition date.

    rolling_3day, rolling_7day, rolling_14day:
        Trailing rolling mean of daily temperature ending on the lagged acquisition date.
        For example, rolling_7day is the mean of the current day and previous 6 days.
    """
    target_datetimes = pd.DatetimeIndex(datetimes) - pd.to_timedelta(lag_days, unit="D")

    temp_series = temp_series.sort_index()
    temp_series = temp_series[~temp_series.index.duplicated(keep="first")]
    temp_series = temp_series.interpolate(method="time").ffill().bfill()

    if mode == "hourly":
        # First try strict hourly matching. This is appropriate for true hourly data.
        matched = temp_series.reindex(
            target_datetimes,
            method="nearest",
            tolerance=pd.Timedelta(hours=3),
        )

        # If strict hourly matching fails, the temperature CSV is probably daily
        # or lower-frequency data. In that case, fall back to daily mean matching
        # instead of crashing.
        if matched.isna().any():
            daily = temp_series.resample("D").mean()
            daily = daily.interpolate(method="time").ffill().bfill()

            target_days = target_datetimes.normalize()
            matched_daily = daily.reindex(
                target_days,
                method="nearest",
                tolerance=pd.Timedelta(days=1),
            )

            if not matched_daily.isna().any():
                print(
                    "[WARNING] Hourly temperature matching failed for at least one acquisition. "
                    "Falling back to daily mean temperature matching."
                )
                matched = matched_daily
            else:
                raise RuntimeError(
                    "Hourly temperature matching failed, and daily fallback also failed. "
                    "This usually means the temperature series does not cover all SAR acquisition dates."
                )

    else:
        daily = temp_series.resample("D").mean()
        daily = daily.interpolate(method="time").ffill().bfill()

        if mode == "daily_mean":
            predictor = daily
        elif mode.startswith("rolling_") and mode.endswith("day"):
            window = int(mode.replace("rolling_", "").replace("day", ""))
            predictor = daily.rolling(window=window, min_periods=1).mean()
        else:
            raise ValueError(f"Unsupported temperature mode: {mode}")

        target_days = target_datetimes.normalize()
        matched = predictor.reindex(
            target_days,
            method="nearest",
            tolerance=pd.Timedelta(days=1),
        )

    if matched.isna().any():
        missing_count = int(matched.isna().sum())
        raise RuntimeError(
            f"{missing_count} acquisition dates could not be matched to temperature predictor "
            f"(mode={mode}, lag_days={lag_days}). "
            "Check that the temperature series covers the full InSAR date range."
        )

    return matched.to_numpy(), target_datetimes


def get_temperature(dates, point, args):
    """
    For the selected InSAR point, get the temperature data,
    match it to the SAR acquisition dates using the selected mode,
    convert it to anomaly, and return it for the regression.
    """
    datetimes = make_datetimes(dates, args.hour)

    if args.temperature_csv is not None:
        temp_series = load_temperature_series_from_csv(args.temperature_csv)
    else:
        lat = args.lat if args.lat is not None else point["Y"]
        lon = args.lon if args.lon is not None else point["X"]
        temp_series = download_temperature_series(
            datetimes,
            lat,
            lon,
            lag_days=args.temperature_lag_days,
        )

    temp_C, matched_temperature_datetimes = match_temperature_predictor(
        temp_series,
        datetimes,
        mode=args.temperature_mode,
        lag_days=args.temperature_lag_days,
    )

    temp_anom_C = temp_C - np.nanmean(temp_C)

    return datetimes, temp_C, temp_anom_C, matched_temperature_datetimes


def invert_linear(t, T, y, use_thermal=True):
    """
    Estimate the best-fitting straight-line deformation trend,
    optionally include a temperature-related displacement term.
    d(t) = intercept + slope * time + thermal_coefficient(alpha) * temperature_anomaly
    d(t) = β0 + β1 * t + α T'
    """
    if use_thermal:
        G = np.column_stack([np.ones_like(t), t, T])
    else:
        G = np.column_stack([np.ones_like(t), t])

    m, _, _, _ = np.linalg.lstsq(G, y, rcond=None)
    pred = G @ m

    params = {
        "intercept_mm": m[0],
        "slope_mm_yr": m[1],
        "thermal_coeff_mm_C": m[2] if use_thermal else np.nan,
    }
    return pred, params


def invert_quadratic(t, T, y, use_thermal=True):
    """
    Second-order polynomial deformation trend,
    optionally include a temperature-related displacement term.
    d(t) = intercept + slope * time + quadratic_term * time^2 + thermal_coefficient(alpha) * temperature_anomaly
    d(t) = β0 + β1 * t + β2 * t^2 + α T'
    """
    if use_thermal:
        G = np.column_stack([np.ones_like(t), t, t**2, T])
    else:
        G = np.column_stack([np.ones_like(t), t, t**2])

    m, _, _, _ = np.linalg.lstsq(G, y, rcond=None)
    pred = G @ m

    params = {
        "intercept_mm": m[0],
        "slope_mm_yr": m[1],
        "quadratic_term_mm_yr2": m[2],
        "thermal_coeff_mm_C": m[3] if use_thermal else np.nan,
    }
    return pred, params


def exp_func_thermal(xdata, intercept, A, tau, alpha):
    """
    Mathematical form of the exponential deformation model with a thermal term
    d(t) = intercept + exponential amplitude * (1 - exp(-t / tau)) + thermal_coefficient(alpha) * temperature_anomaly
    d(t) = β0 + A * (1 - exp(-t / τ)) + α T'
    xdata is a tuple of (t, T) where t is time and T is temperature anomaly. This allows curve_fit to pass both arrays together.
    """
    t, T = xdata
    return intercept + A * (1.0 - np.exp(-t / tau)) + alpha * T


def exp_func_no_thermal(t, intercept, A, tau):
    """
    Mathematical form of the exponential deformation model without a thermal term
    d(t) = intercept + exponential amplitude * (1 - exp(-t / tau))
    d(t) = β0 + A * (1 - exp(-t / τ))
    """
    return intercept + A * (1.0 - np.exp(-t / tau))


def invert_exp(t, T, y, tau_upper, use_thermal=True):
    """
    Fit the exponential model to InSAR displacement data, with or without a thermal term.
    Exponential decay model / first-order exponential approach model.
    d(t) = β₀ + A(1 - exp(-t / τ))
    d(t) = β₀ + A(1 - exp(-t / τ)) + α T' (with thermal term)
    """
    if use_thermal:
        p0 = [0.0, y[-1], 2.0, 1.0]
        lower = [-np.inf, -10000.0, 0.05, -100.0]
        upper = [np.inf, 10000.0, tau_upper, 100.0]

        # non-linear inversion, curve_fit() tries to find the parameters that make
        # exp_func_thermal((t, T), intercept, A, tau, alpha) match the observed displacement y.
        # popt means unpacks the parameters.
        popt, _ = curve_fit(
            exp_func_thermal,
            (t, T),
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50000,
        )

        pred = exp_func_thermal((t, T), *popt)
        params = {
            "intercept_mm": popt[0],
            "exp_amplitude_mm": popt[1],
            "tau_years": popt[2],
            "thermal_coeff_mm_C": popt[3],
        }
    else:
        p0 = [0.0, y[-1], 2.0]
        lower = [-np.inf, -10000.0, 0.05]
        upper = [np.inf, 10000.0, tau_upper]

        # exp_func_no_thermal(t, intercept, A, tau)
        popt, _ = curve_fit(
            exp_func_no_thermal,
            t,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50000,
        )

        pred = exp_func_no_thermal(t, *popt)
        params = {
            "intercept_mm": popt[0],
            "exp_amplitude_mm": popt[1],
            "tau_years": popt[2],
            "thermal_coeff_mm_C": np.nan,
        }

    return pred, params


def invert_model(model, t, T, y, tau_upper, use_thermal=True):
    """
    Model dispatcher.
    Receive model name, then send the data to the correct inversion function.
    """

    model = model.lower()

    if model == "linear":
        return invert_linear(t, T, y, use_thermal=use_thermal)
    if model == "quadratic":
        return invert_quadratic(t, T, y, use_thermal=use_thermal)
    if model == "exp":
        return invert_exp(t, T, y, tau_upper, use_thermal=use_thermal)
    raise ValueError(f"Unsupported model: {model}. Use 'linear', 'quadratic', or 'exp'.")


def parameter_count(model, use_thermal=True):
    """
    Tell how many free parameters each model has, for the AIC/BIC calculation.
    """

    model = model.lower()

    if model == "linear":
        return 3 if use_thermal else 2
    if model == "quadratic":
        return 4 if use_thermal else 3
    if model == "exp":
        return 4 if use_thermal else 3
    raise ValueError(f"Unsupported model: {model}. Use 'linear', 'quadratic', or 'exp'.")


def compute_metrics(y, yhat, model, use_thermal=True):
    """
    Evaluate how well a fitted model matches the observed InSAR displacement time series.
     - residual
     - RMSE
     - R²
     - RSS
     - AIC
     - BIC
     """
    residual = y - yhat
    n = len(y)
    k = parameter_count(model, use_thermal=use_thermal)

    rmse = np.sqrt(np.mean(residual**2))
    rss = np.sum(residual**2)
    ss_res = rss
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    rss_safe = max(rss, np.finfo(float).eps)

    aic = n * np.log(rss_safe / n) + 2 * k
    bic = n * np.log(rss_safe / n) + k * np.log(n)

    return residual, rmse, r2, rss, aic, bic


def format_model_name(model):
    """
    Convert short internal model names into more readable names.
    """
    names = {"linear": "Linear", 
             "quadratic": "Quadratic", 
             "exp": "Exponential",
    }
    return names.get(model, model)


def result_key(model, use_thermal):
    """
    Create internal key used to store model results.
     - linear_thermal
     - linear_no_thermal
     - quadratic_thermal
     - quadratic_no_thermal
     - exp_thermal
     - exp_no_thermal
    """
    return f"{model}_{'thermal' if use_thermal else 'no_thermal'}"


def model_label(model, use_thermal):
    """
    Create clean label for plots and tables.
    """
    base = format_model_name(model)
    return f"{base} + thermal" if use_thermal else f"{base} only"


def compare_models(t, T, y, tau_upper):
    """
    Test all model options and decide which one fits the InSAR time series best after accounting for model complexity.
    This is the core function that runs all the inversions and collects the results.
    """
    rows = []
    results = {}

    for model in ["linear", "quadratic", "exp"]:
        for use_thermal in [False, True]:
            key = result_key(model, use_thermal)
            pred, params = invert_model(model, t, T, y, tau_upper, use_thermal=use_thermal)
            residual, rmse, r2, rss, aic, bic = compute_metrics(
                y, pred, model, use_thermal=use_thermal
            )

            warning = ""
            if model == "exp" and params["tau_years"] > 0.95 * tau_upper:
                warning = "tau_near_upper_bound"

            row = {
                "model_key": key,
                "inversion_model": model,
                "thermal_term": use_thermal,
                "model_label": model_label(model, use_thermal),
                "num_parameters": parameter_count(model, use_thermal=use_thermal),
                "RMSE_mm": rmse,
                "R2": r2,
                "RSS": rss,
                "AIC": aic,
                "BIC": bic,
                "warning": warning,
            }

            for pkey, val in params.items():
                row[pkey] = val

            rows.append(row)
            results[key] = {
                "model": model,
                "use_thermal": use_thermal,
                "label": model_label(model, use_thermal),
                "pred": pred,
                "params": params,
                "residual": residual,
                "rmse": rmse,
                "r2": r2,
                "rss": rss,
                "aic": aic,
                "bic": bic,
                "warning": warning,
            }

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("BIC").reset_index(drop=True)
    comparison["rank_by_BIC"] = np.arange(1, len(comparison) + 1)

    return comparison, results


def summarize_thermal_improvement(comparison):
    """
    For each deformation model, compare "without thermal" vs "with thermal"
    and calculate how much the thermal term improves RMSE, AIC, and BIC.
     - delta_RMSE = RMSE_no_thermal - RMSE_with_thermal (positive means thermal is better)
     - delta_AIC = AIC_no_thermal - AIC_with_thermal (positive means thermal is better)
     - delta_BIC = BIC_no_thermal - BIC_with_thermal (positive means thermal is better)
    """
    rows = []

    for model in ["linear", "quadratic", "exp"]:
        no = comparison[comparison["model_key"] == result_key(model, False)].iloc[0]
        th = comparison[comparison["model_key"] == result_key(model, True)].iloc[0]

        rows.append({
            "inversion_model": model,
            "RMSE_no_thermal": no["RMSE_mm"],
            "RMSE_with_thermal": th["RMSE_mm"],
            "delta_RMSE_no_minus_with": no["RMSE_mm"] - th["RMSE_mm"],
            "AIC_no_thermal": no["AIC"],
            "AIC_with_thermal": th["AIC"],
            "delta_AIC_no_minus_with": no["AIC"] - th["AIC"],
            "BIC_no_thermal": no["BIC"],
            "BIC_with_thermal": th["BIC"],
            "delta_BIC_no_minus_with": no["BIC"] - th["BIC"],
            "thermal_coeff_mm_C": th["thermal_coeff_mm_C"],
            "thermal_preferred_by_AIC": th["AIC"] < no["AIC"],
            "thermal_preferred_by_BIC": th["BIC"] < no["BIC"],
        })

    return pd.DataFrame(rows)


def add_textbox(ax, text, loc="outside upper right"):
    """
    Write clean text box either outside the plot area or inside the plot area.
    """
    fig = ax.figure

    if loc == "outside upper right":
        fig.text(
            0.735, 0.88, text,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92)
        )
        return

    if loc == "outside lower right":
        fig.text(
            0.735, 0.56, text,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92)
        )
        return

    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92)
    )


def build_point_text(point):
    """
    Prepare short text summary of the selected InSAR point for plotting.
    """
    lines = [
        f"Point ID: {point['point_id']}",
        f"Velocity: {point['velocity']:.3f} mm/yr",
        f"Coherence: {point['coherence']:.3f}",
        f"Lon: {point['X']:.6f}",
        f"Lat: {point['Y']:.6f}",
    ]
    return "\n".join(lines)


def plot_observed_timeseries(dates, y, point, outdir):
    """
    Show the raw displacement history of the selected InSAR point before fitting any model.
    """
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.scatter(dates, y, s=12, color="black", alpha=0.85, label="Observed InSAR displacement")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("LOS displacement (mm)")
    ax.set_title(f"Observed InSAR LOS displacement time series — Point {point['point_id']}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    add_textbox(ax, build_point_text(point), loc="outside upper right")
    fig.subplots_adjust(right=0.70, bottom=0.24)
    fig.savefig(outdir / "01_observed_timeseries.png", dpi=300)
    plt.close(fig)


def plot_model_comparison(dates, y, point, comparison, results, outdir):
    """
    Show the observed InSAR displacement together with three thermal model fits.
    """
    best_key_bic = comparison.iloc[0]["model_key"]
    best_key_aic = comparison.sort_values("AIC").iloc[0]["model_key"]

    fig, ax = plt.subplots(figsize=(13.5, 6))
    ax.scatter(dates, y, s=10, alpha=0.85, color="black", label="Observed InSAR displacement")

    style = {
        "linear_thermal": {"color": "red", "linestyle": "-", "label": "Linear + thermal"},
        "quadratic_thermal": {"color": "purple", "linestyle": "--", "label": "Quadratic + thermal"},
        "exp_thermal": {"color": "green", "linestyle": "-.", "label": "Exponential + thermal"},
    }

    for key in ["linear_thermal", "quadratic_thermal", "exp_thermal"]:
        ax.plot(
            dates,
            results[key]["pred"],
            linewidth=2.0,
            color=style[key]["color"],
            linestyle=style[key]["linestyle"],
            label=style[key]["label"],
        )

    text = (
        f"Best model by BIC: {results[best_key_bic]['label']}\n"
        f"Best model by AIC: {results[best_key_aic]['label']}\n"
        f"Point ID: {point['point_id']}"
    )

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("LOS displacement (mm)")
    ax.set_title("Observed InSAR data model comparison: thermal models")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    add_textbox(ax, text, loc="outside upper right")
    fig.subplots_adjust(right=0.70, bottom=0.24)
    fig.savefig(outdir / "02_model_comparison.png", dpi=300)
    plt.close(fig)


def plot_thermal_vs_no_thermal(dates, y, comparison, results, outdir):
    """
    Show whether adding the thermal term visibly improves each deformation model.
    """
    fig, ax = plt.subplots(figsize=(13.5, 6))
    ax.scatter(dates, y, s=9, alpha=0.75, color="black", label="Observed InSAR displacement")

    style = {
        "linear_no_thermal": {"color": "red", "linestyle": ":", "label": "Linear only"},
        "linear_thermal": {"color": "red", "linestyle": "-", "label": "Linear + thermal"},
        "quadratic_no_thermal": {"color": "purple", "linestyle": ":", "label": "Quadratic only"},
        "quadratic_thermal": {"color": "purple", "linestyle": "--", "label": "Quadratic + thermal"},
        "exp_no_thermal": {"color": "green", "linestyle": ":", "label": "Exponential only"},
        "exp_thermal": {"color": "green", "linestyle": "-.", "label": "Exponential + thermal"},
    }

    for key in [
        "linear_no_thermal", "linear_thermal",
        "quadratic_no_thermal", "quadratic_thermal",
        "exp_no_thermal", "exp_thermal",
    ]:
        ax.plot(
            dates,
            results[key]["pred"],
            linewidth=1.7,
            color=style[key]["color"],
            linestyle=style[key]["linestyle"],
            label=style[key]["label"],
        )

    best_key_bic = comparison.iloc[0]["model_key"]
    best_key_aic = comparison.sort_values("AIC").iloc[0]["model_key"]

    text = (
        f"Best by BIC: {results[best_key_bic]['label']}\n"
        f"Best by AIC: {results[best_key_aic]['label']}"
    )

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("LOS displacement (mm)")
    ax.set_title("Thermal vs no-thermal model comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    add_textbox(ax, text, loc="outside upper right")
    fig.subplots_adjust(right=0.70, bottom=0.28)
    fig.savefig(outdir / "08_thermal_vs_no_thermal.png", dpi=300)
    plt.close(fig)


def compute_fitted_thermal_and_corrected(y, T, alpha):
    """
    Compute the fitted thermal component and the thermally corrected
    displacement time series using one thermal coefficient alpha (while keeping the long-term deformation trend).

    Important:
    We remove only the thermal term, not the deformation trend term.

    The thermal component is anchored to the first acquisition so that
    correction does not introduce an arbitrary vertical shift:
        fitted_thermal = alpha * T
        fitted_thermal = fitted_thermal - fitted_thermal[0]
        corrected = y - fitted_thermal
    """
    if np.isfinite(alpha):
        fitted_thermal = alpha * T
        fitted_thermal = fitted_thermal - fitted_thermal[0]
        corrected = y - fitted_thermal
    else:
        fitted_thermal = np.zeros_like(y, dtype=float)
        corrected = y.copy()

    return fitted_thermal, corrected


def plot_thermal_correction(dates, y, T, point, best_key, results, outdir):
    """
    Show how the observed time series changes after removing temperature-correlated displacement component from best model.
    """
    params = results[best_key]["params"]
    alpha = params.get("thermal_coeff_mm_C", np.nan)

    fitted_thermal, corrected = compute_fitted_thermal_and_corrected(y, T, alpha)

    if np.isfinite(alpha):
        correction_label = "Thermal-corrected displacement using best-model thermal coefficient"
    else:
        correction_label = "No thermal term in best model; corrected displacement equals observed"

    fig, ax = plt.subplots(figsize=(13.5, 6))
    ax.scatter(dates, y, s=10, alpha=0.75, color="black", label="Observed InSAR displacement")
    ax.plot(dates, corrected, linewidth=2.0, color="green", label=correction_label)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("LOS displacement (mm)")
    ax.set_title(f"Thermal correction using {results[best_key]['label']}")

    alpha_text = f"{alpha:.3f} mm/°C" if np.isfinite(alpha) else "N/A"
    text = (
        f"Best model: {results[best_key]['label']}\n"
        f"Thermal coefficient: {alpha_text}\n"
        f"Point ID: {point['point_id']}"
    )

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    add_textbox(ax, text, loc="outside upper right")
    fig.subplots_adjust(right=0.70, bottom=0.24)
    fig.savefig(outdir / "03_thermal_correction_best_model.png", dpi=300)
    plt.close(fig)

    return fitted_thermal, corrected


def get_thermal_corrected_displacement(y, T, result):
    """
    Return thermally corrected displacement for one fitted thermal model.

    This removes ONLY the thermal component from the observed displacement:
        corrected = observed - alpha * T_anomaly

    It does NOT remove the deformation trend. This is the correct quantity for
    checking whether seasonal/thermal oscillations are reduced while preserving
    the long-term deformation signal.
    """
    alpha = result["params"].get("thermal_coeff_mm_C", np.nan)
    return compute_fitted_thermal_and_corrected(y, T, alpha)


def plot_fits_and_corrected_subplot(dates, y, T, point, comparison, results, outdir, offset_mm=30.0):
    """
    Create a 2-panel figure for one point.

    Top panel:
      - observed displacement
      - total fitted model curves:
          linear + thermal
          quadratic + thermal
          exponential + thermal

    Bottom panel:
      - original observed displacement
      - thermally corrected displacement from each thermal model
      - each series is vertically offset so the dots do not overlap

    Important:
      The bottom panel shows full displacement time series, not residuals.
      The correction removes ONLY the fitted thermal component alpha*T_anomaly.
      The long-term deformation trend is intentionally preserved.
    """
    best_key_bic = comparison.iloc[0]["model_key"]
    best_key_aic = comparison.sort_values("AIC").iloc[0]["model_key"]

    keys = ["linear_thermal", "quadratic_thermal", "exp_thermal"]

    style = {
        "observed": {
            "color": "black",
            "linestyle": "-",
            "label": "Observed displacement",
        },
        "linear_thermal": {
            "color": "red",
            "linestyle": "-",
            "label_fit": "Linear + thermal fit",
            "label_corrected": "Linear corrected",
        },
        "quadratic_thermal": {
            "color": "purple",
            "linestyle": "--",
            "label_fit": "Quadratic + thermal fit",
            "label_corrected": "Quadratic corrected",
        },
        "exp_thermal": {
            "color": "green",
            "linestyle": "-.",
            "label_fit": "Exponential + thermal fit",
            "label_corrected": "Exponential corrected",
        },
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 10), sharex=True)

    # =========================================================
    # Top panel: observed data + total fitted curves
    # =========================================================
    ax1.scatter(
        dates,
        y,
        s=12,
        color=style["observed"]["color"],
        alpha=0.85,
        label=style["observed"]["label"],
    )

    for key in keys:
        ax1.plot(
            dates,
            results[key]["pred"],
            color=style[key]["color"],
            linestyle=style[key]["linestyle"],
            linewidth=2.0,
            label=style[key]["label_fit"],
        )

    ax1.axhline(0, linewidth=0.8)
    ax1.set_ylabel("LOS displacement (mm)")
    ax1.set_title("Observed displacement and fitted deformation+thermal models")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    text = (
        f"Best model by BIC: {results[best_key_bic]['label']}\n"
        f"Best model by AIC: {results[best_key_aic]['label']}\n"
        f"Point ID: {point['point_id']}"
    )
    add_textbox(ax1, text, loc="outside upper right")

    # =========================================================
    # Bottom panel: original and thermally corrected displacement
    # =========================================================
    offsets = {
        "observed": 0.0,
        "linear_thermal": -offset_mm,
        "quadratic_thermal": -2.0 * offset_mm,
        "exp_thermal": -3.0 * offset_mm,
    }

    # Original observed displacement
    ax2.scatter(
        dates,
        y + offsets["observed"],
        s=12,
        color=style["observed"]["color"],
        alpha=0.85,
        label=f"Observed displacement",
    )
    ax2.plot(
        dates,
        y + offsets["observed"],
        color=style["observed"]["color"],
        linewidth=1.0,
        alpha=0.45,
    )

    # Thermally corrected displacement for each model
    for key in keys:
        alpha = results[key]["params"].get("thermal_coeff_mm_C", np.nan)
        _, corrected = get_thermal_corrected_displacement(y, T, results[key])

        if np.isfinite(alpha):
            label = f"{style[key]['label_corrected']} (α={alpha:.3f} mm/°C)"
        else:
            label = style[key]["label_corrected"]

        ax2.scatter(
            dates,
            corrected + offsets[key],
            s=12,
            color=style[key]["color"],
            alpha=0.85,
            label=label,
        )
        ax2.plot(
            dates,
            corrected + offsets[key],
            color=style[key]["color"],
            linewidth=1.0,
            alpha=0.45,
        )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("LOS displacement (mm)")
    ax2.set_title("Observed vs thermally corrected displacement")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    fig.subplots_adjust(right=0.70, bottom=0.18, hspace=0.42)
    fig.savefig(outdir / "09_fits_and_corrected_subplot.png", dpi=300)
    plt.close(fig)


def plot_detrended_thermal_residual_check(dates, results, outdir):
    """
    Create a 3-panel diagnostic plot showing whether the thermal term reduces
    seasonal/temperature-correlated residuals after removing the long-term
    deformation component.

    For each model family:
      Before thermal correction:
          residual_no_thermal = observed - deformation_only_model

      After thermal correction:
          residual_with_thermal = observed - deformation_plus_thermal_model

    This is different from the full corrected displacement plot. Long-term deformation trend has already been removed,
    so remaining seasonal structure is easier to see.
    """

    pairs = [
        {
            "title": "Linear model",
            "no_key": "linear_no_thermal",
            "thermal_key": "linear_thermal",
        },
        {
            "title": "Quadratic model",
            "no_key": "quadratic_no_thermal",
            "thermal_key": "quadratic_thermal",
        },
        {
            "title": "Exponential model",
            "no_key": "exp_no_thermal",
            "thermal_key": "exp_thermal",
        },
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10), sharex=True)

    for ax, pair in zip(axes, pairs):
        no_key = pair["no_key"]
        thermal_key = pair["thermal_key"]

        residual_no = results[no_key]["residual"]
        residual_th = results[thermal_key]["residual"]

        rmse_no = results[no_key]["rmse"]
        rmse_th = results[thermal_key]["rmse"]
        delta_rmse = rmse_no - rmse_th

        ax.scatter(
            dates,
            residual_no,
            s=10,
            color="gray",
            alpha=0.75,
            label=f"Without thermal term (RMSE={rmse_no:.2f} mm)",
        )
        ax.plot(
            dates,
            residual_no,
            color="gray",
            linewidth=0.9,
            alpha=0.45,
        )

        ax.scatter(
            dates,
            residual_th,
            s=10,
            color="blue",
            alpha=0.75,
            label=f"With thermal term (RMSE={rmse_th:.2f} mm)",
        )
        ax.plot(
            dates,
            residual_th,
            color="blue",
            linewidth=0.9,
            alpha=0.45,
        )

        ax.axhline(0, linewidth=0.8, color="red")
        ax.set_ylabel("Residual (mm)")
        ax.set_title(
            f"{pair['title']}: detrended residuals before and after thermal term"
        )

        text = (
            f"RMSE without thermal: {rmse_no:.3f} mm\n"
            f"RMSE with thermal: {rmse_th:.3f} mm\n"
            f"ΔRMSE: {delta_rmse:.3f} mm"
        )

        ax.text(
            0.02,
            0.95,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
        )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    axes[-1].set_xlabel("Date")

    fig.suptitle(
        "Detrended residual check: effect of adding the thermal term",
        fontsize=14,
        y=0.98,
    )

    fig.subplots_adjust(top=0.92, bottom=0.10, hspace=0.55)
    fig.savefig(outdir / "10_detrended_thermal_residual_check.png", dpi=300)
    plt.close(fig)


def plot_residuals(dates, results, outdir):
    """
    Save separate residual plots for all six fitted models:

      *_no_thermal.png  = deformation-only residuals
      *_thermal.png     = deformation + thermal residuals
    """

    key_to_file = {
        "linear_no_thermal": "04_residuals_linear_no_thermal.png",
        "linear_thermal": "04_residuals_linear_thermal.png",
        "quadratic_no_thermal": "05_residuals_quadratic_no_thermal.png",
        "quadratic_thermal": "05_residuals_quadratic_thermal.png",
        "exp_no_thermal": "06_residuals_exponential_no_thermal.png",
        "exp_thermal": "06_residuals_exponential_thermal.png",
    }

    for key, filename in key_to_file.items():
        residual = results[key]["residual"]

        fig, ax = plt.subplots(figsize=(13.5, 4.5))
        ax.scatter(
            dates, residual,
            s=10,
            color="black",
            alpha=0.85,
            label="Residual = observed - fitted model"
        )
        ax.axhline(0, linewidth=1.0, color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Residual (mm)")
        ax.set_title(f"Residuals of {results[key]['label']} model")

        text = (
            f"RMSE: {results[key]['rmse']:.3f} mm\n"
            f"Mean residual: {np.mean(residual):.3f} mm\n"
            f"Residual std: {np.std(residual):.3f} mm\n"
            f"Warning: {results[key]['warning'] or 'None'}"
        )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
        add_textbox(ax, text, loc="outside upper right")
        fig.subplots_adjust(right=0.70, bottom=0.28)
        fig.savefig(outdir / filename, dpi=300)
        plt.close(fig)


def plot_temperature(dates, temp_C, temp_anom_C, outdir, temperature_mode="hourly", temperature_lag_days=0.0):
    """
    Show which temperature signal was used in the thermal regression.
     - temp_C: the actual temperature predictor (e.g., hourly, daily, or lagged)
     - temp_anom_C: the temperature anomaly used in the regression (temp_C minus its mean)
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, temp_C, color="red", linewidth=1.5, label="Temperature predictor")
    ax.plot(dates, temp_anom_C, color="blue", linewidth=1.5, label="Temperature anomaly")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature / anomaly (°C)")
    ax.set_title("Matched acquisition temperature predictor")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    text = (
        f"Temperature mode: {temperature_mode}\n"
        f"Temperature lag: {temperature_lag_days:g} days\n"
        f"Mean predictor: {np.nanmean(temp_C):.2f} °C\n"
        f"Anomaly range: {np.nanmin(temp_anom_C):.2f} to {np.nanmax(temp_anom_C):.2f} °C"
    )
    add_textbox(ax, text, loc="outside upper right")
    fig.subplots_adjust(right=0.70, bottom=0.24)
    fig.savefig(outdir / "07_temperature.png", dpi=300)
    plt.close(fig)


def save_outputs(dates, datetimes, y, temp_C, temp_anom_C, point, comparison, results,
                 fitted_thermal, corrected, outdir, thermal_summary,
                 temperature_mode="hourly", temperature_lag_days=0.0,
                 matched_temperature_datetimes=None):
    """
    Save fitted time series, model comparison table, thermal improvement table, and estimated parameters to CSV files.
    """

    out = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "datetime": datetimes,
        "t_years": (dates - dates[0]).days.to_numpy() / 365.25,
        "observed_mm": y,
        "temperature_C": temp_C,
        "temperature_anomaly_C": temp_anom_C,
        "temperature_mode": temperature_mode,
        "temperature_lag_days": temperature_lag_days,
        "best_model_fitted_thermal_mm": fitted_thermal,
        "thermal_corrected_mm": corrected,
    })

    if matched_temperature_datetimes is not None:
        out["matched_temperature_datetime"] = matched_temperature_datetimes
    
    for key in ["linear_thermal", "quadratic_thermal", "exp_thermal"]:
        thermal_component, corrected_model = get_thermal_corrected_displacement(
            y, temp_anom_C, results[key]
        )
        out[f"{key}_component_mm"] = thermal_component
        out[f"{key}_corrected_mm"] = corrected_model


    for key, result in results.items():
        out[f"{key}_model_mm"] = result["pred"]
        out[f"{key}_residual_mm"] = result["residual"]

    for key, value in point.items():
        out[key] = value

    out.to_csv(outdir / "observed_point_timeseries_fitted.csv", index=False)
    comparison.to_csv(outdir / "model_comparison.csv", index=False)
    thermal_summary.to_csv(outdir / "thermal_improvement_summary.csv", index=False)

    param_rows = []
    for key, result in results.items():
        row = {
            "model_key": key,
            "inversion_model": result["model"],
            "thermal_term": result["use_thermal"],
            "model_label": result["label"],
        }
        row.update(result["params"])
        row["RMSE_mm"] = result["rmse"]
        row["R2"] = result["r2"]
        row["AIC"] = result["aic"]
        row["BIC"] = result["bic"]
        row["warning"] = result["warning"]
        param_rows.append(row)

    pd.DataFrame(param_rows).to_csv(outdir / "estimated_parameters_all_models.csv", index=False)


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    dates, t, y, point = load_point(
        args.csv,
        point_id=args.point_id,
        select_most_negative=args.select_most_negative,
        min_coherence=args.min_coherence,
    )

    datetimes, temp_C, temp_anom_C, matched_temperature_datetimes = get_temperature(dates, point, args)

    comparison, results = compare_models(t, temp_anom_C, y, args.exp_tau_upper)
    best_key = comparison.iloc[0]["model_key"]
    thermal_summary = summarize_thermal_improvement(comparison)

    plot_observed_timeseries(dates, y, point, outdir)
    plot_model_comparison(dates, y, point, comparison, results, outdir)
    plot_thermal_vs_no_thermal(dates, y, comparison, results, outdir)

    fitted_thermal, corrected = plot_thermal_correction(
        dates, y, temp_anom_C, point, best_key, results, outdir
    )

    plot_fits_and_corrected_subplot(
        dates,
        y,
        temp_anom_C,
        point,
        comparison,
        results,
        outdir,
        offset_mm=30.0,
    )

    plot_detrended_thermal_residual_check(dates, results, outdir)

    plot_residuals(dates, results, outdir)

    plot_temperature(
        dates,
        temp_C,
        temp_anom_C,
        outdir,
        temperature_mode=args.temperature_mode,
        temperature_lag_days=args.temperature_lag_days,
    )

    save_outputs(
        dates, datetimes, y, temp_C, temp_anom_C, point,
        comparison, results, fitted_thermal, corrected, outdir, thermal_summary,
        temperature_mode=args.temperature_mode,
        temperature_lag_days=args.temperature_lag_days,
        matched_temperature_datetimes=matched_temperature_datetimes,
    )

    print("\nDone.")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Selected point_id: {point['point_id']}")
    print(f"Point velocity from CSV: {point['velocity']:.3f} mm/yr")
    print(f"Point coherence from CSV: {point['coherence']:.3f}")
    print(f"Temperature mode: {args.temperature_mode}")
    print(f"Temperature lag days: {args.temperature_lag_days:g}")
    print(f"Best model by BIC: {results[best_key]['label']}")

    print("\nModel comparison:")
    cols = [
        "rank_by_BIC", "model_key", "model_label", "num_parameters",
        "RMSE_mm", "R2", "AIC", "BIC", "warning"
    ]
    print(comparison[cols].to_string(index=False))

    print("\nThermal improvement summary:")
    print(thermal_summary.to_string(index=False))

    print("\nEstimated parameters:")
    params_df = pd.read_csv(outdir / "estimated_parameters_all_models.csv")
    print(params_df.to_string(index=False))


if __name__ == "__main__":
    main()

