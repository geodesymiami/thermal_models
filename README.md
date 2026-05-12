# thermal_models

Utilities for fitting deformation and temperature-correlated thermal models to observed InSAR point time series.

The main script, `fit_thermal_models.py`, fits deformation models to one selected InSAR point and evaluates whether adding a temperature-related term improves the fit.

## Overview

This repository is designed for post-processing observed InSAR displacement time series. The script reads an InSAR CSV file containing `DYYYYMMDD` displacement columns, selects one point, matches the acquisition dates with a temperature predictor, and fits six candidate models:

1. Linear only
2. Linear + thermal
3. Quadratic only
4. Quadratic + thermal
5. Exponential only
6. Exponential + thermal

The thermal term is modeled using temperature anomaly:

```text
T' = T - mean(T)
