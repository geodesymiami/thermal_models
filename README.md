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
```
## Usage:

The repository has test_data (CSVs of Ritz Carlton and Porsche Design)

```
fit_thermal_models.py --help 
fit_thermal_models.py --csv test_data/RitzCarlton/TSX_036_20170923_20251008_N2592W08012_N2592W08012_N2592W08012_N2592W08012.csv --point-id 18582 --temperature-mode rolling_3day --out results/RitzCarlton_point18582_rolling3day
```

For choosing different points you can look them from insarmaps links;
Ritz Carlton:

https://insarmaps.miami.edu/start/25.9224/-80.1225/16.6014?flyToDatasetCenter=false&startDataset=TSX_036_20170923_20251008_N2592W08012_N2592W08012_N2592W08012_N2592W08012&minScale=-1&maxScale=1

Porsche Design Tower:

https://insarmaps.miami.edu/start/25.9475/-80.1196/16.8810?flyToDatasetCenter=false&startDataset=TSX_036_20170923_20251008_N2595W08012_N2595W08012_N2595W08012_N2595W08012&minScale=-1&maxScale=1
```
