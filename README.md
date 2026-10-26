# Digital Resilience Ecosystem for Adaptive Malaria Control (dream)

[![tests](https://github.com/Bonface-Osoro/dream/actions/workflows/tests.yml/badge.svg)](https://github.com/Bonface-Osoro/dream/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Bonface-Osoro/dream/branch/main/graph/badge.svg)](https://codecov.io/gh/Bonface-Osoro/dream)
![License](https://img.shields.io/github/license/Bonface-Osoro/dream)

Climate change is fundamentally altering malaria transmission dynamics in ways that traditional surveillance systems are not equipped to detect. Recent resurgence patterns in East Africa show that even where core interventions remain in place, ecological shifts such as changes in dominant mosquito vector species can rapidly create new transmission regimes. As warming temperatures expand malaria suitability into highland and previously low-risk areas, outbreaks increasingly occur in immunologically naïve populations. Because most forecasting models rely on historical data, they struggle to anticipate these novel conditions, leaving health systems in a reactive cycle of rising cases followed by emergency response.

Here, we develop a climate-adaptive malaria early response framework designed to close that “prevention gap”, (the lag between environmental change and public health action). By integrating ecological intelligence, climate data, AI-driven prediction, and predefined response triggers, we convert emerging risk signals into anticipatory intervention. Built for operational use rather than academic modeling alone, the framework also generates decision-ready economic evidence, including cost per case and cost per DALY averted, aligned with Global Fund and PMI investment criteria. The goal is to enable scalable, climate-resilient malaria control across highland East Africa and beyond.

Specific Objectives
===================
1.	Characterize Climate-Driven Ecological Regime Shifts.
2.	Develop a Climate-Adaptive Early Warning System Through Ecological Deep Learning.
3.	Translate Predictions into Anticipatory Action.



Methodology
==============
Integrated model applied in the study(see `Figure 1`). 

#### Figure 1 Proposed Method.
<p align="center">
  <img src="/docs/method.png" />
</p>

Results
=======
`Figure 2` shows the Malaria Risk Index (MRI) for Uganda from 2009 to 2020.
#### Figure 2 Modeling Preliminary Results.
<p align="center">
  <img src="/docs/mri.png" />
</p>

## Folder structure

Every script reads `base_path = data` from `scripts/script_config.ini` and resolves all other folders from it, so run the scripts from the repository root or from `scripts/`. Folders marked (input) must be filled by you. All other folders are created by the scripts.

```
dream/
├── data/                                  (inputs, not tracked by Git)
│   ├── raw/
│   │   ├── UGA/ ZMB/ ZWE/
│   │   │   ├── dhs/MIS_DHS_datasets.zip   (input) DHS/MIS survey archives, nested zips are allowed
│   │   │   └── gps_surv/                  (input) DHS GPS cluster archives
│   │   ├── global_netcdf_ndvi/            (input) GIMMS-3G+ NDVI .nc4 files [2]
│   │   ├── chirps_tif/                    (input) CHIRPS monthly .tif files [3]
│   │   ├── elevation/                     (input) NASADEM raster [5]
│   │   ├── shapefiles/                    (input) GADM shapefiles [6]
│   │   └── *.nc                           (input) ERA5 monthly temperature [4]
│   └── arcgis_processed/                  written by ArcGIS (Step 4), read by Python
│       └── UGA/ ZMB/ ZWE/
│           ├── ndvi/  precipitation/  temperature/   one <year>_<month>.xlsx per month
│           └── <ISO3>_full_results.csv    survey table joined to the covariates (Step 5)
├── results/
│   ├── processed/
│   │   ├── <ISO3>_dhs/                    file1 ... file7 tables (Steps 1, 2, 3, 5, 6)
│   │   └── covariates/<ISO3>/             combined monthly covariate tables (Step 5)
│   └── final/
│       ├── <ISO3>_dhs/                    validation of the risk index (Step 3)
│       ├── xgboost/UGA_dhs/  lstm/UGA_dhs/    models trained on Uganda (Step 7)
│       ├── ZMB_validation/  ZWE_validation/   transfer learning (Step 8)
│       └── federated/                     sovereignty constructs (Step 9)
├── scripts/                               entry points, run in the order below
├── src/dream/                             library code imported by the scripts
├── vis/                                   R scripts, figures are written to vis/figures/
└── docs/
```

`<ISO3>` is `UGA` (Uganda), `ZMB` (Zambia) or `ZWE` (Zimbabwe).

## Reproducing the results

Uganda is the source country and is processed first. Zambia and Zimbabwe are then processed with the same steps and used as transfer targets. Run the steps in order. Each step reads the files written by the step before it.

### Step 0. Set up the environment

```
pip install -e .
pip install pandas geopandas scikit-learn pymc arviz nutpie pytensor torch xgboost shap joblib matplotlib tqdm openpyxl
```

The R figures need `ggplot2`, `dplyr`, `tidyverse`, `sf`, `patchwork`, `scales`, `fields`, `ggnewscale`, `lubridate` and `rstudioapi`. The R scripts locate `vis/` through RStudio, so open them in RStudio and source them from there. Step 4 needs ArcGIS Pro with the Spatial Analyst extension. A GPU is optional but makes Steps 7 to 9 much faster.

Place the downloaded data in `data/raw/` as shown in the folder structure. If CHIRPS files are downloaded as `.tif.gz`, decompress them with `unzip_chirps_rasters` in `docs/helper_ai_codes/unzip_chirps.py`.

### Step 1. Preprocess the DHS survey data

`scripts/run_dhs.py` runs Steps 1 to 3, 5 and 6 as a sequence of function calls. Some calls are commented out so that each stage can be run on its own. Uncomment the stage you need, run it, then comment it out again. To process Zambia or Zimbabwe, replace `UGA` with `ZMB` or `ZWE` in the path constants at the top of the file.

Run this call (`src/dream/surv_index.py`):

```
build_malaria_index_table(DATA_RAW, OUTPUT_PATH)
```

It unpacks the nested DHS archives, selects the malaria related variables from the Individual and Household recodes, and joins them. Output: `results/processed/<ISO3>_dhs/file1_<ISO3>_malaria_survey_input.csv`.

### Step 2. Combine the survey data with the GPS cluster locations

```
merge_with_gps_clusters(OUTPUT_PATH, GPS_RAW, MERGED)
select_model_columns(MERGED, REFINED)
```

Outputs: `file2_<ISO3>_malaria_survey_with_gps.csv` (survey plus cluster coordinates) and `file3_<ISO3>_malaria_survey_model_input.csv` (columns kept for the index).

### Step 3. Build the annual Malaria Risk Index with PCA

```
build_risk_index(REFINED, RISK_INDEXED)
split_by_survey_year(RISK_INDEXED, RISK_INDEXED_BY_YEAR)
```

`build_risk_index` standardises the household and environment indicators and combines them with PCA into `malaria_risk_score` and `malaria_risk_category`. Output: `file4_<ISO3>_malaria_risk_index.csv`, and one file per survey year in `results/processed/<ISO3>_dhs/by_year/`.

Compare the annual risk categories with the measured RDT result later, in Step 6, once the covariates are attached.

### Step 4. Process the satellite covariates in ArcGIS

This step is manual and is done once per country. The geographic input is the set of survey points (`LONGNUM`, `LATNUM`) in the `by_year` files from Step 3.

1. Prepare the monthly rasters of each covariate for the survey years of the country.
   - NDVI: convert the GIMMS-3G+ `.nc4` files to rasters with `Make NetCDF Raster Layer` and aggregate the half month composites to monthly values.
   - Precipitation: the CHIRPS monthly `.tif` files.
   - Temperature: the ERA5 monthly files, from the `.nc` file with `Make NetCDF Raster Layer`.
   - Elevation: the NASADEM raster. It does not change with time, so it is extracted once.
2. Clip the rasters to the country boundary from `data/raw/shapefiles/`.
3. Add the survey points of each survey year as an XY event layer, then run `Extract Values to Points` for every month.
4. Export each table to Excel as `<year>_<month>.xlsx`, with the month in lower case three letters (`2009_jan.xlsx`). Keep the child identifier columns and `RASTERVALU`. Missing values stay as `-9999`, the scripts convert them to missing values.
5. Save the files in `data/arcgis_processed/<ISO3>/ndvi/`, `.../precipitation/` and `.../temperature/`.

### Step 5. Combine the covariates with the survey and GPS data

Uncomment and run these calls in `scripts/run_dhs.py` (`src/dream/covariates.py` and `src/dream/survGeo.py`):

```
combine_covariate_monthly(MONTHLY_NDVI, COMBINED, 'ndvi')
combine_covariate_monthly(MONTHLY_PRECIPITATION, COMBINED, 'precipitation')
combine_covariate_monthly(MONTHLY_TEMPERATURE, COMBINED, 'temperature')
merge_monthly_covariates(NDVI_ALL, PRECIPITATION_ALL, TEMPERATURE_ALL, ALL_COVARIATES, repeats = 'collapse')
merge_risk_with_outcome(RISK_INDEXED, REFINED, VALIDATION)
build_survey_covariate_table(VALIDATION, ALL_COVARIATES, SURV_COVARIATES, repeats = 'collapse')
```

Outputs: `results/processed/covariates/<ISO3>/<ISO3>_combined_monthly_*.csv`, `file5_<ISO3>_malaria_validation.csv` (risk index plus RDT result) and `data/arcgis_processed/<ISO3>/<ISO3>_full_results.csv` (one row per child and month, with NDVI, precipitation, temperature and elevation).

### Step 6. Downscale the annual index to a monthly index with the state space model

```
build_monthly_risk_table(SURV_COVARIATES, MONTHLY_COVARIATES)
validate_mri_categories(MONTHLY_COVARIATES, VALIDATION_RESULTS)
```

`build_monthly_risk_table` fits, for each survey year, a Bayesian state space (dynamic) model in PyMC that spreads the annual score over the 12 months using the monthly covariates. Sampling is slow, so run it on a machine with several cores. Outputs: `file7_<ISO3>_malaria_monthly_risk_covariates.csv` (the file read by all models) and `results/final/UGA_dhs/UGA_mri_validation_results.csv`, which holds the positivity rate of each risk category with Wilson confidence intervals.

Run Steps 1 to 6 for `UGA`, `ZMB` and `ZWE`. Step 8 and 9 need the `file7` table of all three.

### Step 7. Train XGBoost and LSTM on Uganda

```
python scripts/geo_xgboost.py
python scripts/geo_lstm.py
```

Both read `results/processed/UGA_dhs/file7_UGA_malaria_monthly_risk_covariates.csv` and train on survey years up to 2018 and test after 2018. Outputs:

- `results/final/xgboost/UGA_dhs/`: `xgb_model.pkl`, training log, test metrics, metrics per location, SHAP tables and plots.
- `results/final/lstm/UGA_dhs/`: `best_lstm_model.pt`, training log, test metrics, metrics per location, SHAP tables and plots.

### Step 8. Transfer learning to Zambia and Zimbabwe

```
python scripts/xg_boost_validate.py
python scripts/lstm_validate.py
```

Both load the Uganda model from Step 7 and fine-tune it on the target country. The target is set by `TEST_DATA_PATH` and `OUTPUT_DIR` near the top of each script. They are set to Zambia. To run Zimbabwe, change `ZMB` to `ZWE` in those two paths and run again. The XGBoost script fits a residual booster, the LSTM script compares four layer freezing strategies. Outputs: `results/final/ZMB_validation/` and `results/final/ZWE_validation/`, each with `xgboost_comparative/` and `lstm_comparative/` (global, per location and per time metrics, and comparison plots).

### Step 9. Quantify data sovereignty with federated learning

```
python scripts/federated.py
```

It trains the LSTM for Uganda, Zambia and Zimbabwe under five regimes (climatology, local, transfer, FedAvg, centralised) and computes the sovereignty cost, the federation gain and the exact Shapley contribution of each country. Outputs in `results/final/federated/`: `federated_results.csv`, `heterogeneity_mmd.csv`, `constructs.csv`, `shapley.csv` and the FedAvg round logs `rounds_fedavg_*.csv`. This step needs all three `file7` tables from Step 6.

### Step 10. Produce the figures

Open each script in RStudio and source it. Figures are written to `vis/figures/`.

| Order | Script | Reads | Figure |
|-------|--------|-------|--------|
| 1 | `vis/1_risk_index_validation.R` | Step 3 and 6 outputs, shapefiles | `1_risk_indices.png` |
| 2 | `vis/2_geoai_graphics.R` | Step 7 SHAP and LSTM outputs | `2_explainable_ai.png` |
| 3 | `vis/3_training.R` | Step 7 and 8 outputs | `2_training_plots.png`, `2_training_plots_1.png` |
| 4 | `vis/pp2_fig1_monthly_mri.R` | Zambia `file7` | `pp2_fig1_ZMB_monthly_mri.png` |
| 5 | `vis/pp2_fig2_mri_monthly_map.R` | Uganda and Zambia `file7`, shapefiles | `pp2_fig2_monthly_mri_anomaly_plot.png` and an animated GIF |
| 6 | `vis/sovereignity.R` | Step 6 and 9 outputs, shapefiles | `4_costs.png`, `performance.png`, `3_mri_maps.png` |
| 7 | `vis/uga_mri_timelapse.R` | Uganda monthly index | `UGA_mri_timelapse.gif` |

`vis/3_paper_training.R` is an earlier version of `vis/3_training.R`.

## Background and funding

**dream** has been developed by researchers at the Digital Epidemiology (**De**)
laboratory of the **Digital futures** at University of Cincinnati. 

We would like to thank the Geography and GIS Department and **Digital futures** for providing the funding for the project. 

## Required Data
[1] The Demographic and Health Surveys. (2026). The Demographic and Health Surveys (DHS) Program: Survey datasets (Uganda) [Data set]. https://dhsprogram.com/data/dataset_admin/index.cfm

[2] Pinzon, J. E., Pak, E. W., Tucker, C. J., Bhatt, U. S., Frost, G. V., & Macander, M. J. (2023). Global Vegetation Greenness (NDVI) from AVHRR GIMMS-3G+, 1981-2022 (Version 1). ORNL Distributed Active Archive Center. https://doi.org/10.3334/ORNLDAAC/2187 Date Accessed: 2026-04-02 (Download link: https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-global-veg-greenness-gimms-3g-2187-1)

[3] Funk, C., Peterson, P., Landsfeld, M. et al. The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. Sci Data 2, 150066 (2015). https://doi.org/10.1038/sdata.2015.66 (download link: https://data.chc.ucsb.edu/products/CHIRPS-2.0/)

[4] Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023): ERA5 monthly averaged data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.f17050d7 (Accessed on 08-Apr-2026) (Download link: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview)

[5] NASA JPL (2021). NASADEM Merged DEM Global 1 arc second V001. Distributed by OpenTopography. https://doi.org/10.5069/G93T9FD9. Accessed 2026-04-13 (Download link: https://portal.opentopography.org/rasterOutput?jobId=rt1776082869769)

[6] GADM administrative boundaries, version 4.1 (levels 0 to 4 for Uganda, levels 0 to 3 for Zambia and Zimbabwe). https://gadm.org
