"""
Malaria Risk Index (MRI) - Comparative Evaluation Framework
============================================================

This script requires only the Uganda base XGBoost model (.pkl)
and the Zimbabwe data CSV. It splits the Zimbabwe data into fine-tune,
validation, and test sets, then trains a residual booster internally using
warm-start boosting. The best number of trees is selected on the validation
set to prevent data leakage into the held-out test period. Both the baseline
Uganda model and the fine-tuned model are then evaluated on the test set and
compared. The residual booster is saved for reuse. All comparison outputs
are written to OUTPUT_DIR: the booster pickle, the n_trees search log, global
metrics for both approaches, per-location and per-time metrics with delta
columns showing the gain from fine-tuning, and a six-panel comparison plot.

Zimbabwe has three survey rounds, 2005, 2010, and 2015, and every location
is surveyed in only one of them, with 12 months of data. The rounds are
therefore used in time order: 2005 to fit the residual booster, 2010 to
select the number of trees, and 2015 to test, so no location appears in
more than one set.

The Uganda model predicts the risk HORIZON rows ahead from lag and rolling
features, which xg_load_and_prepare_data builds with a shift by row position
within each location. The same features and target are built here, so the
Uganda model is evaluated on the inputs it was trained on. FEATURE_MODE
chooses how: 'row_order' repeats the Uganda training script exactly on the
file as it is, and 'calendar' builds true monthly lags on one row per
location and month, for a model that was trained that way.

"""

import configparser
import os
import sys
import pickle
import joblib
import warnings
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
logging.basicConfig(level = logging.INFO,
    format = '%(asctime)s  %(levelname)s  %(message)s',
    datefmt = '%H:%M:%S')

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')
log = logging.getLogger(__name__)

MODEL_PATH     = os.path.join(DATA_RESULTS, 'xgboost', 'UGA_dhs', 'xgb_model.pkl')
TEST_DATA_PATH = os.path.join(DATA_PROCESSED, 'ZWE_dhs', 'file7_ZWE_malaria_monthly_risk_covariates.csv')
TARGET_COLUMN  = 'monthly_mri'
OUTPUT_DIR     = os.path.join(DATA_RESULTS, 'ZWE_validation', 'xgboost_comparative')
os.makedirs(OUTPUT_DIR, exist_ok = True)

# Data split by survey round. Each location is surveyed in one round only,
# so a split by year is also a split by location: nothing is shared.
# Time order is kept: the earliest round trains the residual booster, the
# next one selects the number of trees, and the latest is held out.
FINETUNE_YEARS = [2005]   # used to train the residual booster
VAL_YEARS      = [2010]   # used to select best n_trees (no data leakage)
TEST_YEARS     = [2015]   # held-out, never touched during fine-tuning

# Residual booster hyperparameters
BOOSTER_LEARNING_RATE = 0.01
BOOSTER_MAX_DEPTH     = 4
BOOSTER_SUBSAMPLE     = 0.8
BOOSTER_COLSAMPLE     = 0.8

# Candidate tree counts to search over
BOOSTER_N_TREES_GRID = [100, 200, 300, 400, 500, 600, 750,
    1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]

# Spatial / temporal identifier columns
ID_COLUMNS = ['latitude', 'longitude', 'year', 'month', 'month_num']

FEATURE_COLUMNS = None

# Environmental covariates kept in 'calendar' mode, where the rows are
# reduced to one per location and month. A feature the model expects that
# is not here is reported as missing.
COVARIATE_COLUMNS = ['ndvi', 'precipitation_mm', 'temperature_C',
    'elevation_m']

# History features and target, as built by xg_load_and_prepare_data in the
# Uganda training script. Keep them identical to it.
LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6]
EVAL_TARGET = 'target'

# The number of steps ahead the Uganda model was trained to predict. It is
# the horizon passed to xg_load_and_prepare_data when that model was
# trained, so it cannot be worked out here and must be set.
HORIZON = 6

# 'row_order' : repeat the Uganda training script on the file as it is,
#               shifting by row position within each location.
# 'calendar'  : one row per location and month, shifted in calendar order
#               within each location and year.
FEATURE_MODE = 'row_order'
FEATURE_MODES = ('row_order', 'calendar')


def load_pickle(path, label):
    """
    This function loads a pickle file.

    Parameters    
    ----------
    path : str
        The path to the pickle file.
    label : str
        A label for the object being loaded.

    Returns
    -------
    object
        The loaded object.
    """
    if not os.path.exists(path):

        log.error('%s not found: %s', label, path)
        sys.exit(1)
    with open(path, 'rb') as f:

        obj = pickle.load(f)
    log.info('%s loaded  ->  %s', label, path)


    return obj


def load_data(path):

    """
    This function loads a CSV or Excel file 
    into a pandas DataFrame.

    Parameters
    ----------
    path : str
        The path to the CSV or Excel file.

    Returns
    -------
    pandas.DataFrame
        The loaded data.
    """
    if not os.path.exists(path):

        log.error('Data file not found: %s', path)
        sys.exit(1)
    df = (pd.read_excel(path) if path.endswith(('.xlsx', '.xls'))
          else pd.read_csv(path, sep = None, engine = 'python'))
    
    log.info('Data loaded  ->  %s  (%d rows x %d cols)', path, *df.shape)


    return df


def get_model_feature_names(model):

    """
    Extracts the feature names from a trained model.

    Parameters
    ----------
    model : object
        The trained model.

    Returns
    -------
    list of str
        The feature names.
    """
    if hasattr(model, 'named_steps'):

        return get_model_feature_names(list(model.named_steps.values())[-1])
    if hasattr(model, 'feature_names_in_'):

        return list(model.feature_names_in_)
    try:

        names = model.get_booster().feature_names
        if names:

            return names
    except Exception:

        pass


    return None

 
def resolve_features(df, target, id_cols, feature_cols, model):

    """
    Resolves the feature columns to be used for training or prediction.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    target : str
        The target column name.
    id_cols : list of str
        The identifier column names.
    feature_cols : list of str or None
        The feature column names. If None, features will be auto-detected.
    model : object
        The trained model.

    Returns
    -------
    list of str
        The resolved feature column names.
    """

    if feature_cols is not None:

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:

            log.error('Supplied feature columns missing from data: %s', missing)
            sys.exit(1)
        log.info('Using supplied features (%d): %s', len(feature_cols), feature_cols)
        return feature_cols
    
    names = get_model_feature_names(model)
    if names:

        missing = [c for c in names if c not in df.columns]
        if missing:

            log.error('Model expects features missing from data: %s', missing)
            sys.exit(1)
        log.info('Features from model (%d): %s', len(names), names)
        return names
    
    exclude = set(id_cols) | {target, EVAL_TARGET}
    auto = [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    log.warning('Auto-detected features (%d): %s', len(auto), auto)


    return auto


def compute_metrics(y_true, y_pred):
    """"
    Computes regression metrics between true and predicted values.
    Parameters
    ----------
    y_true : array-like
        The true values.
    y_pred : array-like
        The predicted values.

    Returns
    -------
    dict
        A dictionary containing the computed metrics.
    """
    return {
        'R2':   r2_score(y_true, y_pred),
        'MSE':  mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'N':    int(len(y_true)),
    }


def metrics_per_group(df, group_cols, 
    obs = 'observed', pred = 'predicted'):

    """
    Computes regression metrics per group defined by group_cols.

    Parameters    
    ----------
    df : pandas.DataFrame
        The input data.
    group_cols : list of str
        The column names to group by.
    obs : str, default "observed"
        The column name for the observed values.
    pred : str, default "predicted"
        The column name for the predicted values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the metrics for each group.
    """
    records = []
    for keys, grp in df.groupby(group_cols):

        if not isinstance(keys, tuple):

            keys = (keys,)
        m = compute_metrics(grp[obs].values, grp[pred].values)
        records.append({**dict(zip(group_cols, keys)), **m})


    return pd.DataFrame(records)


def prepare_data(df, target):

    """
    Reduces the data to one row per location, year, and month.

    The file holds one row per child and month, but the children at one
    location share the raster values, so the covariates are identical
    and the target, the risk of each child, is averaged. A file that
    already has one row per location and month passes through unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    target : str
        The target column name.

    Returns
    -------
    pandas.DataFrame
        One row per location and month, sorted by location, year, and
        month. Raises ValueError when a required column is missing, or
        when a covariate has more than one value at one location and
        month, since the mean would then hide a difference.
    """

    keys = ['longitude', 'latitude', 'year', 'month_num']
    absent = [c for c in keys + [target] if c not in df.columns]
    if absent:
        raise ValueError(f'the data is missing column(s) {absent}')

    covariates = [c for c in COVARIATE_COLUMNS if c in df.columns]
    spread = df.groupby(keys)[covariates].nunique()
    if (spread > 1).to_numpy().any():
        raise ValueError(
            'a covariate has more than one value at one location and month'
        )

    aggregations = {c: 'mean' for c in covariates + [target]}
    if 'month' in df.columns:
        aggregations['month'] = 'first'

    out = df.groupby(keys, as_index=False).agg(aggregations)
    out = out.sort_values(keys).reset_index(drop=True)

    log.info('One row per location and month  ->  %d rows (%d locations)',
             len(out), len(out[['longitude', 'latitude']].drop_duplicates()))
    return out


def build_features(df, series, horizon, mode):

    """
    Builds the history features and the target the Uganda model uses.

    The features are those of xg_load_and_prepare_data: the series shifted
    by each of LAGS, its rolling mean over each of ROLL_WINDOWS, which
    includes the current row, and the series shifted back by horizon as
    the target. Which rows count as earlier depends on the mode.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    series : str
        The monthly risk column.
    horizon : int
        The number of steps ahead the model predicts.
    mode : str
        'row_order' shifts by row position within each location, in the
        order of the file, as the Uganda training script does. In the
        monthly file that is child by child, each child's months in
        alphabetical order, so a lag is not a calendar lag. 'calendar'
        first reduces the data to one row per location and month, then
        shifts in month order within each location and year.

    Returns
    -------
    pandas.DataFrame
        The data with the lag, rolling, and target columns added, not yet
        cleaned of missing values. Raises ValueError for an unknown mode,
        or when horizon is not a positive whole number, since a wrong
        horizon would evaluate the model against the wrong target.
    """

    if mode not in FEATURE_MODES:
        raise ValueError(f'FEATURE_MODE must be one of {FEATURE_MODES}, '
                         f'got {mode!r}')
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(
            'set HORIZON to the number of steps ahead the Uganda model was '
            'trained to predict, the horizon passed to '
            f'xg_load_and_prepare_data; got {horizon!r}')

    if mode == 'calendar':
        df = prepare_data(df, series)
        keys = ['longitude', 'latitude', 'year']
    else:
        df = df.copy()
        keys = ['longitude', 'latitude']
        log.warning(
            'Features follow the row order of the file within each location, '
            'as in the Uganda training script, not calendar order. Months '
            'run in this order for the first location: %s',
            list(df['month'].head(12)) if 'month' in df.columns else 'n/a')

    grouped = df.groupby(keys)[series]
    for lag in LAGS:
        df[f'mri_lag{lag}'] = grouped.shift(lag)
    for window in ROLL_WINDOWS:
        df[f'mri_roll{window}'] = grouped.transform(
            lambda x: x.rolling(window).mean())
    df[EVAL_TARGET] = grouped.shift(-horizon)

    log.info('Built lags %s, rolling windows %s, and the target %d step(s) '
             'ahead  (mode %s)', LAGS, ROLL_WINDOWS, horizon, mode)
    return df


def split_by_year(df, finetune_years, val_years, test_years):

    """
    Splits the data into fine-tune, validation, and test sets by year.

    Parameters
    ----------
    df : pandas.DataFrame
        The prepared data, with a year column.
    finetune_years : list
        Years used to train the residual booster.
    val_years : list
        Years used to select the number of trees.
    test_years : list
        Years held out for testing.

    Returns
    -------
    tuple
        The fine-tune, validation, and test DataFrames. Raises
        ValueError when a year is assigned to two sets, or when a set
        has no rows, naming the years the data holds.
    """

    groups = {'fine-tune': finetune_years,
              'validation': val_years,
              'test': test_years}

    assigned = {}
    for name, years in groups.items():
        for year in years:
            if year in assigned:
                raise ValueError(
                    f'year {year} is in both the {assigned[year]} and '
                    f'{name} sets')
            assigned[year] = name

    available = sorted(df['year'].unique().tolist())
    parts = {}
    for name, years in groups.items():
        part = df[df['year'].isin(years)].copy()
        if part.empty:
            raise ValueError(
                f'no rows for the {name} years {list(years)}; the data '
                f'holds the years {available}')
        parts[name] = part

    places = {name: set(zip(p['longitude'], p['latitude']))
              for name, p in parts.items()}
    log.info('Locations  ->  fine-tune %d | val %d | test %d | shared '
             'fine-tune/val %d, fine-tune/test %d, val/test %d',
             len(places['fine-tune']), len(places['validation']),
             len(places['test']),
             len(places['fine-tune'] & places['validation']),
             len(places['fine-tune'] & places['test']),
             len(places['validation'] & places['test']))

    return parts['fine-tune'], parts['validation'], parts['test']


def train_residual_booster(model, ft_df, val_df, features, target, out_dir):
    """
    Trains a residual booster on the fine-tune split.
    Selects best n_trees using validation split (no test data leakage).
    Saves the booster and returns it.

    Parameters
    ----------
    model : object
        The Uganda base model.
    ft_df : pandas.DataFrame
        The fine-tune rows.
    val_df : pandas.DataFrame
        The validation rows.
    features : list of str
        The feature columns.
    target : str
        The target column name.
    out_dir : str
        The directory the booster and the search log are saved to.

    Returns
    -------
    tuple
        The best booster, its number of trees, and its validation R2.
    """

    X_ft,  y_ft  = ft_df[features],  ft_df[target].values
    X_val, y_val = val_df[features], val_df[target].values

    log.info('Fine-tune split : %d rows  (years %s)',
             len(ft_df), FINETUNE_YEARS)
    log.info('Validation split: %d rows  (years %s)',
             len(val_df), VAL_YEARS)

    residuals = y_ft - model.predict(X_ft)
    log.info('Residual stats  mean=%.4f  std=%.4f  min=%.4f  max=%.4f',
             residuals.mean(), residuals.std(),
             residuals.min(),  residuals.max())

    log.info('%-10s  %-10s  %-10s  %-10s  %s',
             'n_trees', 'val_R2', 'val_RMSE', 'val_MAE', 'note')
    log.info("-" * 58)

    search_rows  = []
    best_val_r2  = -np.inf
    best_n       = None
    best_booster = None
    prev_val_r2  = -np.inf

    for n in BOOSTER_N_TREES_GRID:

        rb = XGBRegressor(n_estimators = n, learning_rate = BOOSTER_LEARNING_RATE,
            max_depth = BOOSTER_MAX_DEPTH, subsample = BOOSTER_SUBSAMPLE,
            colsample_bytree = BOOSTER_COLSAMPLE, tree_method = 'hist')
        
        rb.fit(X_ft, residuals)

        val_pred = model.predict(X_val) + rb.predict(X_val)
        val_r2   = r2_score(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae  = mean_absolute_error(y_val, val_pred)
        gain     = val_r2 - prev_val_r2

        note = ""
        if gain < 0.005:

            note += 'plateau '
        if val_r2 < best_val_r2 - 0.005:
            note += 'degrading'

        log.info('%-10d  %-10.4f  %-10.4f  %-10.4f  %s',
                 n, val_r2, val_rmse, val_mae, note)

        search_rows.append({
            'n_trees': n, 'val_R2': val_r2,
            'val_RMSE': val_rmse, 'val_MAE': val_mae, 'gain': gain
        })

        if val_r2 > best_val_r2:
            best_val_r2  = val_r2
            best_n       = n
            best_booster = rb

        prev_val_r2 = val_r2

    log.info("Best n_trees=%d  val_R2=%.4f", best_n, best_val_r2)

    pd.DataFrame(search_rows).to_csv(
        os.path.join(out_dir, 'finetuning_search.csv'), index = False)

    booster_path = os.path.join(out_dir, 'xgb_residual_booster_zwe.pkl')
    joblib.dump(best_booster, booster_path) 

    return best_booster, best_n, best_val_r2


def plot_comparison(res_base, res_ft, gm_base, gm_ft, out_path):

    """
    Plot a comparison of the baseline and fine-tuned models.
    Parameters
    ----------
    res_base : pd.DataFrame
        The baseline results.
    res_ft : pd.DataFrame
        The fine-tuned results.
    gm_base : dict
        The global metrics for the baseline.
    gm_ft : dict
        The global metrics for the fine-tuned model.
    out_path : str
        The path to save the comparison plot.

    Returns
    -------
    None
    """
    colors = {'baseline': '#E07B39', 'finetuned': '#2176AE'}
    labels = {'baseline': 'Baseline (Uganda only)',
              'finetuned': 'Fine-tuned (warm-start)'}

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 2, figure = fig, 
                            hspace = 0.40, wspace = 0.30)

    # Panels 1 & 2 - Obs vs Pred scatter
    for col_idx, (tag, res, gm) in enumerate([
            ('baseline', res_base, gm_base),
            ('finetuned', res_ft,  gm_ft)]):
        
        ax = fig.add_subplot(gs[0, col_idx])
        obs  = res['observed'].values
        pred = res['predicted'].values
        ax.scatter(obs, pred, alpha = 0.15, s = 5, 
                   color = colors[tag])
        lims = [min(obs.min(), pred.min()) - 0.02,
                max(obs.max(), pred.max()) + 0.02]
        ax.plot(lims, lims, "r--", lw=1.2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Observed MRI'); ax.set_ylabel('Predicted MRI')
        ax.set_title(
            f"{labels[tag]}\nR2 = {gm['R2']:.4f}  RMSE = {gm['RMSE']:.4f}")

    # Panel 3 - Residual distributions overlaid
    ax3 = fig.add_subplot(gs[1, 0])
    for tag, res in [('baseline', res_base), ('finetuned', res_ft)]:

        resid = res['observed'].values - res['predicted'].values
        ax3.hist(resid, bins = 60, alpha = 0.55, color = colors[tag],
                 label = labels[tag], edgecolor = 'none')
    ax3.axvline(0, color = 'black', lw = 1.2, linestyle = '--')
    ax3.set_xlabel('Residual (obs - pred)'); ax3.set_ylabel('Count')
    ax3.set_title('Residual Distribution Comparison'); ax3.legend()

    # Panel 4 - Temporal mean MRI
    ax4 = fig.add_subplot(gs[1, 1])
    if 'year' in res_base.columns and 'month_num' in res_base.columns:

        obs_trend = (res_base.groupby(['year', 'month_num'])['observed']
                     .mean().reset_index()
                     .sort_values(['year', 'month_num']))
        ax4.plot(range(len(obs_trend)), obs_trend['observed'],
                 label = 'Observed', color ='black', lw = 1.8)
        for tag, res in [('baseline', res_base), ('finetuned', res_ft)]:
            trend = (res.groupby(['year', 'month_num'])['predicted']
                     .mean().reset_index()
                     .sort_values(['year', 'month_num']))
            ax4.plot(range(len(trend)), trend['predicted'],
                     label = labels[tag], color = colors[tag], 
                     lw = 1.5, linestyle = "--")
        ax4.set_xlabel('Time step'); ax4.set_ylabel('Mean MRI')
        ax4.set_title('Temporal Trend Comparison'); ax4.legend()

    # Panel 5 - Bar chart of global metrics
    ax5 = fig.add_subplot(gs[2, 0])
    metric_names = ['R2', 'RMSE', 'MAE']
    x     = np.arange(len(metric_names))
    width = 0.35
    vals_base = [gm_base[m] for m in metric_names]
    vals_ft   = [gm_ft[m]   for m in metric_names]
    bars1 = ax5.bar(x - width/2, vals_base, width,
        label = labels['baseline'], color = colors['baseline'], alpha = 0.8)
    bars2 = ax5.bar(x + width/2, vals_ft,   width,
        label = labels['finetuned'], color = colors['finetuned'], alpha = 0.8)
    ax5.set_xticks(x); ax5.set_xticklabels(metric_names)
    ax5.set_title('Global Metric Comparison'); ax5.legend()
    ax5.axhline(0, color ='black', lw = 0.8)
    for bars in [bars1, bars2]:

        for bar in bars:

            h = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2,
                     h + (0.01 if h >= 0 else -0.04),
                     f'{h:.3f}', ha = 'center', 
                     va = 'bottom', fontsize = 8)

    # Panel 6 - Per-location R2 scatter (baseline vs finetuned)
    ax6 = fig.add_subplot(gs[2, 1])
    loc_base = metrics_per_group(res_base, ['latitude', 'longitude'])
    loc_ft   = metrics_per_group(res_ft,   ['latitude', 'longitude'])
    merged   = loc_base.merge(loc_ft, on=['latitude', 'longitude'],
                               suffixes=('_base', '_ft'))
    ax6.scatter(merged['R2_base'], merged['R2_ft'],
                alpha = 0.3, s = 8, color = '#6B4E71')
    
    lims = [min(merged['R2_base'].min(), merged['R2_ft'].min()) - 0.05,
            max(merged['R2_base'].max(), merged['R2_ft'].max()) + 0.05]
    ax6.plot(lims, lims, 'r--', lw = 1.2, label = 'No-change line')
    ax6.set_xlabel('Baseline R2 (per location)')
    ax6.set_ylabel('Fine-tuned R2 (per location)')
    ax6.set_title('Per-Location R2: Baseline vs Fine-tuned'); ax6.legend()

    fig.suptitle(
        'MRI Model Comparison - Baseline vs Fine-tuned (Warm-Start Boosting)\n'
        f'Test year(s): {", ".join(str(y) for y in TEST_YEARS)}',
        fontsize = 14, fontweight = 'bold')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)


def run_single(df, model, features, target, id_cols,
               residual_booster, approach_label):

    X      = df[features]
    y_true = df[target].values
    y_pred = model.predict(X)

    if residual_booster is not None:

        correction = residual_booster.predict(X)
        y_pred     = y_pred + correction

    present_id = [c for c in id_cols if c in df.columns]
    results = df[present_id].copy()
    results['observed']  = y_true
    results['predicted'] = y_pred
    results['residual']  = y_true - y_pred
    results['abs_error'] = np.abs(results['residual'])
    results['approach']  = approach_label

    gm = compute_metrics(y_true, y_pred)

    loc_cols = [c for c in ['latitude', 'longitude'] if c in results.columns]
    per_loc  = metrics_per_group(results, loc_cols) if loc_cols else pd.DataFrame()

    time_cols = [c for c in ['year', 'month_num'] if c in results.columns]
    per_time  = metrics_per_group(results, time_cols) if time_cols else pd.DataFrame()

    return results, gm, per_loc, per_time


def save_comparison_csvs(gm_base, gm_ft,
                         per_loc_base, per_loc_ft,
                         per_time_base, per_time_ft,
                         out_dir, res_base=None, res_ft=None):
  
    """Saves CSV files comparing the baseline and fine-tuned
      models at global, per-location, and per-time levels.

    Parameters
    ----------
    gm_base : dict
        Global metrics for the baseline model.
    gm_ft : dict
        Global metrics for the fine-tuned model.
    per_loc_base : pd.DataFrame
        Per-location metrics for the baseline model.
    per_loc_ft : pd.DataFrame
        Per-location metrics for the fine-tuned model.
    per_time_base : pd.DataFrame
        Per-time metrics for the baseline model.
    per_time_ft : pd.DataFrame
        Per-time metrics for the fine-tuned model.
    out_dir : str
        Output directory to save the CSV files.
    Returns
    -------
    None
    """
    pd.DataFrame([
        {'approach': 'baseline',  **gm_base},
        {'approach': 'finetuned', **gm_ft},
    ]).to_csv(os.path.join(out_dir, 
    'comparison_global_metrics.csv'), index = False)

# Per-location (with delta columns)
    if not per_loc_base.empty and not per_loc_ft.empty:

        loc_merge = per_loc_base.merge(
            per_loc_ft, on = ['latitude', 'longitude'],
            suffixes = ('_baseline', '_finetuned'))
        for m in ['R2', 'RMSE', 'MAE']:
            loc_merge[f"{m}_delta"] = (loc_merge[f'{m}_finetuned']
                                       - loc_merge[f'{m}_baseline'])
        if res_base is not None and res_ft is not None:
            for tag, res in [('baseline', res_base), ('finetuned', res_ft)]:
                obs_pred = res.groupby(['latitude', 'longitude'])[
                    ['observed', 'predicted']].mean().reset_index()
                obs_pred = obs_pred.rename(columns={
                    'observed':  f'observed_{tag}',
                    'predicted': f'predicted_{tag}'})
                loc_merge = loc_merge.merge(obs_pred, on=['latitude', 'longitude'])
        loc_merge.to_csv(
            os.path.join(out_dir, 'comparison_per_location.csv'), index = False)

    if not per_time_base.empty and not per_time_ft.empty:

        time_merge = per_time_base.merge(
            per_time_ft, on = ['year', 'month_num'],
            suffixes = ('_baseline', '_finetuned'))
        for m in ['R2', 'RMSE', 'MAE']:
            time_merge[f'{m}_delta'] = (time_merge[f'{m}_finetuned']
                                        - time_merge[f'{m}_baseline'])
        if res_base is not None and res_ft is not None:
            for tag, res in [('baseline', res_base), ('finetuned', res_ft)]:
                obs_pred = res.groupby(['year', 'month_num'])[
                    ['observed', 'predicted']].mean().reset_index()
                obs_pred = obs_pred.rename(columns={
                    'observed':  f'observed_{tag}',
                    'predicted': f'predicted_{tag}'})
                time_merge = time_merge.merge(obs_pred, on=['year', 'month_num'])
        time_merge.to_csv(
            os.path.join(out_dir, 'comparison_per_time.csv'), index = False)

def run_comparative_evaluation(
        model_path   = MODEL_PATH,
        data_path    = TEST_DATA_PATH,
        target       = TARGET_COLUMN,
        id_cols      = None,
        feature_cols = FEATURE_COLUMNS,
        output_dir   = OUTPUT_DIR):

    """
    Fits the residual booster on the fine-tune years, selects its size
    on the validation years, and compares the Uganda base model with
    and without the booster on the test years.

    Parameters
    ----------
    model_path : str
        The Uganda base model pickle.
    data_path : str
        The Zimbabwe data CSV.
    target : str
        The target column name.
    id_cols : list of str or None
        The identifier columns copied into the results.
    feature_cols : list of str or None
        The feature columns. When None they are taken from the model.
    output_dir : str
        The directory all outputs are written to.

    Returns
    -------
    None
    """

    if id_cols is None:

        id_cols = ID_COLUMNS

    os.makedirs(output_dir, exist_ok = True)
    model = load_pickle(model_path, 'Base model')
    df    = load_data(data_path)

    if target not in df.columns:
        log.error('Column %s not found. Available: %s', target, list(df.columns))
        sys.exit(1)

    # History features and the target, as in the Uganda training script
    df = build_features(df, target, HORIZON, FEATURE_MODE)

    # Feature engineering
    if 'month_sin' not in df.columns and 'month_num' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        log.info('Engineered month_sin / month_cos')

    features = resolve_features(df, target, id_cols, feature_cols, model)

    # Only the columns the model uses must be present. The Uganda script
    # dropped a row with a missing value in any column, which here would
    # drop every row, since malaria_rdt_result is empty for Zimbabwe.
    before = len(df)
    df = df.dropna(subset = features + [EVAL_TARGET]).reset_index(drop = True)
    log.info('Rows with a missing feature or target dropped: %d of %d',
             before - len(df), before)

    ft_df, val_df, test_df = split_by_year(
        df, FINETUNE_YEARS, VAL_YEARS, TEST_YEARS)
    test_df = test_df.reset_index(drop = True)

    # ── STEP 1: Train residual booster ────────────────────────────────────
    log.info('=' * 60)
    log.info('STEP 1 - TRAINING RESIDUAL BOOSTER (warm-start)')
    log.info('  Fine-tune : %s  |  Validation : %s  |  Test : %s',
             FINETUNE_YEARS, VAL_YEARS, TEST_YEARS)
    log.info('=' * 60)

    residual_booster, best_n, best_val_r2 = train_residual_booster(
        model, ft_df, val_df, features, EVAL_TARGET, output_dir)

    log.info("=" * 60)
    log.info("STEP 2 — APPROACH 1: BASELINE (no adaptation)")
    log.info("=" * 60)
    res_base, gm_base, loc_base, time_base = run_single(
        test_df, model, features, EVAL_TARGET, id_cols,
        residual_booster = None,
        approach_label   = 'baseline')
    
    log.info("=" * 60)
    log.info('STEP 3 — APPROACH 2: FINE-TUNED (n_trees=%d)', best_n)
    log.info("=" * 60)
    res_ft, gm_ft, loc_ft, time_ft = run_single(
        test_df, model, features, EVAL_TARGET, id_cols,
        residual_booster = residual_booster,
        approach_label   = 'finetuned')
    
    # ── STEP 3: Save comparison files ─────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 4 — SAVING COMPARISON FILES")
    log.info("=" * 60)
    save_comparison_csvs(gm_base, gm_ft, loc_base, loc_ft,
                            time_base, time_ft, output_dir,
                            res_base=res_base, res_ft=res_ft)
    plot_comparison(res_base, res_ft, gm_base, gm_ft,
                    os.path.join(output_dir, 'comparison_plots.png'))

if __name__ == "__main__":

    run_comparative_evaluation(
        model_path   = MODEL_PATH,
        data_path    = TEST_DATA_PATH,
        target       = TARGET_COLUMN,
        feature_cols = FEATURE_COLUMNS,
        output_dir   = OUTPUT_DIR)
