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

MODEL_PATH     = os.path.join(DATA_RESULTS, 'xgboost', 'xgb_model.pkl')
TEST_DATA_PATH = os.path.join(DATA_RESULTS, 'mri', 'zimbabwe', 'ZWE_malaria_risk_index_monthly.csv')
TARGET_COLUMN  = 'monthly_mri'
OUTPUT_DIR     = os.path.join(DATA_RESULTS, 'zimbabwe_validation')
os.makedirs(OUTPUT_DIR, exist_ok = True)

# Data split years 
FINETUNE_YEARS = (2015, 2017)   # inclusive range used to train residual booster
VAL_YEARS      = (2018, 2019)   # used to select best n_trees (no data leakage)
TEST_YEARS     = (2020, 2022)   # held-out, never touched during fine-tuning

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
    
    exclude = set(id_cols) | {target}
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


def train_residual_booster(model, df, features, target, out_dir):
    """
    Trains a residual booster on the fine-tune split.
    Selects best n_trees using validation split (no test data leakage).
    Saves the booster and returns it.
    """
    ft_mask  = ((df['year'] >= FINETUNE_YEARS[0]) &
                (df['year'] <= FINETUNE_YEARS[1]))
    val_mask = ((df['year'] >= VAL_YEARS[0]) &
                (df['year'] <= VAL_YEARS[1]))

    ft_df  = df[ft_mask].copy()
    val_df = df[val_mask].copy()

    X_ft,  y_ft  = ft_df[features],  ft_df[target].values
    X_val, y_val = val_df[features], val_df[target].values

    log.info('Fine-tune split : %d rows  (%d-%d)',
             len(ft_df),  FINETUNE_YEARS[0], FINETUNE_YEARS[1])
    log.info('Validation split: %d rows  (%d-%d)',
             len(val_df), VAL_YEARS[0], VAL_YEARS[1])

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
        f'Test period: {TEST_YEARS[0]}-{TEST_YEARS[1]}',
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
                         out_dir):
  
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
        loc_merge.to_csv(
            os.path.join(out_dir, 'comparison_per_location.csv'), index = False)

    if not per_time_base.empty and not per_time_ft.empty:

        time_merge = per_time_base.merge(
            per_time_ft, on = ['year', 'month_num'],
            suffixes = ('_baseline', '_finetuned'))
        for m in ['R2', 'RMSE', 'MAE']:

            time_merge[f'{m}_delta'] = (time_merge[f'{m}_finetuned']
                                        - time_merge[f'{m}_baseline'])
        time_merge.to_csv(
            os.path.join(out_dir, 'comparison_per_time.csv'), index = False)


def run_comparative_evaluation(
        model_path   = MODEL_PATH,
        data_path    = TEST_DATA_PATH,
        target       = TARGET_COLUMN,
        id_cols      = None,
        feature_cols = FEATURE_COLUMNS,
        output_dir   = OUTPUT_DIR):

    if id_cols is None:

        id_cols = ID_COLUMNS

    os.makedirs(output_dir, exist_ok = True)
    model = load_pickle(model_path, 'Base model')
    df    = load_data(data_path)

    if 'monthly_mri' not in df.columns:
        log.error('Column monthly_mri not found. Available: %s', list(df.columns))
        sys.exit(1)

    # Feature engineering
    if 'month_sin' not in df.columns and 'month_num' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        log.info('Engineered month_sin / month_cos')

    df = df.dropna(subset = [target]).reset_index(drop = True)

    features = resolve_features(df, target, id_cols, feature_cols, model)

    # ── STEP 1: Train residual booster ────────────────────────────────────
    log.info('=' * 60)
    log.info('STEP 1 - TRAINING RESIDUAL BOOSTER (warm-start)')
    log.info('  Fine-tune : %d-%d  |  Validation : %d-%d  |  Test : %d-%d',
             FINETUNE_YEARS[0], FINETUNE_YEARS[1],
             VAL_YEARS[0],      VAL_YEARS[1],
             TEST_YEARS[0],     TEST_YEARS[1])
    log.info('=' * 60)

    residual_booster, best_n, best_val_r2 = train_residual_booster(
        model, df, features, target, output_dir)

    test_mask = ((df["year"] >= TEST_YEARS[0]) &
                 (df["year"] <= TEST_YEARS[1]))
    test_df   = df[test_mask].copy().reset_index(drop=True)

    if len(test_df) == 0:
        log.error('No test rows found for years %d-%d. Check TEST_YEARS in config.',
                  TEST_YEARS[0], TEST_YEARS[1])
        sys.exit(1)

    log.info("=" * 60)
    log.info("STEP 2 — APPROACH 1: BASELINE (no adaptation)")
    log.info("=" * 60)
    res_base, gm_base, loc_base, time_base = run_single(
        test_df, model, features, target, id_cols,
        residual_booster = None,
        approach_label   = 'baseline')
    
    log.info("=" * 60)
    log.info('STEP 3 — APPROACH 2: FINE-TUNED (n_trees=%d)', best_n)
    log.info("=" * 60)
    res_ft, gm_ft, loc_ft, time_ft = run_single(
        test_df, model, features, target, id_cols,
        residual_booster = residual_booster,
        approach_label   = 'finetuned')
    
    # ── STEP 3: Save comparison files ─────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 4 — SAVING COMPARISON FILES")
    log.info("=" * 60)
    save_comparison_csvs(gm_base, gm_ft, loc_base, loc_ft,
                         time_base, time_ft, output_dir)
    plot_comparison(res_base, res_ft, gm_base, gm_ft,
                    os.path.join(output_dir, 'comparison_plots.png'))

if __name__ == "__main__":

    run_comparative_evaluation(
        model_path   = MODEL_PATH,
        data_path    = TEST_DATA_PATH,
        target       = TARGET_COLUMN,
        feature_cols = FEATURE_COLUMNS,
        output_dir   = OUTPUT_DIR)
