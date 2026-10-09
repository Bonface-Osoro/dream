"""
Malaria Risk Index (MRI) - LSTM Comparative Evaluation Framework

This self-reliant script requires only the Uganda-trained LSTM model (.pt)
and the Zimbabwe data CSV. It fine-tunes the base model on Zimbabwe data
using four distinct layer-freezing strategies: freezing all LSTM layers and
retraining only the fully-connected head, freezing the first LSTM layer only,
freezing the second LSTM layer only, and retraining all layers end-to-end.
Each fine-tuned variant is evaluated on the same held-out test round and
compared against the baseline Uganda model. All comparison outputs are written
to OUTPUT_DIR: per-strategy metrics, per-location and per-time comparison
tables with delta columns, and a multi-panel comparison plot.

Zimbabwe has three survey rounds, 2005, 2010, and 2015, and every location
is surveyed in only one of them, with 12 months of data. The rounds are
therefore used in time order: 2005 to fine-tune, 2010 to validate and stop
early, and 2015 to test, so no location appears in more than one set.
"""

import configparser
import os
import sys
import csv
import copy
import warnings
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
logging.basicConfig(level = logging.INFO,
    format = '%(asctime)s  %(levelname)s  %(message)s',
    datefmt = '%H:%M:%S')

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')
DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
log = logging.getLogger(__name__)

MODEL_PATH     = os.path.join(DATA_RESULTS, 'lstm', 'UGA_dhs', 'best_lstm_model.pt')
TEST_DATA_PATH = os.path.join(DATA_PROCESSED, 'ZWE_dhs', 'file7_ZWE_malaria_monthly_risk_covariates.csv')
TARGET_COLUMN  = 'monthly_mri'
OUTPUT_DIR     = os.path.join(DATA_RESULTS, 'ZWE_validation', 'lstm_comparative')
os.makedirs(OUTPUT_DIR, exist_ok = True)

# Data split by survey round. Each location is surveyed in one round only,
# so a split by year is also a split by location: nothing is shared.
# Time order is kept: the earliest round fine-tunes, the next one
# validates, and the latest is held out for testing.
FINETUNE_YEARS = [2005]
VAL_YEARS      = [2010]
TEST_YEARS     = [2015]

INPUT_SIZE  = 7    
HIDDEN_SIZE = 64
NUM_LAYERS  = 2

# A location has 12 months, and 11 once the first month, which has no lag,
# is dropped. LOOK_BACK + HORIZON must therefore be 11 or less, and both
# should equal the values the Uganda model was trained with.
LOOK_BACK = 6
HORIZON   = 3

FEATURES = [
    'ndvi', 'precipitation_mm', 'temperature_C', 'elevation_m',
    'month_sin', 'month_cos', 'mri_lag1'
]

# Fine-tuning training parameters
FINETUNE_EPOCHS    = 50
FINETUNE_LR        = 0.0005
FINETUNE_BATCH     = 32
FINETUNE_PATIENCE  = 10

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FREEZE STRATEGIES
#     Each entry defines which parameter groups are frozen (requires_grad=False)
#     Key   : short label used in filenames and plots
#     Value : list of parameter name prefixes to freeze
# ─────────────────────────────────────────────────────────────────────────────

FREEZE_STRATEGIES = {
    'freeze_all_lstm':   ['lstm'],          # freeze all LSTM, retrain fc only
    'freeze_lstm_l0':    ['lstm.weight_ih_l0', 'lstm.weight_hh_l0',
                          'lstm.bias_ih_l0',   'lstm.bias_hh_l0'],   # freeze layer 0 only
    'freeze_lstm_l1':    ['lstm.weight_ih_l1', 'lstm.weight_hh_l1',
                          'lstm.bias_ih_l1',   'lstm.bias_hh_l1'],   # freeze layer 1 only
    'retrain_all':       [],                # no freezing — full fine-tune
}



class MRILSTM(nn.Module):
    '''LSTM model for predicting monthly Malaria Risk Index (MRI).'''

    def __init__(self, input_size, hidden_size = 64, num_layers = 2):

        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first = True, dropout = 0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_and_prepare(path):

    """
    Loads the CSV and prepares it for modelling.

    The file holds one row per child and month, but the children at one
    location share the raster values, so the rows are reduced to one per
    location, year, and month, taking the mean risk of the children
    there. month_sin, month_cos, and mri_lag1 are then added. Only the
    columns the model uses are kept, so a column that is empty for a
    country, such as malaria_rdt_result for Zimbabwe, cannot remove
    every row.

    Parameters
    ----------
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: One row per location and month, sorted by
        location, year, and month, without the first month of each
        location and year, which has no lag.
    """

    if not os.path.exists(path):
        log.error('Data file not found: %s', path)
        sys.exit(1)

    df = pd.read_csv(path, sep=None, engine='python')

    keys = ['longitude', 'latitude', 'year', 'month_num']
    covariates = ['ndvi', 'precipitation_mm', 'temperature_C', 'elevation_m']
    needed = keys + covariates + [TARGET_COLUMN]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise ValueError(f'{path} is missing column(s) {absent}')

    spread = df.groupby(keys)[covariates].nunique()
    if (spread > 1).to_numpy().any():
        raise ValueError(
            'a covariate has more than one value at one location and month'
        )

    df = df.groupby(keys, as_index=False)[covariates + [TARGET_COLUMN]].mean()
    df = df.sort_values(keys).reset_index(drop=True)

    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    # The lag stays inside one location and year, so the first month has none.
    df['mri_lag1'] = df.groupby(
        ['longitude', 'latitude', 'year'])[TARGET_COLUMN].shift(1)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_locations = len(df[['longitude', 'latitude']].drop_duplicates())

    log.info('Data loaded  ->  %d rows x %d cols, %d locations '
             '(%d row(s) without a lag or a value dropped)',
             *df.shape, n_locations, before - len(df))
    return df


def split_by_year(df, finetune_years, val_years, test_years):

    """
    Splits the data into fine-tune, validation, and test sets by year.

    Parameters
    ----------
        df : pd.DataFrame
            The prepared data, with a year column.
        finetune_years : list
            Years used to fine-tune.
        val_years : list
            Years used to validate and stop early.
        test_years : list
            Years held out for testing.

    Returns
    -------
        tuple: The fine-tune, validation, and test DataFrames. Raises
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


def scale_features(train_df, other_df, features):

    """
    Scales the specified features in train_df and other_df using
    MinMaxScaler, fitting one scaler on train_df and applying it to
    other_df.

    One scaler is used for all locations. A scaler per location cannot
    work here, because each location is surveyed in one year only, so a
    validation or test location never has a fine-tuning scaler, and a
    feature that is constant within a location, such as elevation, would
    be scaled to 0 everywhere.

    Parameters
    ----------
        train_df : pd.DataFrame
            The training DataFrame.
        other_df : pd.DataFrame
            The other DataFrame to scale.
        features : list
            The list of features to scale.

    Returns
    -------
        tuple: The scaled train and other DataFrames, and the fitted
        scaler.
    """

    sort_cols = ['longitude', 'latitude', 'year', 'month_num']
    train_scaled = train_df.sort_values(sort_cols).reset_index(drop=True)
    other_scaled = other_df.sort_values(sort_cols).reset_index(drop=True)

    scaler = MinMaxScaler()
    train_scaled[features] = scaler.fit_transform(train_scaled[features])
    other_scaled[features] = scaler.transform(other_scaled[features])

    return train_scaled, other_scaled, scaler


def scale_target(y_train, y_other):
  
    """"
    Scales the target variable y_train 
    and y_other using MinMaxScaler.

    Parameters
    ----------
        y_train : np.ndarray
            The training target variable.
        y_other : np.ndarray
            The other target variable.

    Returns
    -------
        tuple: A tuple of the scaled y_train 
        and y_other, and the fitted scaler.
    """

    sc = MinMaxScaler()
    y_train_s = sc.fit_transform(y_train.reshape(-1, 1))
    y_other_s = sc.transform(y_other.reshape(-1, 1))
    return y_train_s, y_other_s, sc


def create_sequences(df, features, target, look_back, horizon):

    """
    Creates (X, y, locations, ids) sequences per location and year.

    A sequence never crosses from one location or year to another.

    Parameters
    ----------
        df : pd.DataFrame
            The DataFrame to create sequences from.
        features : list
            The list of features to include in the sequences.
        target : str
            The name of the target variable.
        look_back : int
            The number of time steps to look back.
        horizon : int
            The number of time steps to predict.

    Returns
    -------
        tuple: The sequences X, the targets y, the (longitude, latitude)
        of each sequence, and a DataFrame with the identifiers of the row
        each target belongs to, in the same order as y.
    """
    id_cols = [c for c in ['longitude', 'latitude', 'year', 'month',
                           'month_num'] if c in df.columns]
    X, y, locs, ids = [], [], [], []

    for (lon, lat, _year), grp in df.groupby(
            ['longitude', 'latitude', 'year']):

        grp = grp.sort_values('month_num')
        data = grp[features].values
        tgt = grp[target].values

        for i in range(len(grp) - look_back - horizon + 1):
            at = i + look_back + horizon - 1
            X.append(data[i:i + look_back])
            y.append(tgt[at])
            locs.append((lon, lat))
            ids.append(grp.iloc[at][id_cols])

    ids = pd.DataFrame(ids, columns=id_cols).reset_index(drop=True)
    return np.array(X), np.array(y), locs, ids


def to_loader(X, y, device, batch_size, shuffle = False):
    
    """
    Converts numpy arrays to a PyTorch DataLoader.

    Parameters
    ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target data.
        device : torch.device
            The device to move the data to.
        batch_size : int
            The batch size.
        shuffle : bool, optional
            Whether to shuffle the data, by default False

    Returns
    -------
        DataLoader: A PyTorch DataLoader.
    """
    Xt = torch.tensor(X, dtype = torch.float32).to(device)
    yt = torch.tensor(y, dtype = torch.float32).to(device)
    return DataLoader(TensorDataset(Xt, yt), batch_size = batch_size,
                      shuffle = shuffle)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FREEZING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def apply_freeze_strategy(model, freeze_prefixes):

    """
    This function freezes the parameters of a 
    model based on a list of prefixes.

    Parameters
    ----------
        model : torch.nn.Module
            The model whose parameters to freeze.
        freeze_prefixes : list
            A list of prefixes to match against parameter names.

    Returns
    -------
        tuple: A tuple of the frozen and trainable parameter names.
    """
    frozen, trainable = [], []

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in freeze_prefixes):
            param.requires_grad = False
            frozen.append(name)
        else:
            param.requires_grad = True
            trainable.append(name)

    log.info('Frozen    : %s', frozen   if frozen    else 'none')
    log.info('Trainable : %s', trainable if trainable else 'none')

    return frozen, trainable


def finetune_model(base_model, ft_loader, val_loader,
                   freeze_prefixes, strategy_label, out_dir, device):

    """"
    This function fine-tunes a base model on a given 
    dataset with early stopping.
    Parameters
    ----------
        base_model : torch.nn.Module
            The base model to fine-tune.
        ft_loader : DataLoader
            The training data loader.
        val_loader : DataLoader
            The validation data loader.
        freeze_prefixes : list
            A list of prefixes to match against parameter names.
        strategy_label : str
            A label for the fine-tuning strategy.
        out_dir : str
            The output directory to save the best model.
        device : torch.device
            The device to move the model to.

    Returns
    -------
        tuple: A tuple of the fine-tuned model and the 
        training log as a DataFrame.
    """
    model = copy.deepcopy(base_model).to(device)
    apply_freeze_strategy(model, freeze_prefixes)

    # Only pass parameters that require gradients to the optimiser
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.Adam(trainable_params, lr = FINETUNE_LR)
    criterion  = nn.MSELoss()

    os.makedirs(out_dir, exist_ok = True)
    best_path  = os.path.join(out_dir, f'best_{strategy_label}.pt')
    best_loss  = float('inf')
    patience_c = 0
    log_rows   = []

    for epoch in range(FINETUNE_EPOCHS):

        model.train()
        train_loss = 0.0

        for Xb, yb in ft_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)

        train_loss /= len(ft_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb), yb).item() * Xb.size(0)

        val_loss /= len(val_loader.dataset)
        log_rows.append({'epoch': epoch + 1,
                         'train_loss': train_loss,
                         'val_loss': val_loss,
                         'strategy': strategy_label})

        if val_loss < best_loss:
            best_loss  = val_loss
            patience_c = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_c += 1
            if patience_c >= FINETUNE_PATIENCE:
                log.info('[%s] Early stopping at epoch %d', strategy_label, epoch + 1)
                break

    # Load best weights
    model.load_state_dict(torch.load(best_path))
    log.info('[%s] Best val_loss=%.5f', strategy_label, best_loss)

    return model, pd.DataFrame(log_rows)


def predict(model, loader, y_scaler, device):
   
    """
    Runs inference on a model with a given data loader 
    and inverse-transforms the predictions.

    Parameters
    ----------
        model : torch.nn.Module
            The model to run inference on.
        loader : DataLoader
            The data loader for the input data.
        y_scaler : sklearn.preprocessing.StandardScaler
            The scaler to inverse-transform the predictions.
        device : torch.device
            The device to move the model to.

    Returns
    -------
        tuple: A tuple of the true and predicted values.
    """

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            preds.append(model(Xb).cpu().numpy())
            trues.append(yb.cpu().numpy())

    y_pred = y_scaler.inverse_transform(np.concatenate(preds)).ravel()
    y_true = y_scaler.inverse_transform(np.concatenate(trues)).ravel()

    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    '''Returns dict of R2, MSE, RMSE, MAE, N.'''

    return {
        'R2':   r2_score(y_true, y_pred),
        'MSE':  mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'N':    int(len(y_true)),
    }


def metrics_per_group(results_df, group_cols):
    
    """
    Computes metrics for each group in the results DataFrame.
    Parameters
    ----------
        results_df : pd.DataFrame
            The results DataFrame.
        group_cols : list
            The columns to group by.

    Returns
    -------
        pd.DataFrame: A DataFrame with the computed metrics 
        for each group.
    """
    records = []

    for keys, grp in results_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        m = compute_metrics(grp['observed'].values, grp['predicted'].values)
        records.append({**dict(zip(group_cols, keys)), **m})

    return pd.DataFrame(records)


def build_results_df(y_true, y_pred, locations, id_df, id_cols, approach_label):
    
    """
    This function assembles a row-level results DataFrame with identifiers.

    Parameters
    ----------
        y_true : array-like
            The true values.
        y_pred : array-like
            The predicted values.
        locations : array-like
            The locations of the data points.
        id_df : pd.DataFrame
            The DataFrame with the identifiers.
        id_cols : list
            The columns to include in the results DataFrame.
        approach_label : str
            The label for the approach.

    Returns
    -------
        pd.DataFrame: A DataFrame with the results.
    """
    if len(id_df) != len(y_true):
        raise ValueError(
            f'{len(id_df)} identifier rows for {len(y_true)} predictions')
    present = [c for c in id_cols if c in id_df.columns]
    results = id_df[present].copy().reset_index(drop = True)
    results['observed']  = y_true
    results['predicted'] = y_pred
    results['residual']  = y_true - y_pred
    results['abs_error'] = np.abs(results['residual'])
    results['approach']  = approach_label
    return results


def plot_comparison(all_results, all_metrics, out_path):

    """
    This function creates a multi-panel comparison figure.

    Parameters
    ----------
        all_results : dict
            A dictionary of results for each approach.
        all_metrics : dict
            A dictionary of metrics for each approach.
        out_path : str
            The path to save the comparison figure.

    Returns
    -------
        None
    """
    strategies  = list(all_results.keys())
    n_strats    = len(strategies)
    palette     = ['#E07B39', '#2176AE', '#57A773', '#6B4E71', '#C84B31']
    colors      = {s: palette[i % len(palette)] for i, s in enumerate(strategies)}

    # Number of rows: 1 row of scatter (one panel per strategy) + 2 fixed rows
    n_scatter_cols = n_strats
    fig = plt.figure(figsize = (6 * n_scatter_cols, 18))
    gs  = gridspec.GridSpec(3, n_scatter_cols, figure = fig,
                            hspace = 0.45, wspace = 0.30)

    # ── Row 1: Obs vs Pred per strategy ──────────────────────────────────
    for col, strat in enumerate(strategies):
        res  = all_results[strat]
        gm   = all_metrics[strat]
        obs  = res['observed'].values
        pred = res['predicted'].values
        ax   = fig.add_subplot(gs[0, col])
        ax.scatter(obs, pred, alpha = 0.15, s = 5, color = colors[strat])
        lims = [min(obs.min(), pred.min()) - 0.02,
                max(obs.max(), pred.max()) + 0.02]
        ax.plot(lims, lims, 'r--', lw = 1.2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Observed MRI'); ax.set_ylabel('Predicted MRI')
        ax.set_title(f'{strat}\nR2={gm["R2"]:.4f}  RMSE={gm["RMSE"]:.4f}',
                     fontsize = 9)

    # ── Row 2 left: Residual distributions overlaid ───────────────────────
    ax2 = fig.add_subplot(gs[1, :n_scatter_cols // 2])
    for strat, res in all_results.items():
        resid = res['observed'].values - res['predicted'].values
        ax2.hist(resid, bins = 60, alpha = 0.45, color = colors[strat],
                 label = strat, edgecolor = 'none')
    ax2.axvline(0, color = 'black', lw = 1.2, linestyle = '--')
    ax2.set_xlabel('Residual (obs - pred)'); ax2.set_ylabel('Count')
    ax2.set_title('Residual Distribution — All Strategies')
    ax2.legend(fontsize = 8)

    # ── Row 2 right: Temporal mean MRI ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, n_scatter_cols // 2:])
    first_res = list(all_results.values())[0]
    if 'year' in first_res.columns and 'month_num' in first_res.columns:
        obs_trend = (first_res.groupby(['year', 'month_num'])['observed']
                     .mean().reset_index()
                     .sort_values(['year', 'month_num']))
        ax3.plot(range(len(obs_trend)), obs_trend['observed'],
                 label = 'Observed', color = 'black', lw = 2.0)
        for strat, res in all_results.items():
            trend = (res.groupby(['year', 'month_num'])['predicted']
                     .mean().reset_index()
                     .sort_values(['year', 'month_num']))
            ax3.plot(range(len(trend)), trend['predicted'],
                     label = strat, color = colors[strat],
                     lw = 1.4, linestyle = '--')
    ax3.set_xlabel('Time step'); ax3.set_ylabel('Mean MRI')
    ax3.set_title('Temporal Trend — All Strategies')
    ax3.legend(fontsize = 8)

    # ── Row 3 left: Global metric bar chart ───────────────────────────────
    ax4 = fig.add_subplot(gs[2, :n_scatter_cols // 2])
    metric_names = ['R2', 'RMSE', 'MAE']
    x     = np.arange(len(metric_names))
    width = 0.8 / n_strats
    for i, strat in enumerate(strategies):
        vals = [all_metrics[strat][m] for m in metric_names]
        offset = (i - n_strats / 2 + 0.5) * width
        bars = ax4.bar(x + offset, vals, width,
                       label = strat, color = colors[strat], alpha = 0.85)
        for bar in bars:
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     h + (0.005 if h >= 0 else -0.02),
                     f'{h:.3f}', ha = 'center', va = 'bottom', fontsize = 7)
    ax4.set_xticks(x); ax4.set_xticklabels(metric_names)
    ax4.axhline(0, color = 'black', lw = 0.8)
    ax4.set_title('Global Metrics — All Strategies')
    ax4.legend(fontsize = 8)

    # ── Row 3 right: Per-location R2 — all fine-tuned vs baseline ─────────
    ax5 = fig.add_subplot(gs[2, n_scatter_cols // 2:])
    baseline_loc = metrics_per_group(all_results['baseline'],
                                     ['latitude', 'longitude'])
    for strat in strategies:
        if strat == 'baseline':
            continue
        ft_loc = metrics_per_group(all_results[strat], ['latitude', 'longitude'])
        merged = baseline_loc.merge(ft_loc, on = ['latitude', 'longitude'],
                                    suffixes = ('_base', '_ft'))
        ax5.scatter(merged['R2_base'], merged['R2_ft'],
                    alpha = 0.25, s = 6, color = colors[strat], label = strat)

    all_r2 = pd.concat([
        metrics_per_group(r, ['latitude', 'longitude'])['R2']
        for r in all_results.values()
    ])
    lims = [all_r2.min() - 0.05, all_r2.max() + 0.05]
    ax5.plot(lims, lims, 'r--', lw = 1.2, label = 'No-change line')
    ax5.set_xlabel('Baseline R2 (per location)')
    ax5.set_ylabel('Fine-tuned R2 (per location)')
    ax5.set_title('Per-Location R2: All Fine-tuned vs Baseline')
    ax5.legend(fontsize = 8)

    fig.suptitle(
        'LSTM MRI Model Comparison - Baseline vs Fine-tuning Strategies\n'
        f'Test year(s): {", ".join(str(y) for y in TEST_YEARS)}',
        fontsize = 14, fontweight = 'bold')
    fig.savefig(out_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)
    log.info('Comparison plot  ->  %s', out_path)


def save_comparison_csvs(all_metrics, all_results, out_dir):
    
    """
    Saves global, per-location and per-time comparison CSVs with delta columns.

    Parameters
    ----------
        all_metrics : dict
            A dictionary of metrics for each approach.
        all_results : dict
            A dictionary of results for each approach.
        out_dir : str
            The directory to save the comparison CSVs.

    Returns
    -------
        None
    """
    strategies = list(all_metrics.keys())

    # ── Global metrics ────────────────────────────────────────────────────
    pd.DataFrame([
        {'approach': s, **all_metrics[s]} for s in strategies
    ]).to_csv(os.path.join(out_dir, 'comparison_global_metrics.csv'), index = False)
    log.info('Global comparison  ->  comparison_global_metrics.csv')

    # ── Per-location ──────────────────────────────────────────────────────
    base_loc = metrics_per_group(all_results['baseline'], ['latitude', 'longitude'])
    merged_loc = base_loc.copy()

    for strat in strategies:
        if strat == 'baseline':
            continue
        ft_loc = metrics_per_group(all_results[strat], ['latitude', 'longitude'])
        ft_loc = ft_loc.rename(columns = {
            m: f'{m}_{strat}' for m in ['R2', 'MSE', 'RMSE', 'MAE', 'N']})
        merged_loc = merged_loc.merge(ft_loc, on = ['latitude', 'longitude'])

    # Rename baseline metric columns
    merged_loc = merged_loc.rename(columns = {
        m: f'{m}_baseline' for m in ['R2', 'MSE', 'RMSE', 'MAE', 'N']})

    # Add delta columns for each fine-tuned strategy
    for strat in strategies:
        if strat == 'baseline':
            continue
        for m in ['R2', 'RMSE', 'MAE']:
            merged_loc[f'{m}_delta_{strat}'] = (merged_loc[f'{m}_{strat}']
                                                - merged_loc[f'{m}_baseline'])
    
    for strat in strategies:
        obs_pred = all_results[strat].groupby(['latitude', 'longitude'])[
            ['observed', 'predicted']].mean().reset_index()
        obs_pred = obs_pred.rename(columns={
            'observed':  f'observed_{strat}',
            'predicted': f'predicted_{strat}'})
        merged_loc = merged_loc.merge(obs_pred, on=['latitude', 'longitude'])

    merged_loc.to_csv(
        os.path.join(out_dir, 'comparison_per_location.csv'), index = False)
    log.info('Per-location comparison  ->  comparison_per_location.csv')

    # ── Per-time ──────────────────────────────────────────────────────────
    base_time = metrics_per_group(all_results['baseline'], ['year', 'month_num'])
    merged_time = base_time.copy()

    for strat in strategies:
        if strat == 'baseline':
            continue
        ft_time = metrics_per_group(all_results[strat], ['year', 'month_num'])
        ft_time = ft_time.rename(columns = {
            m: f'{m}_{strat}' for m in ['R2', 'MSE', 'RMSE', 'MAE', 'N']})
        merged_time = merged_time.merge(ft_time, on = ['year', 'month_num'])

    merged_time = merged_time.rename(columns = {
        m: f'{m}_baseline' for m in ['R2', 'MSE', 'RMSE', 'MAE', 'N']})

    for strat in strategies:
        if strat == 'baseline':
            continue
        for m in ['R2', 'RMSE', 'MAE']:
            merged_time[f'{m}_delta_{strat}'] = (merged_time[f'{m}_{strat}']
                                                 - merged_time[f'{m}_baseline'])
    
    for strat in strategies:
        obs_pred = all_results[strat].groupby(['year', 'month_num'])[
            ['observed', 'predicted']].mean().reset_index()
        obs_pred = obs_pred.rename(columns={
            'observed':  f'observed_{strat}',
            'predicted': f'predicted_{strat}'})
        merged_time = merged_time.merge(obs_pred, on=['year', 'month_num'])

    merged_time.to_csv(
        os.path.join(out_dir, 'comparison_per_time.csv'), index = False)
    log.info('Per-time comparison  ->  comparison_per_time.csv')

def run_lstm_comparative_evaluation(
        model_path   = MODEL_PATH,
        data_path    = TEST_DATA_PATH,
        target       = TARGET_COLUMN,
        output_dir   = OUTPUT_DIR):
    
    """
    Runs a comparative evaluation of the LSTM model.

    Parameters
    ----------
        model_path : str
            The path to the pre-trained model.
        data_path : str
            The path to the test data.
        target : str
            The name of the target column.
        output_dir : str
            The directory to save the output files.

    Returns
    -------
        None
    """

    os.makedirs(output_dir, exist_ok = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info('Device: %s', device)

    # ── Load base model ───────────────────────────────────────────────────
    if not os.path.exists(model_path):
        log.error('Model not found: %s', model_path)
        sys.exit(1)

    base_model = MRILSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    base_model.load_state_dict(torch.load(model_path, map_location = device))
    log.info('Base model loaded  ->  %s', model_path)

    # ── Load and prepare data ─────────────────────────────────────────────
    df = load_and_prepare(data_path)

    # ── Three-way split by year ───────────────────────────────────────────
    ft_df, val_df, test_df = split_by_year(
        df, FINETUNE_YEARS, VAL_YEARS, TEST_YEARS)

    log.info('Fine-tune: %d rows | Val: %d rows | Test: %d rows',
             len(ft_df), len(val_df), len(test_df))

    # ── Scale features (fit on fine-tune set, apply to val and test) ────────
    ft_scaled, val_scaled,  _ = scale_features(ft_df, val_df,  FEATURES)
    _,         test_scaled, _ = scale_features(ft_df, test_df, FEATURES)

    # ── Create sequences ──────────────────────────────────────────────────
    X_ft,   y_ft,   _,         _          = create_sequences(ft_scaled,   FEATURES, target, LOOK_BACK, HORIZON)
    X_val,  y_val,  _,         _          = create_sequences(val_scaled,  FEATURES, target, LOOK_BACK, HORIZON)
    X_test, y_test, locs_test, test_id_df = create_sequences(test_scaled, FEATURES, target, LOOK_BACK, HORIZON)

    for name, X in (('fine-tune', X_ft), ('validation', X_val), ('test', X_test)):
        if len(X) == 0:
            raise ValueError(
                f'no {name} sequences: a location needs at least '
                f'LOOK_BACK + HORIZON = {LOOK_BACK + HORIZON} consecutive months')

    log.info('Sequences: fine-tune %d | val %d | test %d',
             len(X_ft), len(X_val), len(X_test))

    # ── Scale target (fit on fine-tune set, apply to val and test) ──────────
    y_ft_s,   y_val_s,  y_scaler = scale_target(y_ft, y_val)
    y_ft_s,   y_test_s, _        = scale_target(y_ft, y_test)

    # ── DataLoaders ───────────────────────────────────────────────────────
    ft_loader   = to_loader(X_ft,   y_ft_s,  device, FINETUNE_BATCH, shuffle = True)
    val_loader  = to_loader(X_val,  y_val_s, device, FINETUNE_BATCH)
    test_loader = to_loader(X_test, y_test_s, device, FINETUNE_BATCH)

    # test_id_df holds the identifiers of each test prediction, in order
    id_cols    = ['latitude', 'longitude', 'year', 'month', 'month_num']

    # ── STEP 1: Baseline evaluation (no fine-tuning) ──────────────────────
    log.info('=' * 60)
    log.info('STEP 1 - BASELINE (Uganda model, no adaptation)')
    log.info('=' * 60)

    y_true_base, y_pred_base = predict(base_model, test_loader, y_scaler, device)
    gm_base = compute_metrics(y_true_base, y_pred_base)
    log.info('Baseline  R2=%.4f  RMSE=%.4f  MAE=%.4f',
             gm_base['R2'], gm_base['RMSE'], gm_base['MAE'])

    res_base = build_results_df(y_true_base, y_pred_base,
                                locs_test, test_id_df, id_cols, 'baseline')

    all_results = {'baseline': res_base}
    all_metrics = {'baseline': gm_base}
    all_logs    = []

    # ── STEP 2: Fine-tune each strategy ───────────────────────────────────
    for i, (strat_label, freeze_prefixes) in enumerate(FREEZE_STRATEGIES.items(), 2):

        log.info('=' * 60)
        log.info('STEP %d - FINE-TUNING: %s', i, strat_label)
        log.info('=' * 60)

        ft_model, ft_log = finetune_model(
            base_model, ft_loader, val_loader,
            freeze_prefixes, strat_label, output_dir, device)

        ft_log['strategy'] = strat_label
        all_logs.append(ft_log)

        y_true_ft, y_pred_ft = predict(ft_model, test_loader, y_scaler, device)
        gm_ft = compute_metrics(y_true_ft, y_pred_ft)
        log.info('[%s]  R2=%.4f  RMSE=%.4f  MAE=%.4f',
                 strat_label, gm_ft['R2'], gm_ft['RMSE'], gm_ft['MAE'])

        res_ft = build_results_df(y_true_ft, y_pred_ft,
                                  locs_test, test_id_df, id_cols, strat_label)

        all_results[strat_label] = res_ft
        all_metrics[strat_label] = gm_ft

    # ── STEP 3: Save fine-tuning training logs ────────────────────────────
    pd.concat(all_logs).to_csv(
        os.path.join(output_dir, 'finetuning_training_logs.csv'), index = False)
    log.info('Training logs  ->  finetuning_training_logs.csv')

    # ── STEP 4: Save comparison files ────────────────────────────────────
    log.info('=' * 60)
    log.info('STEP %d - SAVING COMPARISON FILES', len(FREEZE_STRATEGIES) + 2)
    log.info('=' * 60)

    save_comparison_csvs(all_metrics, all_results, output_dir)
    plot_comparison(all_results, all_metrics,
                    os.path.join(output_dir, 'comparison_plots.png'))

    # ── Print summary ──────────────────────────────────────────────────────
    w = 72
    print('\n' + '=' * w)
    print('  LSTM MRI - COMPARATIVE EVALUATION SUMMARY')
    header = f'  {"Approach":<22}  {"R2":>9}  {"RMSE":>9}  {"MAE":>9}  {"N":>9}'
    for strat, gm in all_metrics.items():
        print(f'  {strat:<22}  {gm["R2"]:>9.4f}  {gm["RMSE"]:>9.4f}'
              f'  {gm["MAE"]:>9.4f}  {gm["N"]:>9,}')


if __name__ == '__main__':
    run_lstm_comparative_evaluation()
