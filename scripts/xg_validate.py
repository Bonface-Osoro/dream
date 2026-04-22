import configparser
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import *

warnings.filterwarnings("ignore")
CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')

feature_cols = ['ndvi',
    'precipitation_mm',
    'temperature_C',
    'elevation_m',
    'month_sin',
    'month_cos',
    'longitude',
    'latitude']

month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12}

def load_model(model_path):
    """
    Load a pickled model from disk.

    Parameters
    ----------
        model_path : str
            Path to the pickled model file.
    Returns
    -------
        model : object
            The loaded model.
    """
    if not os.path.exists(model_path):

        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:

        model = pickle.load(f)
    print(f"Model loaded from: {model_path}")

    return model


def load_data(data_path):

    """
    Load CSV/TSV data, auto-detecting the 
    separator if not specified.

    Parameters
    ----------
        data_path : str
            Path to the data file.
    Returns
    -------
        df : pandas.DataFrame
            The loaded data.
    """
    sep = None
    if not os.path.exists(data_path):

        raise FileNotFoundError(f"Data file not found: {data_path}")

    if sep:

        df = pd.read_csv(data_path, sep=sep)
    else:

        try:

            df = pd.read_csv(data_path, sep="\t")
            if df.shape[1] <= 1:  

                df = pd.read_csv(data_path)
        except Exception:

            df = pd.read_csv(data_path)

    print(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def preprocess(df):
    """Coerce dtypes and patch any known quirks 
    (e.g. 'sept' month label).

    Parameters
    ----------
        df : pandas.DataFrame
            The input data.
    Returns
    -------
        df : pandas.DataFrame
            The preprocessed data.
    """
    df = df.copy()

    # Normalise the text 'month' column if present (not used as feature but
    # useful for debugging). Repair 'sept' → 9 in month_num if needed.
    if "month" in df.columns:
        df["month_str"] = df["month"].astype(str).str.strip().str.lower()
        df["month_num_derived"] = df["month_str"].map(month_map)
        # Fill missing month_num from the string column
        if "month_num" in df.columns:
            mask = df["month_num"].isna()
            df.loc[mask, "month_num"] = df.loc[mask, "month_num_derived"]
        else:
            df["month_num"] = df["month_num_derived"]

    # ── Cyclical month encoding (what the model actually expects) ────────────
    # month_num must exist at this point (raw column or derived above).
    if "month_num" not in df.columns:
        raise ValueError(
            "Cannot find 'month_num' (or a 'month' text column to derive it from). "
            "Ensure your data has one of these columns."
        )
    df["month_num"] = pd.to_numeric(df["month_num"], errors="coerce")
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    # Ensure all feature columns are numeric
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_metrics(y_true, y_pred):

    """Return R², MAE, MSE, RMSE for 
    a pair of arrays.
    Parameters
    ----------
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.
    Returns
    -------
        metrics : dict
            Dictionary containing the computed metrics.
    """
    n = len(y_true)
    if n < 2:

        return {'r2': np.nan, 'mae': np.nan, 'mse': 
                np.nan, 'rmse': np.nan, 'n_samples': n}
    return {
        'r2': round(r2_score(y_true, y_pred), 6),
        'mae': round(mean_absolute_error(y_true, y_pred), 6),
        'mse': round(mean_squared_error(y_true, y_pred), 6),
        'rmse': round(np.sqrt(mean_squared_error(y_true, y_pred)), 6),
        'n_samples': n,
    }


def evaluate_xg_boost(model_path, data_path, target_col,
             output_dir):
    
    """
    Evaluate a pickled XGBoost model on a dataset, 
    computing overall and per-location metrics.

    Parameters
    ----------
        model_path : str
            Path to the pickled XGBoost model.
        data_path : str
            Path to the data file.
        target_col : str
            Name of the target column.
        output_dir : str
            Directory to save the evaluation outputs.
    Returns
    -------
        loc_df : pandas.DataFrame
            DataFrame containing per-location metrics.
        df : pandas.DataFrame
            DataFrame containing full predictions and residuals.
        overall : dict
            Dictionary containing overall metrics.
    """
    sep = None
    os.makedirs(output_dir, exist_ok = True)

    model = load_model(model_path)
    df    = load_data(data_path)
    df    = preprocess(df)

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:

        raise ValueError(
            f"The following expected feature columns are missing from the data:\n"
            f"  {missing_features}\n"
            f"Available columns: {list(df.columns)}")
    
    if target_col not in df.columns:

        raise ValueError(
            f"Target column '{target_col}' not found. Available: {list(df.columns)}"
        )

    before = len(df)
    df = df.dropna(subset = feature_cols + [target_col])
    after  = len(df)

    if before != after:

        print(f"Dropped {before - after} rows with missing values.")

    X = df[feature_cols]
    y = df[target_col].values

    y_pred = model.predict(X)
    df['predicted'] = y_pred
    df['residual']  = y - y_pred

    overall = compute_metrics(y, y_pred)
    print("\n── Overall metrics ─────────────────────────────────────────")
    for k, v in overall.items():

        print(f"   {k:12s}: {v}")

    print("\n── Computing per-location metrics … ────────────────────────")
    location_records = []
    for (lat, lon), grp in df.groupby(['latitude', 'longitude'], sort = False):
        m = compute_metrics(grp[target_col].values, grp['predicted'].values)
        location_records.append({
            'latitude':  lat,
            'longitude': lon,
            **m,
        })

    loc_df = pd.DataFrame(location_records).sort_values(
        ['latitude', 'longitude']).reset_index(drop = True) 

    summary_path = os.path.join(output_dir, "ZWE_per_location_metrics.csv")
    loc_df.to_csv(summary_path, index=False)
    print(f"Per-location metrics saved  → {summary_path}")

    predictions_path = os.path.join(output_dir, "ZWE_full_predictions.csv")
    df.to_csv(predictions_path, index=False)
    print(f"Full predictions saved       → {predictions_path}")

    overall_path = os.path.join(output_dir, "ZWE_overall_metrics.csv")
    pd.DataFrame([overall]).to_csv(overall_path, index=False)
    print(f"Overall metrics saved        → {overall_path}")

    print("\n── Per-location metrics (preview, first 20 rows) ───────────")
    print(loc_df.head(20).to_string(index=False))

    return loc_df, df, overall

model_path = os.path.join(DATA_RESULTS, 'xgboost', 'xgb_model.pkl')
data_path = os.path.join(DATA_RESULTS, 'mri', 'ZWE_malaria_risk_index_monthly.csv')
output_dir = os.path.join(DATA_RESULTS, 'zimbabwe_validation')
if __name__ == "__main__":

    evaluate_xg_boost(
        model_path = model_path,
        data_path = data_path,
        target_col = "monthly_mri",
        output_dir = output_dir
    )