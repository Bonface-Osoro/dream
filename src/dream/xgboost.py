import os
import csv
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def xg_load_and_prepare_data(csv_path, horizon):

    """Loads data and creates features for XGBoost model.
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the data.
    horizon : int
        Forecast horizon (number of steps ahead to predict).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the prepared features.
    """

    print("Loading data and creating features...")
    df = pd.read_csv(csv_path)

    # Seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    # Lag features
    for lag in [1, 2, 3, 6, 12]:

        df[f'mri_lag{lag}'] = df.groupby(['longitude',
                        'latitude'])['monthly_mri'].shift(lag)

    # Rolling features
    df['mri_roll3'] = df.groupby(['longitude','latitude'])[
        'monthly_mri'].transform(lambda x: x.rolling(3).mean())
    df['mri_roll6'] = df.groupby(['longitude','latitude'])[
        'monthly_mri'].transform(lambda x: x.rolling(6).mean())

    df['target'] = df.groupby(['longitude','latitude'])[
        'monthly_mri'].shift(-horizon)

    df = df.dropna().reset_index(drop=True)

    return df


def xg_split_data(df, features, split_year):

    """Splits the data into training and testing sets based on a year threshold.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the prepared features.
    features : list
        List of feature column names to use for modeling.
    split_year : int
        Year threshold for splitting the data.

    Returns
    -------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training targets.
    X_test : pandas.DataFrame
        Testing features.
    y_test : pandas.Series
        Testing targets.
    locations_test : list
        List of test location coordinates.
    features : list
        List of feature names.
    """

    print("Splitting data by year...")

    train_df = df[df['year'] <= split_year].copy()
    test_df  = df[df['year'] > split_year].copy()

    target = 'target'

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    locations_test = list(zip(test_df['longitude'], test_df['latitude']))

    return X_train, y_train, X_test, y_test, locations_test, features


def train_xgb(X_train, y_train, X_test, y_test):

    """Trains an XGBoost regression model with early stopping.
    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training targets.
    X_test : pandas.DataFrame
        Testing features.
    y_test : pandas.Series
        Testing targets.

    Returns
    -------
    model : XGBRegressor
        Trained XGBoost model.
    """

    print("Training XGBoost model...")

    model = XGBRegressor(n_estimators = 500,
        learning_rate = 0.05, max_depth = 6,
        subsample = 0.8, colsample_bytree = 0.8,
        tree_method = 'hist', device = 'cuda',
        eval_metric = 'rmse', early_stopping_rounds = 20)

    model.fit(X_train, y_train, eval_set = [(X_train, y_train), 
            (X_test, y_test)], verbose = True)

    return model


def save_xg_and_logs(model, results_dir):

    """Saves the trained model and training logs 
    to the specified directory.

    Parameters
    ----------
    model : XGBRegressor
        Trained XGBoost model.
    results_dir : str
        Directory to save the model and logs.

    Returns
    -------
    model_path : str
        Path to the saved model file.
    """

    os.makedirs(results_dir, exist_ok = True)

    model_path = os.path.join(results_dir, 'xgb_model.pkl')
    joblib.dump(model, model_path)

    print('Saving training log...')
    results = model.evals_result()
    train_rmse = results['validation_0']['rmse']
    val_rmse   = results['validation_1']['rmse']

    log_csv_path = os.path.join(results_dir, 'xgb_training_log.csv')

    with open(log_csv_path, 'w', newline = '') as f:

        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Train_RMSE', 'Val_RMSE'])

        for i in range(len(train_rmse)):

            writer.writerow([i+1, train_rmse[i], val_rmse[i]])

    return model_path


def evaluate_xg_model(model, X_test, y_test, test_df, 
                      results_dir):

    """Evaluates the XGBoost model on the test set and saves metrics.
    Parameters
    ----------
    model : XGBRegressor
        Trained XGBoost model.
    X_test : pandas.DataFrame
        Testing features.
    y_test : pandas.Series
        Testing targets.
    test_df : pandas.DataFrame
        DataFrame with the test data.
    results_dir : str
        Directory to save the metrics.

    Returns
    -------
    y_pred : numpy.ndarray
        Predicted values for the test set.
    """
    print("Evaluating model...")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print('MSE:', mse)
    print('MAE:', mae)
    print('R2:', r2)

    # Naive baseline
    if 'mri_lag1' in test_df.columns:

        y_naive = test_df['mri_lag1'].values
        print('Naive MSE:', mean_squared_error(y_test, y_naive))

    print('Model MSE:', mse)

    # Save metrics
    metrics_path = os.path.join(results_dir, 'xg_test_metrics.csv')

    with open(metrics_path, 'w', newline = '') as f:

        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['MSE', mse])
        writer.writerow(['MAE', mae])
        writer.writerow(['R2', r2])

    return y_pred


def build_xg_prediction_dict(locations, y_true, 
                             y_pred):
    """Builds a dictionary of predictions 
    and true values by location.

    Parameters    
    ----------
    locations : list
        List of (longitude, latitude) tuples.
    y_true : numpy.ndarray
        Array of true values.
    y_pred : numpy.ndarray
        Array of predicted values.

    Returns
    -------
    pred_by_loc : dict
        Dictionary mapping locations to their 
        true and predicted values.
    """

    pred_by_loc = {}

    for (lon, lat), pred, true in zip(locations, 
                                      y_pred, y_true):

        if (lon, lat) not in pred_by_loc:

            pred_by_loc[(lon, lat)] = {'true': [], 'pred': []}

        pred_by_loc[(lon, lat)]['true'].append(true)
        pred_by_loc[(lon, lat)]['pred'].append(pred)

    return pred_by_loc


def save_per_location_xg_metrics(pred_by_loc, 
                                 results_dir):
    """Calculates and saves per-location metrics for the XGBoost model.
    Parameters
    ----------
    pred_by_loc : dict
        Dictionary mapping locations to their true and predicted values.
    results_dir : str
        Directory to save the metrics.

    """

    print("Saving per-location metrics...")

    rows = []

    for (lon, lat), vals in pred_by_loc.items():

        if len(vals['true']) > 1:

            rows.append({'longitude': lon,
                'latitude': lat,
                'R2': r2_score(vals['true'], vals['pred']),
                'MSE': mean_squared_error(vals['true'], vals['pred']),
                'MAE': mean_absolute_error(vals['true'], vals['pred']),
                'n_samples': len(vals['true'])})

    df = pd.DataFrame(rows)

    path = os.path.join(results_dir, 'per_location_xg_metrics.csv')
    df.to_csv(path, index = False)


def xg_sanity_checks(y_true, y_pred):

    """Performs sanity checks on the predictions and true values.
    Parameters
    ----------
    y_true : numpy.ndarray
        Array of true values.
    y_pred : numpy.ndarray
        Array of predicted values.
    """

    print("Pred std:", y_pred.std())
    print("True std:", y_true.std())
    print("Pred mean:", y_pred.mean())
    print("True mean:", y_true.mean())

def plot_xg_predictions(pred_by_loc, save_dir, n, show):
    """
    Plots and saves prediction vs true values for first n locations.

    Parameters:
    ----------
    pred_by_loc : dict
        Dictionary with structure { (lon, lat): {'true': [...], 'pred': [...]} }
    save_dir : str
        Directory to save plots
    n : int
        Number of locations to plot
    show : bool
        Whether to display plots (default False)
    """

    print("Saving prediction plots...")

    os.makedirs(save_dir, exist_ok=True)

    for i, (loc, vals) in enumerate(list(pred_by_loc.items())[:n]):

        plt.figure(figsize = (8,4))
        plt.plot(vals['true'], label='True MRI')
        plt.plot(vals['pred'], label='Predicted MRI')

        plt.title(f'Location: {loc}')
        plt.xlabel('Time step')
        plt.ylabel('MRI')
        plt.legend()

        lon, lat = loc
        filename = f"xg_predicted_loc_{i+1}_lon_{lon:.4f}_lat_{lat:.4f}.png"
        filepath = os.path.join(save_dir, filename)

        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")

        if show:

            plt.show()

        plt.close()


def export_shap_values(model, X_train, X_test, 
                       feature_names, save_dir):

    """Computes SHAP values for the XGBoost model 
    and exports them to a CSV file.

    Parameters
    ----------
    model : xgboost.XGBRegressor
        Trained XGBoost model.
    X_train : pandas.DataFrame
        Training data features.
    X_test : pandas.DataFrame
        Test data features.
    feature_names : list of str
        Names of the features.
    save_dir : str
        Directory to save the exported values.
    """

    print("Computing and exporting SHAP values...")
    os.makedirs(save_dir, exist_ok = True)
    X_sample = X_test.sample(min(1000, len(X_test)), random_state = 42)
    explainer = shap.Explainer(lambda x: model.predict(x), X_train)

    try:

        shap_values = explainer(X_sample)
        is_explanation = True
    except:

        shap_values = explainer.shap_values(X_sample) 
        is_explanation = False

    if is_explanation:

        shap_array = shap_values.values
        filename = 'lagged_shap_values_full.csv'
    else:

        shap_array = shap_values
        filename = 'non_lagged_shap_values_full.csv'

    shap_df = pd.DataFrame(shap_array, columns = feature_names)
    for col in feature_names:

        shap_df[f"{col}_value"] = X_sample[col].values

    csv_path = os.path.join(save_dir, filename)
    shap_df.to_csv(csv_path, index = False)

def export_shap_per_location(model, X_train, X_test, feature_names,
                            locations_test, save_dir):
    """
    Computes SHAP values and exports per-location SHAP importance.

    Parameters
    ----------
    model : trained XGBoost model
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    feature_names : list
    locations_test : list of (lon, lat) tuples aligned with X_test
    save_dir : str
    """

    print("Computing per-location SHAP values...")

    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------
    # SAMPLE DATA (optional)
    # ---------------------------
    X_sample = X_test.copy()  # use full test for location analysis

    # ---------------------------
    # CREATE EXPLAINER
    # ---------------------------
    explainer = shap.Explainer(lambda x: model.predict(x), X_train)

    # ---------------------------
    # COMPUTE SHAP VALUES
    # ---------------------------
    try:
        shap_values = explainer(X_sample)
        shap_array = shap_values.values
    except:
        shap_array = explainer.shap_values(X_sample)

    # ---------------------------
    # BUILD DATAFRAME
    # ---------------------------
    shap_df = pd.DataFrame(shap_array, columns=feature_names)

    # Add location info
    shap_df['longitude'] = [loc[0] for loc in locations_test]
    shap_df['latitude']  = [loc[1] for loc in locations_test]

    # ---------------------------
    # AGGREGATE PER LOCATION
    # ---------------------------
    per_loc_results = []

    grouped = shap_df.groupby(['longitude', 'latitude'])

    for (lon, lat), group in grouped:

        # mean absolute SHAP per feature
        mean_shap = np.abs(group[feature_names]).mean()

        row = {
            'longitude': lon,
            'latitude': lat
        }

        for f in feature_names:
            row[f] = mean_shap[f]

        per_loc_results.append(row)

    per_loc_df = pd.DataFrame(per_loc_results)

    # ---------------------------
    # SAVE CSV
    # ---------------------------
    csv_path = os.path.join(save_dir, "shap_per_location.csv")
    per_loc_df.to_csv(csv_path, index=False)

  
def explain_xgboost(model, X_train, X_test, 
                          feature_names, save_dir):
    
    """Generates SHAP explainability plots for the XGBoost model.
    Parameters
    ----------
    model : xgboost.XGBRegressor
        Trained XGBoost model.
    X_train : pandas.DataFrame
        Training data features.
    X_test : pandas.DataFrame
        Test data features.
    feature_names : list of str
        Names of the features.
    save_dir : str
        Directory to save the plots.
    """

    print("Running SHAP explainability for XGBoost...")
    os.makedirs(save_dir, exist_ok = True)
    explainer = shap.Explainer(lambda x: model.predict(x), X_train)
    X_sample = X_test.sample(min(1000, len(X_test)), random_state = 42)

    try:
        shap_values = explainer(X_sample)   
        is_explanation = True

    except:
        shap_values = explainer.shap_values(X_sample)
        is_explanation = False

    plt.figure()
    if is_explanation:

        shap.summary_plot(shap_values, X_sample, feature_names = feature_names,
                          show = False)
        summary_name = 'lagged_shap_summary.png'
    else:

        shap.summary_plot(shap_values, X_sample, feature_names = feature_names,
                          show = False)
        summary_name = 'non_lagged_shap_summary.png'

    plt.savefig(os.path.join(save_dir, summary_name), bbox_inches ='tight')
    plt.close()

    plt.figure()

    if is_explanation:

        shap.plots.bar(shap_values, show = False)
        bar_name = 'lagged_shap_bar.png'
    else:

        shap.summary_plot(shap_values, X_sample, feature_names = feature_names,
                          plot_type = "bar", show = False)
        bar_name = 'non_lagged_shap_bar.png'

    plt.savefig(os.path.join(save_dir, bar_name), bbox_inches = 'tight')
    plt.close()