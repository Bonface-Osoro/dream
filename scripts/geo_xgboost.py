import configparser
import os
import warnings
import pandas as pd
from dream.xgboost import *

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')       

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')


mri_data = os.path.join(DATA_RESULTS, 'mri', 'malaria_risk_index_monthly.csv')
model_path = os.path.join(DATA_RESULTS, 'xgboost')
df = xg_load_and_prepare_data(mri_data, 6)

lagged_features = ['ndvi', 'precipitation_mm', 'temperature_C', 'elevation_m',
    'month_sin', 'month_cos','mri_lag3', 'mri_lag6', 'mri_lag12',
    'mri_roll3', 'mri_roll6', 'longitude', 'latitude']

non_lagged_features = ['ndvi', 'precipitation_mm', 'temperature_C', 'elevation_m',
    'month_sin', 'month_cos', 'longitude', 'latitude']

X_train, y_train, X_test, y_test, locations_test, _ = xg_split_data(df, non_lagged_features, 2018)
model = train_xgb(X_train, y_train, X_test, y_test)
save_xg_and_logs(model, model_path)

y_pred = evaluate_xg_model(model, X_test, y_test, df[df['year'] > 2018], model_path)
pred_by_loc = build_xg_prediction_dict(locations_test, y_test.values, y_pred)
save_per_location_xg_metrics(pred_by_loc, model_path)

xg_sanity_checks(y_test.values, y_pred)
plot_xg_predictions(pred_by_loc, model_path, 3, False)

export_shap_values(model, X_train, X_test, non_lagged_features, model_path)
export_shap_per_location(model, X_train, X_test, non_lagged_features, locations_test, model_path)
explain_xgboost(model, X_train, X_test, non_lagged_features, model_path)