import configparser
import csv
import os 
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from dream.lstm import *

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')       

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')

# Confirm GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

mri_data = os.path.join(DATA_RESULTS, 'mri', 'malaria_risk_index_monthly.csv')
model_path = os.path.join(DATA_RESULTS, 'lstm')

df = lstm_load_and_prepare(mri_data)
train_df, test_df = lstm_split_data(df, 2018)

features = ['ndvi', 'precipitation_mm', 'temperature_C', 'elevation_m',
    'month_sin', 'month_cos', 'mri_lag1']
target = 'monthly_mri'

train_scaled, test_scaled = lstm_scale_features(train_df, test_df, features)

X_train, y_train, _ = lstm_create_sequences(train_scaled, features, target, 12, 6)
X_test, y_test, locations = lstm_create_sequences(test_scaled, features, target, 12, 6)

y_train, y_test, y_scaler = lstm_scale_target(y_train, y_test)

train_loader, test_loader = lstm_create_dataloaders(X_train, X_test, y_train, y_test, device, 32)

model = MRILSTM(X_train.shape[2]).to(device)
train_lstm_model(model, train_loader, test_loader, model_path, 50)

y_true, y_pred, mse, mae, r2 = evaluate_lstm_model(model, test_loader, y_scaler, model_path)
lstm_metrics(mse, mae, r2, model_path)
lstm_per_location_metrics(locations, y_true, y_pred, model_path)

pred_by_loc = build_lstm_prediction_dict(locations, y_true, y_pred)
shap_df = export_lstm_shap_values(model, X_test, y_test,
    features, model_path, device)
plot_lstm_shap(shap_df, features, model_path)
plot_lstm_predictions(pred_by_loc, model_path, 3, False)