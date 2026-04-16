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
plot_sample_lstm_predictions(pred_by_loc, 3)


'''df = pd.read_csv(mri_data)
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

df['mri_lag1'] = df.groupby(['longitude','latitude'])['monthly_mri'].shift(1)
#df['mri_lag2'] = df.groupby(['longitude','latitude'])['monthly_mri'].shift(2)

df = df.dropna().reset_index(drop = True)

features = ['ndvi', 'precipitation_mm', 'temperature_C', 'elevation_m',
    'month_sin', 'month_cos', 'mri_lag1']
target = 'monthly_mri'
look_back = 12
horizon = 6

train_df = df[df['year'] <= 2018].copy()
test_df  = df[df['year'] > 2018].copy()

print("Scaling features per location (train only)...")
scalers_X = {}
scaled_train = []
scaled_test = []

# ---- TRAIN SCALING ----
for (lon, lat), group in train_df.groupby(['longitude', 'latitude']):

    group = group.sort_values(['year','month_num']).copy()
    scaler = MinMaxScaler()
    group[features] = scaler.fit_transform(group[features])
    scalers_X[(lon, lat)] = scaler
    scaled_train.append(group)

train_scaled = pd.concat(scaled_train).reset_index(drop=True)

# ---- TEST SCALING ----
print("Applying same scaling to test set...")
for (lon, lat), group in test_df.groupby(['longitude', 'latitude']):

    group = group.sort_values(['year','month_num']).copy()

    if (lon, lat) in scalers_X:

        scaler = scalers_X[(lon, lat)]
        group[features] = scaler.transform(group[features])
        scaled_test.append(group)

test_scaled = pd.concat(scaled_test).reset_index(drop = True)

X, y, locations = create_sequences(test_scaled, features, target, look_back, horizon)

X_train, y_train, _ = create_sequences(train_scaled, features, target, look_back)
X_test, y_test, locations_test = create_sequences(test_scaled, features, target, look_back)

# ---------------------------
# TARGET SCALING
# ---------------------------
print("Scaling target...")

y_train = y_train.reshape(-1, 1)
y_test  = y_test.reshape(-1, 1)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled  = y_scaler.transform(y_test)

# ---------------------------
# TENSORS
# ---------------------------
print("Converting to PyTorch tensors...")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# ---------------------------
# DATALOADERS
# ---------------------------
print("Creating DataLoaders...")

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor),
    batch_size=32
)

# ---------------------------
# MODEL
# ---------------------------
class MRILSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = MRILSTM(X_train.shape[2]).to(device)

# ---------------------------
# TRAINING
# ---------------------------
print("Starting training...")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model_path = os.path.join(DATA_RESULTS, 'lstm')
os.makedirs(model_path, exist_ok=True)

log_csv_path = os.path.join(model_path, 'training_log.csv')

with open(log_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])

patience = 5
best_loss = float('inf')
trigger_times = 0
num_epochs = 50

for epoch in range(num_epochs):

    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(test_loader.dataset)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

    with open(log_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, val_loss])

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        best_model_path = os.path.join(model_path, 'best_model.pt')
        torch.save(model.state_dict(), best_model_path)
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered")
            break

# ---------------------------
# EVALUATION
# ---------------------------
print("Evaluating best model on test set...")

model.load_state_dict(torch.load(best_model_path))
model.eval()

y_pred_list = []
y_true_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        y_pred_list.append(y_pred.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())

y_pred = np.concatenate(y_pred_list).reshape(-1, 1)
y_true = np.concatenate(y_true_list).reshape(-1, 1)

y_pred = y_scaler.inverse_transform(y_pred)
y_true = y_scaler.inverse_transform(y_true)

y_pred = y_pred.ravel()
y_true = y_true.ravel()

# ---------------------------
# METRICS
# ---------------------------
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)

# ---------------------------
# SANITY CHECKS
# ---------------------------
print("Pred std:", y_pred.std())
print("True std:", y_true.std())
print("Pred mean:", y_pred.mean())
print("True mean:", y_true.mean())

# FIXED NAIVE BASELINE (uses lag1, not NDVI)
y_naive = X_test[:, -1, features.index('mri_lag1')]

# ---------------------------
# SAVE METRICS
# ---------------------------
metrics_dict = {'MSE': mse, 'MAE': mae, 'R2': r2}
metrics_csv_path = os.path.join(model_path, 'test_metrics.csv')

with open(metrics_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    for metric, value in metrics_dict.items():
        writer.writerow([metric, value])

# ---------------------------
# PLOTTING
# ---------------------------
pred_by_loc = {}

for (lon, lat), pred, true in zip(locations_test, y_pred, y_true):
    if (lon, lat) not in pred_by_loc:
        pred_by_loc[(lon, lat)] = {'true': [], 'pred': []}

    pred_by_loc[(lon, lat)]['true'].append(true)
    pred_by_loc[(lon, lat)]['pred'].append(pred)

print("Saving per-location R² to CSV...")

per_loc_results = []

for (lon, lat), vals in pred_by_loc.items():
    if len(vals['true']) > 1:
        r2_loc = r2_score(vals['true'], vals['pred'])
        mse_loc = mean_squared_error(vals['true'], vals['pred'])
        mae_loc = mean_absolute_error(vals['true'], vals['pred'])

        per_loc_results.append({
            'longitude': lon,
            'latitude': lat,
            'R2': r2_loc,
            'MSE': mse_loc,
            'MAE': mae_loc,
            'n_samples': len(vals['true'])
        })

per_loc_df = pd.DataFrame(per_loc_results)

per_loc_csv_path = os.path.join(model_path, 'per_location_metrics.csv')
per_loc_df.to_csv(per_loc_csv_path, index=False)

print(f"Saved per-location metrics to: {per_loc_csv_path}")

for loc, vals in list(pred_by_loc.items())[:3]:
    plt.figure(figsize=(8,4))
    plt.plot(vals['true'], label='True MRI')
    plt.plot(vals['pred'], label='Predicted MRI')
    plt.title(f'Location: {loc}')
    plt.legend()
    plt.show()'''