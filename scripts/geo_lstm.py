import configparser
import csv
import os 
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from dream.malmo import create_sequences
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

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

mri_data = os.path.join(DATA_RESULTS, 'mri', 'malaria_risk_index.csv')
df = pd.read_csv(mri_data)
features = ['normalized_parasite_rate', 'normalized_incidence_rate',
            'normalized_net_use', 'normalized_net_access',
            'normalized_mortality_rate']
target = 'mri_value'
look_back = 5

X, y, locations = create_sequences(df, features, target, look_back)

# Scaling the features
print("Scaling features...")
scaler = MinMaxScaler()
n_samples, n_timesteps, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

#Convert to PyTorch tensors
print("Converting to PyTorch tensors...")
X_tensor = torch.tensor(X_scaled, dtype = torch.float32)
y_tensor = torch.tensor(y, dtype = torch.float32).unsqueeze(-1)

# Train/Test Split
train_size = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:train_size].to(device), X_tensor[train_size:].to(device)
y_train, y_test = y_tensor[:train_size].to(device), y_tensor[train_size:].to(device)
locations_test = locations[train_size:]

# Create DataLoader for batching
print("Creating DataLoaders...")
batch_size = 32
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)

test_ds = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size = batch_size)

#Define LSTM model
class MRILSTM(nn.Module):

    def __init__(self, input_size, hidden_size = 64, num_layers = 2):
        super(MRILSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out

model = MRILSTM(n_features).to(device)


# Actual LSTM training loop with early stopping
print("Starting training...")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

model_path = os.path.join(DATA_RESULTS, 'lstm')
os.makedirs(model_path, exist_ok = True)

log_csv_path = os.path.join(model_path, 'training_log.csv')

with open(log_csv_path, mode = 'w', newline = '') as f:
    
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

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():

        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(test_loader.dataset)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
    with open(log_csv_path, mode = 'a', newline = '') as f:

        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, val_loss])

    # Early stopping
    if val_loss < best_loss:

        best_loss = val_loss
        trigger_times = 0
        os.makedirs(model_path, exist_ok = True)
        best_model_path = os.path.join(DATA_RESULTS, 'best_model.pt')
        torch.save(model.state_dict(), best_model_path)  

    else:

        trigger_times += 1
        if trigger_times >= patience:

            print("Early stopping triggered")
            break

# Load best model and evaluate on test set
print("Evaluating best model on test set...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

y_pred_list = []
with torch.no_grad():

    for X_batch, _ in test_loader:

        y_pred = model(X_batch)
        y_pred_list.append(y_pred.cpu().numpy())

y_pred = np.concatenate(y_pred_list)

pred_by_loc = {}
for (lon, lat), pred, true in zip(locations_test, y_pred, y_test.cpu().numpy()):
    if (lon, lat) not in pred_by_loc:
        pred_by_loc[(lon, lat)] = {'true': [], 'pred': []}
    pred_by_loc[(lon, lat)]['true'].append(true[0])
    pred_by_loc[(lon, lat)]['pred'].append(pred[0])

# Plot example
for loc, vals in list(pred_by_loc.items())[:3]:  # plot first 3 locations
    plt.figure(figsize=(8,4))
    plt.plot(vals['true'], label='True MRI')
    plt.plot(vals['pred'], label='Predicted MRI')
    plt.title(f'Location: {loc}')
    plt.xlabel('Time step')
    plt.ylabel('MRI Value')
    plt.legend()
    plt.show()