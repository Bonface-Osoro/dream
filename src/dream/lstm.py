import os
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def lstm_create_sequences(df, features, target, look_back, horizon = 1):
    """
    This function creates sequences of features and targets for time series modeling.

    Parameters:
    ----------

        df : dataframe
            Input dataframe with columns for features, target, year, longitude, latitude
        features : list
            List of column names to be used as features
        target : str
            Column name to be used as the target variable
        look_back : int
            Number of past years to include in each sequence
        horizon : int
            How many steps ahead to predict (1 = next step, 3 = 3 months ahead, etc.)

    Returns:
    -------    
        X : np.array
            3D array of shape (num_samples, look_back, num_features)
        y : np.array
            1D array of target values corresponding to each sequence
    """
    
    X, y, locations = [], [], []

    grouped = df.sort_values(['year', 'month_num']).groupby(['longitude', 'latitude'])

    for (lon, lat), group in grouped:

        group = group.sort_values(['year', 'month_num'])

        data = group[features].values
        target_vals = group[target].values

        for i in range(len(group) - look_back - horizon + 1):
            
            X.append(data[i:i + look_back])
            y.append(target_vals[i + look_back + horizon - 1])
            locations.append((lon, lat))

    return np.array(X), np.array(y), locations

def lstm_load_and_prepare(csv_path):
    """
    Loads and prepares the malaria dataset for modeling.
    Parameters:
    ----------
        csv_path : str
            Path to the input CSV file containing the malaria dataset.
    Returns:
    -------
        df : dataframe
            The loaded and prepared dataset.
    """
    print("Loading and preparing data...")

    df = pd.read_csv(csv_path)

    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    df['mri_lag1'] = df.groupby(['longitude','latitude'])['monthly_mri'].shift(1)

    df = df.dropna().reset_index(drop = True)

    return df

def lstm_split_data(df, split_year):
    """Splits the dataset into training and testing sets based on a specified year.
    Parameters:
    ----------
        df : dataframe
            The input dataset.
        split_year : int
            The year to use for splitting the dataset.

    Returns:
    -------
        train_df : dataframe
            The training set.
        test_df : dataframe
            The testing set.
    """
    train_df = df[df['year'] <= split_year].copy()
    test_df  = df[df['year'] > split_year].copy()

    return train_df, test_df

def lstm_scale_features(train_df, test_df, features):
    """Scales the features in the training and testing datasets using MinMaxScaler.
    Parameters:
    ----------
        train_df : dataframe
            The training set.
        test_df : dataframe
            The testing set.
        features : list
            List of column names to be scaled.

    Returns:
    -------
        train_scaled : dataframe
            The training set with scaled features.
        test_scaled : dataframe
            The testing set with scaled features.
    """
    print("Scaling features per location (train only)...")

    scalers_X = {}
    scaled_train = []
    scaled_test = []

    for (lon, lat), group in train_df.groupby(['longitude', 'latitude']):

        group = group.sort_values(['year','month_num']).copy()
        scaler = MinMaxScaler()
        group[features] = scaler.fit_transform(group[features])
        scalers_X[(lon, lat)] = scaler
        scaled_train.append(group)

    train_scaled = pd.concat(scaled_train).reset_index(drop = True)

    print("Applying same scaling to test set...")

    for (lon, lat), group in test_df.groupby(['longitude', 'latitude']):

        group = group.sort_values(['year','month_num']).copy()

        if (lon, lat) in scalers_X:

            scaler = scalers_X[(lon, lat)]
            group[features] = scaler.transform(group[features])
            scaled_test.append(group)

    test_scaled = pd.concat(scaled_test).reset_index(drop = True)

    return train_scaled, test_scaled

def lstm_scale_target(y_train, y_test):
    """Scales the target variable using MinMaxScaler.
    Parameters:
    ----------
        y_train : array
            The training set target values.
        y_test : array
            The testing set target values.

    Returns:
    -------
        y_train_scaled : array
            The training set target values scaled.
        y_test_scaled : array
            The testing set target values scaled.
        scaler : MinMaxScaler
            The fitted scaler object.
    """
    print("Scaling target...")

    y_train = y_train.reshape(-1,1)
    y_test  = y_test.reshape(-1,1)

    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled  = scaler.transform(y_test)

    return y_train_scaled, y_test_scaled, scaler


def lstm_create_dataloaders(X_train, X_test, y_train, y_test, 
                            device, batch_size):
    """Creates PyTorch DataLoaders for training and testing datasets.
    Parameters:
    ----------
        X_train : array
            The training set features.
        X_test : array
            The testing set features.
        y_train : array
            The training set target values.
        y_test : array
            The testing set target values.
        device : torch.device
            The device to move the tensors to.
        batch_size : int
            The batch size for the DataLoaders.

    Returns:
    -------
        train_loader : DataLoader
            The training DataLoader.
        test_loader : DataLoader
            The testing DataLoader.
    """
    print("Converting to PyTorch tensors...")
    
    X_train = torch.tensor(X_train, dtype = torch.float32).to(device)
    X_test  = torch.tensor(X_test, dtype = torch.float32).to(device)

    y_train = torch.tensor(y_train, dtype = torch.float32).to(device)
    y_test  = torch.tensor(y_test, dtype = torch.float32).to(device)

    print("Creating DataLoaders...")

    train_loader = DataLoader(TensorDataset(X_train, y_train), 
                              batch_size = batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), 
                              batch_size = batch_size)

    return train_loader, test_loader


class MRILSTM(nn.Module):
    """LSTM model for predicting monthly Malaria Risk Index (MRI).
    Parameters:
    ----------
        input_size : int
            The number of input features.
        hidden_size : int
            The number of hidden units in the LSTM.
        num_layers : int
            The number of layers in the LSTM.
    """
    def __init__(self, input_size, hidden_size = 64, num_layers = 2):

        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first = True, dropout = 0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm_model(model, train_loader, test_loader, 
                     model_path, num_epochs):
    """Trains the LSTM model and saves the best model based on validation loss.
    Parameters:
    ----------
        model : MRILSTM
            The LSTM model to train.
        train_loader : DataLoader
            The training DataLoader.
        test_loader : DataLoader
            The testing DataLoader.
        model_path : str
            The path to save the trained model.
        num_epochs : int
            The number of epochs to train for.

    Returns:
    -------
        None
    """

    print("Starting training...")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    os.makedirs(model_path, exist_ok = True)
    log_csv_path = os.path.join(model_path, 'lstm_training_log.csv')

    with open(log_csv_path, 'w', newline = '') as f:

        writer = csv.writer(f)
        writer.writerow(['Epoch','Train_Loss','Val_Loss'])

    best_loss = float('inf')
    patience, trigger = 5, 0

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0

        for Xb, yb in train_loader:

            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for Xb, yb in test_loader:

                val_loss += criterion(model(Xb), yb).item() * Xb.size(0)

        val_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch+1}, Train: {train_loss:.5f}, Val: {val_loss:.5f}")

        with open(log_csv_path, 'a', newline = '') as f:

            csv.writer(f).writerow([epoch+1, train_loss, val_loss])

        if val_loss < best_loss:

            best_loss = val_loss
            trigger = 0
            torch.save(model.state_dict(), os.path.join(model_path, 'best_lstm_model.pt'))

        else:

            trigger += 1
            if trigger >= patience:

                print("Early stopping triggered")
                break

def evaluate_lstm_model(model, test_loader, y_scaler, 
                   model_path):
    """Evaluates the trained LSTM model on the test set and computes performance metrics.
    Parameters:
    ----------
        model : MRILSTM
            The LSTM model to evaluate.
        test_loader : DataLoader
            The testing DataLoader.
        y_scaler : StandardScaler
            The scaler used to inverse transform the target values.
        model_path : str
            The path to the trained model.

    Returns:
    -------
        tuple
            A tuple containing the true values, predicted values, and performance metrics.
    """

    print("Evaluating best model...")

    model.load_state_dict(torch.load(os.path.join(model_path,'best_lstm_model.pt')))
    model.eval()

    preds, trues = [], []

    with torch.no_grad():

        for Xb, yb in test_loader:
            preds.append(model(Xb).cpu().numpy())
            trues.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

    y_pred = y_scaler.inverse_transform(y_pred)
    y_true = y_scaler.inverse_transform(y_true)

    y_pred, y_true = y_pred.ravel(), y_true.ravel()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)

    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)

    return y_true, y_pred, mse, mae, r2

def lstm_metrics(mse, mae, r2, model_path):
    """Saves the performance metrics of the LSTM model to a CSV file.
    Parameters:
    ----------
        mse : float
            The mean squared error.
        mae : float
            The mean absolute error.
        r2 : float
            The R-squared score.
        model_path : str
            The path to the trained model.

    Returns:
    -------
        None
    """
    path = os.path.join(model_path, 'lstm_test_metrics.csv')
    with open(path, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric','Value'])
        writer.writerow(['MSE', mse])
        writer.writerow(['MAE', mae])
        writer.writerow(['R2', r2])
        writer.writerow(['model', 'lstm'])


def lstm_per_location_metrics(locations, y_true, 
                              y_pred, model_path):
    """Saves performance metrics for each location to a CSV file.
    Parameters:
    ----------
        locations : list
            A list of (longitude, latitude) tuples.
        y_true : np.ndarray
            The true target values.
        y_pred : np.ndarray
            The predicted target values.
        model_path : str
            The path to the trained model.

    Returns:
    -------
        dict
            A dictionary mapping each location to its performance metrics.
    """

    pred_by_loc = {}

    for (lon, lat), pred, true in zip(locations, y_pred, y_true):

        pred_by_loc.setdefault((lon, lat), {'true':[], 'pred':[]})
        pred_by_loc[(lon, lat)]['true'].append(true)
        pred_by_loc[(lon, lat)]['pred'].append(pred)

    results = []

    for (lon, lat), vals in pred_by_loc.items():

        if len(vals['true']) > 1:
            results.append({
                'longitude': lon,
                'latitude': lat,
                'R2': r2_score(vals['true'], vals['pred']),
                'MSE': mean_squared_error(vals['true'], vals['pred']),
                'MAE': mean_absolute_error(vals['true'], vals['pred']),
                'model': 'lstm',
                'n_samples': len(vals['true'])
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(model_path,'lstm_per_location_metrics.csv'), index = False)

    return pred_by_loc


def build_lstm_prediction_dict(locations, y_pred, y_true):
    """Builds a dictionary mapping each location to its true and predicted values.
    Parameters:
    ----------
        locations : list
            A list of (longitude, latitude) tuples.
        y_pred : np.ndarray
            The predicted target values.
        y_true : np.ndarray
            The true target values.

    Returns:
    -------
        dict
            A dictionary mapping each location to its true and predicted values.
    """

    pred_by_loc = {}

    for (lon, lat), pred, true in zip(locations, y_pred, y_true):

        if (lon, lat) not in pred_by_loc:

            pred_by_loc[(lon, lat)] = {'true': [], 'pred': []}

        pred_by_loc[(lon, lat)]['true'].append(true)
        pred_by_loc[(lon, lat)]['pred'].append(pred)

    return pred_by_loc

def plot_lstm_predictions(pred_by_loc, save_dir, n, show):
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
        filename = f"lstm_predicted_loc_{i+1}_lon_{lon:.4f}_lat_{lat:.4f}.png"
        filepath = os.path.join(save_dir, filename)

        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")

        if show:
            
            plt.show()

        plt.close()