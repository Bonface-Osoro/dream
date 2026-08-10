"""
Federated GeoAI for Cross-Country Malaria Risk Forecasting
Uganda - Zambia - Zimbabwe

Measures the three constructs defined in the theoretical framework:

    Delta_sov(k, g) = R2_centralised(k) - R2_g(k)      sovereignty cost
    Gamma(k, g)     = R2_g(k)           - R2_local(k)  federation gain
    Shapley(k)      = exact, over 2^3 = 8 coalitions   contribution

Six regimes:
    1. climatology   - seasonal mean, no ML, no sharing        (honest floor)
    2. local         - train alone, share nothing              (sovereignty ref)
    3. transfer      - Uganda ships weights, others fine-tune  (asymmetric)
    4. fedavg        - weights averaged, no data moves         (sample or uniform)
    5. fedprox       - fedavg + proximal term for non-IID
    6. centralised   - pool raw data                           (utility ceiling)
"""
import configparser
import os
import copy
import itertools
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')


OUT_DIR = os.path.join(DATA_RESULTS, 'federated')
os.makedirs(OUT_DIR, exist_ok = True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    print(f'[DEVICE] Using: {DEVICE}')
    if DEVICE.type == 'cpu':
        print('[WARNING] No GPU detected — training will be dramatically slower. '
              'If you expect a GPU, check your CUDA install / PyTorch build before '
              'running the full sweep.')
    else:
        print(f'[DEVICE] GPU: {torch.cuda.get_device_name(0)}')

# ── Participants ────────────────────────────────────────────────────────────
# Ordered along the climatic gradient: equatorial -> tropical -> subtropical
CLIENTS = {
    'uganda':   {'path': os.path.join(DATA_RESULTS, 'mri',
                                      'malaria_risk_index_monthly.csv'),
                 'train': (2010, 2017), 'val': (2018, 2018), 'test': (2019, 2020)},
    'zambia':   {'path': os.path.join(DATA_RESULTS, 'mri', 'zambia',
                                      'ZMB_malaria_risk_index_monthly.csv'),
                 'train': (2010, 2017), 'val': (2020, 2020), 'test': (2021, 2022)},
    'zimbabwe': {'path': os.path.join(DATA_RESULTS, 'mri', 'zimbabwe',
                                      'ZWE_malaria_risk_index_monthly.csv'),
                 'train': (2015, 2017), 'val': (2018, 2019), 'test': (2020, 2024)},
}

FEATURES = ['ndvi', 'precipitation_mm', 'temperature_C', 'elevation_m',
            'month_sin', 'month_cos', 'mri_lag1']
TARGET   = 'monthly_mri'

INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS = 7, 32, 1
LOOK_BACK, HORIZON = 12, 6

STRIDE = 3

N_ROUNDS, LOCAL_EPOCHS, LOCAL_LR, BATCH = 60, 2, 0.001, 256

WEIGHT_DECAY = 1e-2

FC_DROPOUT = 0.3

SEEDS    = [0, 1, 2, 3]
MU_GRID  = [0.001, 0.01, 0.1]
EPS_GRID = [1, 8, None]


class MRILSTM(nn.Module):

    def __init__(self, input_size = INPUT_SIZE,
                 hidden = HIDDEN_SIZE, layers = NUM_LAYERS):

        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first = True,
                            dropout = 0.2 if layers > 1 else 0.0)
        self.dropout = nn.Dropout(FC_DROPOUT)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):

        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def load_country(path):

    """
    This function loads a country-level dataset from a CSV file,
    creates cyclical month features using sine and cosine
    transformations, generates a one-month lag of the target variable
    for each spatial location, removes rows with missing values, and
    returns the processed DataFrame.

    Parameters
    ----------
    path : str
        Path to the input CSV file.

    Returns
    -------
    df : pandas.DataFrame
        Processed DataFrame containing the original variables,
        cyclical month features (`month_sin` and `month_cos`),
        the lagged target variable (`mri_lag1`), and only complete
        observations.
    """

    df = pd.read_csv(path)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df['mri_lag1']  = df.groupby(['longitude', 'latitude'])[TARGET].shift(1)
    df = df.dropna().reset_index(drop = True)

    return df


def make_sequences(df):

    """
    This function converts a spatiotemporal DataFrame into input-output
    sequences for time series forecasting. For each unique spatial
    location (longitude, latitude), it creates sliding windows of
    length `LOOK_BACK` using the predictor variables and assigns the
    target value `HORIZON` steps ahead as the prediction target.
    Windows are generated every `STRIDE` months rather than every
    month, to reduce the number of heavily overlapping, near-duplicate
    sequences per location. It also records the calendar year of each
    target value so that sequences can later be assigned to
    train/val/test splits without losing windows that span a split
    boundary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the predictor variables, target
        variable, longitude, latitude, year, and month information,
        covering the FULL timeline for a client (not pre-split).

    Returns
    -------
    X : numpy.ndarray
        Three-dimensional array of shape
        (n_samples, LOOK_BACK, n_features) containing the input
        sequences.

    y : numpy.ndarray
        One-dimensional array containing the target values
        corresponding to each input sequence.

    target_year : numpy.ndarray
        One-dimensional array containing the calendar year of each
        target value, used for date-based train/val/test splitting.
    """

    X, y, target_year = [], [], []
    for _, g in df.sort_values(['year', 'month_num']
                               ).groupby(['longitude', 'latitude']):

        g = g.sort_values(['year', 'month_num'])
        d, t, yrs = g[FEATURES].values, g[TARGET].values, g['year'].values
        for i in range(0, len(g) - LOOK_BACK - HORIZON + 1, STRIDE):

            X.append(d[i:i + LOOK_BACK])
            y.append(t[i + LOOK_BACK + HORIZON - 1])
            target_year.append(yrs[i + LOOK_BACK + HORIZON - 1])

    return np.array(X), np.array(y), np.array(target_year)


def build_clients():

    """
    This function prepares the datasets for each client by loading the
    data, building sliding-window sequences over the client's FULL
    timeline, then assigning each sequence to train/val/test based on
    the calendar year of its target value. The feature scaler is fit
    only on rows falling within the configured training years, then
    applied to the full timeline before windowing so no split sees
    unscaled data. PyTorch DataLoaders are constructed for each split,
    using worker processes and pinned memory to keep the GPU fed
    rather than bottlenecked on data loading.

    Returns
    -------
    clients : dict
        Dictionary containing the processed data for each client. Each
        client includes:

        - 'loaders': PyTorch DataLoaders for the training, validation,
          and testing datasets.
        - 'raw': Tuple of unscaled input sequences and target values
          for each data split.
        - 'y_scaler': Fitted MinMaxScaler used to scale the target
          variable.
        - 'n_train': Number of training samples.
        - 'df_test': Original test DataFrame prior to sequence
          generation.
    """

    clients = {}
    for name, cfg in CLIENTS.items():

        df = load_country(cfg['path'])

        train_mask = (df['year'] >= cfg['train'][0]) & (df['year'] <= cfg['train'][1])

        xs, ys = MinMaxScaler(), MinMaxScaler()
        xs.fit(df.loc[train_mask, FEATURES])
        df[FEATURES] = xs.transform(df[FEATURES])

        X_all, y_all, target_year = make_sequences(df)

        loaders, raw = {}, {}
        for s in ('train', 'val', 'test'):

            mask = (target_year >= cfg[s][0]) & (target_year <= cfg[s][1])
            X, y = X_all[mask], y_all[mask]

            if len(y) == 0:

                raise ValueError(
                    f"{name}: split '{s}' ({cfg[s][0]}-{cfg[s][1]}) produced "
                    f"0 sequences. Needs >= {LOOK_BACK + HORIZON} months of "
                    f"continuous data leading up to and including this range.")

            y_s = ys.fit_transform(y.reshape(-1, 1)) if s == 'train' \
                  else ys.transform(y.reshape(-1, 1))
            loaders[s] = DataLoader(
                TensorDataset(torch.tensor(X, dtype = torch.float32),
                              torch.tensor(y_s, dtype = torch.float32)),
                batch_size = BATCH, shuffle = (s == 'train'),
                num_workers = 2, pin_memory = True, persistent_workers = True)
            raw[s] = (X, y)

        df_test = df[(df['year'] >= cfg['test'][0]) & (df['year'] <= cfg['test'][1])]

        clients[name] = {'loaders': loaders, 'raw': raw, 'y_scaler': ys,
                         'n_train': len(raw['train'][0]),
                         'df_test': df_test}
        print(f'{name:9s} train={clients[name]["n_train"]:>7d}')


    return clients


def local_update(global_state, loader, epochs, mu=0.0, dp_eps=None,
                  patience=10, min_epochs=15, tol=1e-5, verbose=True,
                  max_batches_per_epoch=None, val_loader=None):

    """
    This function performs local model training for a single federated
    learning client. The global model parameters are used to initialize
    the local model, which is then trained for up to `epochs` epochs
    using the client's training data. Optional FedProx and differential
    privacy (DP-SGD) mechanisms can be applied during optimization.

    If `val_loader` is provided, loss on that validation set is
    monitored each epoch and used to decide the "best" epoch and to
    drive early stopping, rather than training loss. Monitoring
    training loss alone allows the model to keep looking like it's
    "improving" while it's actually just memorizing the training set
    — confirmed on Zambia's data, where a model early-stopped on
    training loss reached train R2 as high as 0.997 while test R2 was
    negative. The epoch at which the best (lowest monitored loss)
    state was found is tracked and printed, as a diagnostic for how
    quickly overfitting sets in.

    Parameters
    ----------
    global_state : collections.OrderedDict
        State dictionary containing the parameters of the current
        global model.

    loader : torch.utils.data.DataLoader
        DataLoader containing the client's local training data.

    epochs : int
        Maximum number of local training epochs to perform.

    mu : float, default 0.0
        FedProx proximal regularization strength. If 0, no proximal
        term is applied.

    dp_eps : float or None, default None
        If provided, enables DP-SGD-style noise injection on
        gradients, scaled by 1/dp_eps. If None, no noise is added.

    patience : int, default 10
        Number of consecutive non-improving epochs to tolerate before
        stopping early.

    min_epochs : int, default 15
        Minimum number of epochs to run before early stopping can
        trigger.

    tol : float, default 1e-5
        Minimum absolute decrease in the monitored loss to count as
        an "improvement" for early-stopping purposes.

    verbose : bool, default True
        Whether to print per-epoch progress.

    max_batches_per_epoch : int or None, default None
        If set, caps the number of batches processed per epoch to
        this value. Decouples per-epoch training cost from a
        client's dataset size.

    val_loader : torch.utils.data.DataLoader or None, default None
        If provided, this loader is used to compute validation loss
        each epoch, which drives early stopping and best-state
        selection instead of training loss.

    Returns
    -------
    state_dict : collections.OrderedDict
        State dictionary containing the best model parameters seen
        during training (lowest monitored loss), or the final
        epoch's parameters if patience is None.
    """

    model = MRILSTM().to(DEVICE)
    model.load_state_dict(global_state)
    anchor = [p.detach().clone() for p in model.parameters()]

    opt  = torch.optim.Adam(model.parameters(), lr = LOCAL_LR,
                            weight_decay = WEIGHT_DECAY)
    crit = nn.MSELoss()

    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad_epochs = 0

    for ep in range(epochs):

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        batch_iter = loader if max_batches_per_epoch is None \
                     else itertools.islice(loader, max_batches_per_epoch)

        for Xb, yb in batch_iter:

            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)

            if mu > 0:
                loss = loss + (mu / 2) * sum(
                    ((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))

            loss.backward()

            if dp_eps is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
                sigma = 1.0 / dp_eps
                for p in model.parameters():

                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * sigma * 0.01

            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else float('nan')

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += crit(model(Xb), yb).item()
                    n_val_batches += 1
            monitor_loss = val_loss / n_val_batches if n_val_batches > 0 else float('nan')
        else:
            monitor_loss = avg_loss

        if verbose:
            msg = f'    epoch {ep+1}/{epochs}  train_loss={avg_loss:.5f}'
            if val_loader is not None:
                msg += f'  val_loss={monitor_loss:.5f}'
            msg += f'  batches={n_batches}'
            print(msg)

        if monitor_loss < best_loss - tol:
            best_loss = monitor_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep + 1
            bad_epochs = 0
        else:
            bad_epochs += 1

        if patience is not None and ep + 1 >= min_epochs and bad_epochs >= patience:
            if verbose:
                print(f'    [early stop] no improvement > {tol} for '
                      f'{patience} epochs, stopping at epoch {ep+1}/{epochs} '
                      f'(best_loss={best_loss:.5f}, best_epoch={best_epoch})')
            break

    if verbose and (patience is None or bad_epochs < patience):
        print(f'    [reached max epochs] best_loss={best_loss:.5f}, '
              f'best_epoch={best_epoch}')

    return best_state


def aggregate(states, weights):
    """
    This function aggregates the model parameters from multiple
    federated learning clients using a weighted average (FedAvg).
    Each client's contribution is weighted according to the
    corresponding value in `weights`.

    Parameters
    ----------
    states : list of collections.OrderedDict
        List of model state dictionaries returned by the clients
        after local training.

    weights : list of int or float
        List of aggregation weights, typically representing the
        number of training samples for each client.

    Returns
    -------
    out : collections.OrderedDict
        State dictionary containing the aggregated global model
        parameters.
    """
    total = sum(weights)
    out = copy.deepcopy(states[0])
    for k in out:

        out[k] = sum(s[k].float() * (w / total)
                     for s, w in zip(states, weights))

    return out


def evaluate(state, client, split='test'):
    """
    This function evaluates a trained model on a client's dataset
    split (test by default, or val for round-level monitoring during
    federated training). Predictions are transformed back to the
    original scale before computing evaluation metrics.

    Parameters
    ----------
    state : collections.OrderedDict
        State dictionary containing the trained model parameters.

    client : dict
        Dictionary containing the client's DataLoaders and the
        fitted target scaler (`y_scaler`).

    split : str, default 'test'
        Which client DataLoader split to evaluate on ('train', 'val',
        or 'test').

    Returns
    -------
    metrics : dict
        Dictionary containing the model evaluation metrics:

        - 'R2' : Coefficient of determination (R²).
        - 'RMSE' : Root Mean Squared Error.
        - 'MAE' : Mean Absolute Error.
    """
    model = MRILSTM().to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    P, T = [], []
    with torch.no_grad():

        for Xb, yb in client['loaders'][split]:

            P.append(model(Xb.to(DEVICE)).cpu().numpy())
            T.append(yb.numpy())

    ys = client['y_scaler']
    yp = ys.inverse_transform(np.concatenate(P)).ravel()
    yt = ys.inverse_transform(np.concatenate(T)).ravel()

    return {'R2':   r2_score(yt, yp),
            'RMSE': np.sqrt(mean_squared_error(yt, yp)),
            'MAE':  mean_absolute_error(yt, yp)}


def federate(clients, members, mu=0.0, weighting='sample',
             dp_eps=None, seed=0, log_rounds=False,
             round_patience=8, min_rounds=15, round_tol=1e-4,
             max_batches_per_epoch=500):
    """
    This function trains a global federated learning model using the
    specified client datasets. During each communication round, the
    selected clients perform local model updates, their parameters are
    aggregated using Federated Averaging (FedAvg), and the resulting
    global model is redistributed. Optional FedProx, differential
    privacy (DP-SGD), and round-by-round evaluation are supported.

    Round-level early stopping: after each round, the mean validation
    R2 across `members` is computed. If it fails to improve by more
    than `round_tol` for `round_patience` consecutive rounds (once at
    least `min_rounds` have run), training stops and the best-seen
    global state is used for final evaluation.

    Per-round local training is also capped via
    `max_batches_per_epoch`, so clients with much larger datasets
    (e.g. Zambia) don't dominate every round's wall-clock cost.

    Note: per-round local_update calls here don't pass val_loader,
    since LOCAL_EPOCHS=2 is too short for within-round validation
    monitoring to be meaningful — round-level early stopping (based
    on val R2 across the whole federation) is the mechanism that
    protects against overfitting at this level instead.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets, DataLoaders,
        and metadata for each client.

    members : list of str
        List of client names participating in the federation.

    mu : float, default 0.0
        FedProx proximal regularization strength.

    weighting : str, default 'sample'
        'sample' weights clients by training set size; anything else
        uses uniform weighting.

    dp_eps : float or None, default None
        DP-SGD privacy budget passed through to local_update.

    seed : int, default 0
        Random seed for global model initialization.

    log_rounds : bool, default False
        If True, records per-round test-set metrics for each member.

    round_patience : int, default 8
        Number of consecutive non-improving rounds (on mean val R2)
        to tolerate before stopping early.

    min_rounds : int, default 15
        Minimum number of communication rounds to run before early
        stopping can trigger.

    round_tol : float, default 1e-4
        Minimum absolute improvement in mean val R2 to reset patience.

    max_batches_per_epoch : int or None, default 500
        Passed through to local_update each round; caps per-client
        per-epoch batch count so round cost doesn't scale with the
        largest client's dataset size.

    Returns
    -------
    final : dict
        Dictionary containing the final evaluation metrics (R²,
        RMSE, and MAE) for each participating client, using the
        best-seen global model.

    history : pandas.DataFrame
        DataFrame containing the evaluation metrics recorded after
        each communication round. Empty if `log_rounds` is False.

    gstate : collections.OrderedDict
        State dictionary containing the best global model parameters
        found during training (by mean validation R2).
    """
    torch.manual_seed(seed)
    gstate = MRILSTM().to(DEVICE).state_dict()

    if weighting == 'sample':

        w = [clients[m]['n_train'] for m in members]

    else:

        w = [1.0] * len(members)

    best_gstate = copy.deepcopy(gstate)
    best_val_r2 = -float('inf')
    bad_rounds = 0

    history = []
    for rnd in range(1, N_ROUNDS + 1):

        states = [local_update(gstate, clients[m]['loaders']['train'],
                               LOCAL_EPOCHS, mu, dp_eps, verbose=False,
                               max_batches_per_epoch=max_batches_per_epoch)
                  for m in members]
        gstate = aggregate(states, w)

        if log_rounds:

            for m in members:
                history.append({'round': rnd, 'client': m,
                                **evaluate(gstate, clients[m])})

        val_r2 = float(np.mean([evaluate(gstate, clients[m], split='val')['R2']
                                for m in members]))

        if val_r2 > best_val_r2 + round_tol:
            best_val_r2 = val_r2
            best_gstate = copy.deepcopy(gstate)
            bad_rounds = 0
        else:
            bad_rounds += 1

        if rnd >= min_rounds and bad_rounds >= round_patience:
            print(f'    [federate early stop] round {rnd}/{N_ROUNDS}, '
                  f'best_val_R2={best_val_r2:.4f}')
            break

    final = {m: evaluate(best_gstate, clients[m]) for m in members}

    return final, pd.DataFrame(history), best_gstate


def climatology(clients):
    """
    This function evaluates a climatology baseline for each client by
    predicting the mean target value for each spatial location and
    calendar month using the training data. The climatological
    predictions are compared with the client's test data to compute
    regression performance metrics.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed client datasets,
        including the test DataFrame for each client.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame containing the climatology baseline performance for
        each client with the following columns:

        - 'regime' : Baseline method ('climatology').
        - 'client' : Client name.
        - 'R2' : Coefficient of determination (R²).
        - 'RMSE' : Root Mean Squared Error.
        - 'MAE' : Mean Absolute Error.
    """
    rows = []
    for name, c in clients.items():

        df_tr = load_country(CLIENTS[name]['path'])
        tr = df_tr[(df_tr['year'] >= CLIENTS[name]['train'][0]) &
                   (df_tr['year'] <= CLIENTS[name]['train'][1])]
        clim = tr.groupby(['longitude', 'latitude', 'month_num'])[TARGET] \
                 .mean().rename('clim').reset_index()
        te = c['df_test'].merge(clim, on = ['longitude', 'latitude', 'month_num'],
                                how = 'left')
        te['clim'] = te['clim'].fillna(tr[TARGET].mean())
        rows.append({'regime': 'climatology', 'client': name,
                     'R2':   r2_score(te[TARGET], te['clim']),
                     'RMSE': np.sqrt(mean_squared_error(te[TARGET], te['clim'])),
                     'MAE':  mean_absolute_error(te[TARGET], te['clim'])})


    return pd.DataFrame(rows)


def local_only(clients, seed):
    """
    This function trains an independent model for each client using
    only its local training data. No federated aggregation is
    performed. The trained models are evaluated on the corresponding
    client's test dataset. Per-epoch batch count is capped so clients
    with much larger datasets (e.g. Zambia) don't dominate wall-clock
    time relative to smaller clients. Validation-based early stopping
    (val_loader) is used to prevent overfitting.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets, DataLoaders,
        and metadata for each client.

    seed : int
        Random seed used to initialize the model for reproducibility.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame containing the evaluation results for each client
        with the following columns:

        - 'regime' : Training strategy ('local').
        - 'client' : Client name.
        - 'seed' : Random seed used for training.
        - 'R2' : Coefficient of determination (R²).
        - 'RMSE' : Root Mean Squared Error.
        - 'MAE' : Mean Absolute Error.
    """
    rows = []
    for name, c in clients.items():

        torch.manual_seed(seed)
        st = local_update(MRILSTM().to(DEVICE).state_dict(),
                          c['loaders']['train'], N_ROUNDS * LOCAL_EPOCHS,
                          max_batches_per_epoch=1000,
                          val_loader=c['loaders']['val'])
        rows.append({'regime': 'local', 'client': name, 'seed': seed,
                     **evaluate(st, c)})


    return pd.DataFrame(rows)


def centralised(clients, seed):
    """
    This function trains a centralized model by combining the training
    data from all clients into a single dataset. The model is trained
    on the pooled data without considering client boundaries and is
    subsequently evaluated separately on each client's test dataset.
    Per-epoch batch count is capped for the same reason as local_only.
    A pooled validation loader (all clients' val splits combined) is
    used for early stopping, consistent with local_only.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets, including raw
        training sequences, target scalers, DataLoaders, and test
        data for each client.

    seed : int
        Random seed used to initialize the model and ensure
        reproducibility.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame containing the centralized model evaluation results
        for each client with the following columns:

        - 'regime' : Training strategy ('centralised').
        - 'client' : Client name.
        - 'seed' : Random seed used for training.
        - 'R2' : Coefficient of determination (R²).
        - 'RMSE' : Root Mean Squared Error.
        - 'MAE' : Mean Absolute Error.
    """

    torch.manual_seed(seed)
    Xs = [torch.tensor(c['raw']['train'][0], dtype = torch.float32)
          for c in clients.values()]
    ys = [torch.tensor(c['y_scaler'].transform(c['raw']['train'][1].reshape(-1, 1)),
                       dtype = torch.float32) for c in clients.values()]
    loader = DataLoader(TensorDataset(torch.cat(Xs), torch.cat(ys)),
                        batch_size = BATCH, shuffle = True,
                        num_workers = 2, pin_memory = True, persistent_workers = True)

    Xs_val = [torch.tensor(c['raw']['val'][0], dtype = torch.float32)
              for c in clients.values()]
    ys_val = [torch.tensor(c['y_scaler'].transform(c['raw']['val'][1].reshape(-1, 1)),
                           dtype = torch.float32) for c in clients.values()]
    val_loader = DataLoader(TensorDataset(torch.cat(Xs_val), torch.cat(ys_val)),
                            batch_size = BATCH, shuffle = False,
                            num_workers = 2, pin_memory = True, persistent_workers = True)

    st = local_update(MRILSTM().to(DEVICE).state_dict(), loader,
                      N_ROUNDS * LOCAL_EPOCHS, max_batches_per_epoch=1000,
                      val_loader=val_loader)


    return pd.DataFrame([{'regime': 'centralised', 'client': n, 'seed': seed,
                          **evaluate(st, c)} for n, c in clients.items()])


def mmd(A, B, gamma = None, n = 2000, seed = 0):
    """
    This function computes the Maximum Mean Discrepancy (MMD) between
    two datasets using a Gaussian Radial Basis Function (RBF) kernel.
    MMD measures the distributional difference between two sets of
    samples and can be used to quantify domain shift between client
    datasets in federated learning.

    Parameters
    ----------
    A : numpy.ndarray
        First dataset of samples with shape (n_samples, n_features).

    B : numpy.ndarray
        Second dataset of samples with shape (n_samples, n_features).

    Returns
    -------
    mmd_value : float
        Maximum Mean Discrepancy value between datasets A and B.
        Larger values indicate greater distributional differences.
    """
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace = False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace = False)]
    if gamma is None:
        gamma = 1.0 / A.shape[1]

    def K(P, Q):

        d = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)

        return np.exp(-gamma * d)

    return K(A, A).mean() + K(B, B).mean() - 2 * K(A, B).mean()


def pairwise_heterogeneity(clients):
    """
    This function computes pairwise distributional heterogeneity
    between clients using the Maximum Mean Discrepancy (MMD) metric.
    For each client, the temporal input sequences and target values
    are summarized into feature representations, and the MMD is
    calculated between every pair of clients to quantify differences
    in their data distributions.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets for each client,
        including training input sequences and target values.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame containing pairwise client heterogeneity values with
        the following columns:

        - 'pair' : Pair of clients being compared.
        - 'mmd' : Maximum Mean Discrepancy value between the two
          client distributions. Larger values indicate greater
          distributional differences.
    """
    joint = {}
    for n, c in clients.items():

        X, y = c['raw']['train']
        joint[n] = np.hstack([X.mean(axis = 1), y.reshape(-1, 1)])

    rows = []
    for a, b in itertools.combinations(clients, 2):

        rows.append({'pair': f'{a}-{b}',
                     'mmd': float(mmd(joint[a], joint[b]))})


    return pd.DataFrame(rows)


def shapley(clients, mu, seed):
    """
    This function computes the Shapley value of each client in a
    federated learning system. The Shapley value quantifies the
    contribution of an individual client to the overall model
    performance by evaluating the marginal improvement in performance
    across all possible client coalitions.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets, DataLoaders,
        and metadata for each client.

    mu : float
        FedProx proximal regularization parameter used during
        federated training.

    seed : int
        Random seed used to ensure reproducibility of model training.

    Returns
    -------
    phi : dict
        Dictionary containing the Shapley value for each client.
        Larger values indicate greater contribution to the federated
        model performance based on the evaluated R² improvement.
    """
    names = list(clients)
    N = len(names)

    v = {(): 0.0}
    for r in range(1, N + 1):

        for coal in itertools.combinations(names, r):

            if r == 1:

                m = evaluate(local_update(MRILSTM().to(DEVICE).state_dict(),
                                          clients[coal[0]]['loaders']['train'],
                                          N_ROUNDS * LOCAL_EPOCHS,
                                          max_batches_per_epoch=1000,
                                          val_loader=clients[coal[0]]['loaders']['val']),
                             clients[coal[0]])
                v[coal] = m['R2']

            else:

                final, _, _ = federate(clients, list(coal), mu = mu, seed = seed)
                v[coal] = float(np.mean([final[m]['R2'] for m in coal]))

    from math import factorial
    phi = {}
    for k in names:

        others = [x for x in names if x != k]
        tot = 0.0
        for r in range(len(others) + 1):

            for S in itertools.combinations(others, r):

                w = factorial(len(S)) * factorial(N - len(S) - 1) / factorial(N)
                tot += w * (v[tuple(sorted(S + (k,)))] - v[tuple(sorted(S))])
        phi[k] = tot


    return phi


if __name__ == '__main__':
    import time
    script_start = time.time()

    clients = build_clients()
    members = list(clients)
    out = []

    het = pairwise_heterogeneity(clients)
    het.to_csv(f'{OUT_DIR}/heterogeneity_mmd.csv', index = False)
    print(het)

    out.append(climatology(clients))

    # ── Seed-repeated regimes ────────────────────────────────────────────
    # local, centralised, and fedavg/fedprox feed the headline constructs
    # (delta_sov, gamma) and are the size-1/size-3 coalitions Shapley
    # compares against — these need repeat seeds so a reported gap is
    # distinguishable from ordinary init-to-init noise.
    for seed in SEEDS:
        seed_start = time.time()
        print(f'\n===== seed {seed} (constructs-critical regimes) =====')
        out.append(local_only(clients, seed))
        out.append(centralised(clients, seed))

        for wt in ('sample', 'uniform'):
            fa_start = time.time()
            fa, hist, _ = federate(clients, members, mu = 0.0,
                                   weighting = wt, seed = seed, log_rounds = True)
            print(f'  [fedavg_{wt}] took {(time.time()-fa_start)/60:.1f} min')
            out.append(pd.DataFrame([{'regime': f'fedavg_{wt}', 'client': k,
                                      'seed': seed, **m} for k, m in fa.items()]))
            hist.to_csv(f'{OUT_DIR}/rounds_fedavg_{wt}_s{seed}.csv', index = False)

        for mu in MU_GRID:
            fp_start = time.time()
            fp, _, _ = federate(clients, members, mu = mu, seed = seed)
            print(f'  [fedprox_mu{mu}] took {(time.time()-fp_start)/60:.1f} min')
            out.append(pd.DataFrame([{'regime': f'fedprox_mu{mu}', 'client': k,
                                      'seed': seed, **m} for k, m in fp.items()]))

        print(f'  === seed {seed} total: {(time.time()-seed_start)/60:.1f} min, '
              f'script elapsed: {(time.time()-script_start)/3600:.2f} hr ===')

    # ── Exploratory sweeps: single representative seed ──────────────────
    # Sub-federations and the DP-epsilon grid are read as trends across a
    # parameter (which coalition, how much privacy budget), not as single
    # point estimates feeding a construct — a single seed is enough to see
    # the shape of the trend without the full multi-seed cost.
    explore_seed = SEEDS[0]
    explore_start = time.time()
    print(f'\n===== exploratory sweeps (seed {explore_seed}) =====')

    for coal in itertools.combinations(members, 2):
        f2_start = time.time()
        f2, _, _ = federate(clients, list(coal), mu = 0.01, seed = explore_seed)
        print(f'  [fed_{"+".join(coal)}] took {(time.time()-f2_start)/60:.1f} min')
        out.append(pd.DataFrame([{'regime': f'fed_{"+".join(coal)}',
                                  'client': k, 'seed': explore_seed, **m}
                                 for k, m in f2.items()]))

    for eps in EPS_GRID:
        dp_start = time.time()
        fd, _, _ = federate(clients, members, mu = 0.01,
                            dp_eps = eps, seed = explore_seed)
        tag = f'dp_eps{eps}' if eps else 'dp_none'
        print(f'  [{tag}] took {(time.time()-dp_start)/60:.1f} min')
        out.append(pd.DataFrame([{'regime': tag, 'client': k, 'seed': explore_seed,
                                  **m} for k, m in fd.items()]))

    print(f'  === exploratory sweeps total: '
          f'{(time.time()-explore_start)/60:.1f} min ===')

    res = pd.concat(out, ignore_index = True)
    res.to_csv(f'{OUT_DIR}/federated_results.csv', index = False)

    phi = shapley(clients, mu = 0.01, seed = explore_seed)
    pd.DataFrame([{'client': k, 'shapley': v} for k, v in phi.items()]) \
      .to_csv(f'{OUT_DIR}/shapley.csv', index = False)
    print('\nShapley:', phi)

    piv = res.pivot_table(index = ['client', 'seed'], columns = 'regime',
                          values = 'R2')
    piv['delta_sov_fedprox'] = piv['centralised'] - piv['fedprox_mu0.01']
    piv['gamma_fedprox']     = piv['fedprox_mu0.01'] - piv['local']
    piv.reset_index().to_csv(f'{OUT_DIR}/constructs.csv', index = False)

    print(f'\nDone -> federated_results.csv, constructs.csv, shapley.csv '
          f'(total {(time.time()-script_start)/3600:.2f} hr)')