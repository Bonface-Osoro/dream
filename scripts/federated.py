"""
Federated GeoAI for Cross-Country Malaria Risk Forecasting
Uganda - Zambia - Zimbabwe

Measures the three constructs defined in the theoretical framework:

    Delta_sov(k, g) = R2_centralised(k) - R2_g(k)      sovereignty cost
    Gamma(k, g)     = R2_g(k)           - R2_local(k)  federation gain
    Shapley(k)      = exact, over 2^3 = 8 coalitions   contribution

Regimes in this version:
    1. climatology  - seasonal mean, no ML, no sharing            (honest floor)
    2. local        - train alone, share nothing                  (sovereignty ref)
    3. transfer     - single-source, one-time weight transfer,
                       run with EACH of the three countries as the
                       source in turn (source column identifies
                       which). For each source/target pair, both a
                       zero-shot (no fine-tuning) and a fine-tuned
                       evaluation are recorded (mode column), to
                       diagnose whether fine-tuning is simply
                       re-converging to the local optimum regardless
                       of initialization — see notes on transfer().
    4. fedavg       - weights averaged, no data moves               (sample or uniform)
    5. centralised  - pool raw data                                 (utility ceiling)

Note: FedProx (mu grid), the DP-epsilon sweep, pairwise
sub-federations, and the pooled-source reverse_transfer regime from
earlier versions of this script have been removed from the __main__
sweep. reverse_transfer() is retained as a function (in case it's
needed again) but is no longer called — it has been superseded by
looping transfer() over all three source countries, which isolates
"which country is the source" as a single variable instead of
conflating it with pooling multiple source countries' raw data.
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

# Countries tested as a transfer SOURCE, in turn, in __main__. Each
# non-source country becomes a target for that source.
TRANSFER_SOURCES = ['uganda', 'zambia', 'zimbabwe']

# Retained for reverse_transfer() only (no longer called by __main__).
REVERSE_TRANSFER_SOURCES = ['zambia', 'zimbabwe']
REVERSE_TRANSFER_TARGET = 'uganda'

FEATURES = ['ndvi', 'precipitation_mm', 'temperature_C', 'elevation_m',
            'month_sin', 'month_cos', 'mri_lag1']
TARGET   = 'monthly_mri'

INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS = 7, 32, 1
LOOK_BACK, HORIZON = 12, 6

STRIDE = 3

N_ROUNDS, LOCAL_EPOCHS, LOCAL_LR, BATCH = 60, 2, 0.001, 256

# L2 regularization on the optimizer.
WEIGHT_DECAY = 1e-2

FC_DROPOUT = 0.3

TRANSFER_FINETUNE_EPOCHS = 40

SEEDS = [0, 1, 2, 3]


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


def build_pooled_loader(clients, members, split='train', batch_size=BATCH,
                         shuffle=True):
    """
    This function builds a single DataLoader by pooling the raw
    sequences of multiple clients for a given split. Each client's
    target values are scaled using that client's OWN fitted y_scaler
    before pooling — this keeps each client's target values on a
    comparable relative scale without requiring one global scaler
    fitted after the fact.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets for each client,
        as returned by build_clients().

    members : list of str
        Names of the clients to pool together.

    split : str, default 'train'
        Which split ('train' or 'val') to pool from `raw`.

    batch_size : int, default BATCH
        Batch size for the resulting DataLoader.

    shuffle : bool, default True
        Whether to shuffle the pooled dataset.

    Returns
    -------
    loader : torch.utils.data.DataLoader
        DataLoader over the pooled, scaled data from all `members`.
    """
    Xs = [torch.tensor(clients[m]['raw'][split][0], dtype = torch.float32)
          for m in members]
    ys = [torch.tensor(
              clients[m]['y_scaler'].transform(
                  clients[m]['raw'][split][1].reshape(-1, 1)),
              dtype = torch.float32)
          for m in members]

    return DataLoader(
        TensorDataset(torch.cat(Xs), torch.cat(ys)),
        batch_size = batch_size, shuffle = shuffle,
        num_workers = 2, pin_memory = True, persistent_workers = True)


def local_update(global_state, loader, epochs, mu=0.0, dp_eps=None,
                  patience=10, min_epochs=15, tol=1e-5, verbose=True,
                  max_batches_per_epoch=None, val_loader=None):

    """
    This function performs local model training, starting from
    `global_state`, for up to `epochs` epochs on `loader`. Used for
    from-scratch local/centralised training (starting from a freshly
    initialized model), for federated rounds, and for fine-tuning in
    the transfer regime (starting from another client's already-
    trained weights).

    If `val_loader` is provided, loss on that validation set is
    monitored each epoch and used to decide the "best" epoch and to
    drive early stopping, rather than training loss.

    Parameters
    ----------
    global_state : collections.OrderedDict
        State dictionary containing the parameters to initialize the
        model with.

    loader : torch.utils.data.DataLoader
        DataLoader containing the training data to fit on.

    epochs : int
        Maximum number of training epochs to perform.

    mu : float, default 0.0
        FedProx-style proximal regularization strength. Retained here
        for backward compatibility with federate(), but unused by any
        regime in this version's __main__ sweep.

    dp_eps : float or None, default None
        If provided, enables DP-SGD-style noise injection on
        gradients. Unused in this version's sweep.

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
        Whether to print per-epoch progress, including the epoch at
        which the best monitored loss was found (best_epoch) — useful
        for diagnosing how quickly a fine-tuning run re-converges,
        i.e. how much the initialization actually mattered.

    max_batches_per_epoch : int or None, default None
        If set, caps the number of batches processed per epoch to
        this value.

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
    split. Predictions are transformed back to the original scale
    before computing evaluation metrics.

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
    This function trains a global federated learning model (FedAvg,
    or FedProx if mu > 0) using the specified client datasets. During
    each communication round, the selected clients perform local
    model updates, their parameters are aggregated using Federated
    Averaging, and the resulting global model is redistributed.

    Round-level early stopping: after each round, the mean validation
    R2 across `members` is computed. If it fails to improve by more
    than `round_tol` for `round_patience` consecutive rounds (once at
    least `min_rounds` have run), training stops and the best-seen
    global state is used for final evaluation.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets, DataLoaders,
        and metadata for each client.

    members : list of str
        List of client names participating in the federation.

    mu : float, default 0.0
        FedProx proximal regularization strength. 0.0 = plain FedAvg.

    weighting : str, default 'sample'
        'sample' weights clients by training set size; anything else
        uses uniform weighting.

    dp_eps : float or None, default None
        DP-SGD privacy budget passed through to local_update. Unused
        in this version's sweep.

    seed : int, default 0
        Random seed for global model initialization.

    log_rounds : bool, default False
        If True, records per-round test-set metrics for each member.

    round_patience : int, default 8
        Number of consecutive non-improving rounds to tolerate before
        stopping early.

    min_rounds : int, default 15
        Minimum number of communication rounds to run before early
        stopping can trigger.

    round_tol : float, default 1e-4
        Minimum absolute improvement in mean val R2 to reset patience.

    max_batches_per_epoch : int or None, default 500
        Passed through to local_update each round; caps per-client
        per-epoch batch count.

    Returns
    -------
    final : dict
        Dictionary containing the final evaluation metrics for each
        participating client, using the best-seen global model.

    history : pandas.DataFrame
        Per-round metrics, if log_rounds is True.

    gstate : collections.OrderedDict
        Best global model parameters found during training.
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
    calendar month using the training data.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed client datasets,
        including the test DataFrame for each client.

    Returns
    -------
    results : pandas.DataFrame
        Columns: 'regime' ('climatology'), 'client', 'R2', 'RMSE',
        'MAE'.
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
    only its local training data. No federated aggregation or
    weight-sharing is performed. Validation-based early stopping is
    used to prevent overfitting.

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
        Columns: 'regime' ('local'), 'client', 'seed', 'R2', 'RMSE',
        'MAE'.
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


def transfer(clients, seed, source, finetune_epochs=TRANSFER_FINETUNE_EPOCHS):
    """
    This function implements the "transfer" regime for a single given
    `source` country: an asymmetric, one-directional knowledge
    transfer rather than a federated (round-trip) aggregation. A
    model is first trained from scratch on the `source` client's own
    data (using the same procedure as local_only). That trained
    model's weights are then shipped once to every OTHER client.

    For each non-source client, TWO evaluations are recorded, to
    diagnose how much (if anything) the transferred initialization
    actually contributes versus fine-tuning simply re-converging to
    the target's own local optimum:

    - 'zeroshot' : the source model evaluated DIRECTLY on the
      target's test set, with NO fine-tuning at all. This measures
      the raw, as-is transferability of the source model.
    - 'finetuned' : the source model after fine-tuning on the
      target's own training data for up to `finetune_epochs` epochs
      (with validation-based early stopping). This measures
      performance after adaptation.

    If 'finetuned' performance closely matches the target's own
    'local' regime result regardless of which source was used, that
    indicates the fine-tuning budget is large enough to erase the
    initialization signal — i.e. the transfer regime is not really
    testing what the source model contributes, just how well the
    target can re-learn on its own. The 'zeroshot' numbers are the
    direct evidence for or against this interpretation.

    The source client's own row uses its already-trained local model,
    evaluated the same way as every other regime, with mode='source'
    (there is nothing to fine-tune from, since it originates the
    transferred weights).

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets, DataLoaders,
        and metadata for each client.

    seed : int
        Random seed used for both the source's training and each
        target's fine-tuning, for reproducibility.

    source : str
        Name of the client whose trained weights are transferred to
        every other client.

    finetune_epochs : int, default TRANSFER_FINETUNE_EPOCHS
        Maximum number of fine-tuning epochs for non-source clients.

    Returns
    -------
    results : pandas.DataFrame
        Columns: 'regime' (always 'transfer'), 'client', 'source',
        'mode' ('source' / 'zeroshot' / 'finetuned'), 'seed', 'R2',
        'RMSE', 'MAE'.
    """
    if source not in clients:
        raise ValueError(f"transfer(): source client '{source}' not found "
                          f"in clients dict (available: {list(clients)})")

    torch.manual_seed(seed)
    print(f'  [transfer:{source}] training source model on {source}...')
    source_state = local_update(
        MRILSTM().to(DEVICE).state_dict(),
        clients[source]['loaders']['train'], N_ROUNDS * LOCAL_EPOCHS,
        max_batches_per_epoch=1000,
        val_loader=clients[source]['loaders']['val'])

    rows = [{'regime': 'transfer', 'client': source, 'source': source,
             'mode': 'source', 'seed': seed,
             **evaluate(source_state, clients[source])}]

    for name, c in clients.items():

        if name == source:
            continue

        # Zero-shot: source weights evaluated directly on target's
        # test set, no fine-tuning. Cheap (no training) — pure
        # diagnostic of raw transferability.
        zeroshot_metrics = evaluate(source_state, c)
        rows.append({'regime': 'transfer', 'client': name, 'source': source,
                     'mode': 'zeroshot', 'seed': seed, **zeroshot_metrics})

        torch.manual_seed(seed)
        print(f'  [transfer:{source}] fine-tuning on {name} from '
              f'{source} weights...')
        finetuned_state = local_update(
            source_state, c['loaders']['train'], finetune_epochs,
            max_batches_per_epoch=1000,
            val_loader=c['loaders']['val'])

        rows.append({'regime': 'transfer', 'client': name, 'source': source,
                     'mode': 'finetuned', 'seed': seed,
                     **evaluate(finetuned_state, c)})

    return pd.DataFrame(rows)


def reverse_transfer(clients, seed, sources=None, target=None,
                      finetune_epochs=TRANSFER_FINETUNE_EPOCHS):
    """
    NOTE: no longer called from __main__ — superseded by looping
    transfer() over each of the three countries as a single source
    (see TRANSFER_SOURCES), which isolates "which country is the
    source" as a single variable rather than conflating direction
    with pooling multiple source countries' raw data together (which
    this function does, and which is not privacy-preserving for the
    pooled `sources`). Retained here in case a pooled-source
    comparison is wanted again later.

    Pools `sources`' (default Zambia + Zimbabwe) raw training data to
    train ONE shared source model from scratch, then ships its
    weights once to `target` (default Uganda), which fine-tunes its
    own copy.

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets, DataLoaders,
        and metadata for each client.

    seed : int
        Random seed used for both the pooled source training and the
        target's fine-tuning, for reproducibility.

    sources : list of str or None, default None
        Names of the clients whose data is pooled to train the shared
        source model. Falls back to REVERSE_TRANSFER_SOURCES if None.

    target : str or None, default None
        Name of the client that fine-tunes from the pooled source
        model's weights. Falls back to REVERSE_TRANSFER_TARGET if
        None.

    finetune_epochs : int, default TRANSFER_FINETUNE_EPOCHS
        Maximum number of fine-tuning epochs for the target client.

    Returns
    -------
    results : pandas.DataFrame
        Columns: 'regime' ('reverse_transfer'), 'client', 'seed',
        'R2', 'RMSE', 'MAE'.
    """
    sources = sources if sources is not None else REVERSE_TRANSFER_SOURCES
    target = target if target is not None else REVERSE_TRANSFER_TARGET

    missing = [s for s in sources + [target] if s not in clients]
    if missing:
        raise ValueError(f"reverse_transfer(): client(s) not found: {missing} "
                          f"(available: {list(clients)})")
    if target in sources:
        raise ValueError("reverse_transfer(): target must not also be a source "
                          f"(target={target}, sources={sources})")

    torch.manual_seed(seed)
    pooled_train = build_pooled_loader(clients, sources, split='train')
    pooled_val   = build_pooled_loader(clients, sources, split='val', shuffle=False)

    print(f'  [reverse_transfer] training pooled source model on '
          f'{"+".join(sources)}...')
    source_state = local_update(
        MRILSTM().to(DEVICE).state_dict(), pooled_train,
        N_ROUNDS * LOCAL_EPOCHS, max_batches_per_epoch=1000,
        val_loader=pooled_val)

    rows = []
    for s in sources:
        rows.append({'regime': 'reverse_transfer', 'client': s, 'seed': seed,
                     **evaluate(source_state, clients[s])})

    torch.manual_seed(seed)
    print(f'  [reverse_transfer] fine-tuning on {target} from '
          f'{"+".join(sources)} weights...')
    finetuned_state = local_update(
        source_state, clients[target]['loaders']['train'], finetune_epochs,
        max_batches_per_epoch=1000,
        val_loader=clients[target]['loaders']['val'])

    rows.append({'regime': 'reverse_transfer', 'client': target, 'seed': seed,
                 **evaluate(finetuned_state, clients[target])})

    return pd.DataFrame(rows)


def centralised(clients, seed):
    """
    This function trains a centralized model by combining the training
    data from all clients into a single dataset. The model is trained
    on the pooled data without considering client boundaries and is
    subsequently evaluated separately on each client's test dataset.

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
        Columns: 'regime' ('centralised'), 'client', 'seed', 'R2',
        'RMSE', 'MAE'.
    """

    torch.manual_seed(seed)
    all_members = list(clients)
    loader = build_pooled_loader(clients, all_members, split='train')
    val_loader = build_pooled_loader(clients, all_members, split='val',
                                     shuffle=False)

    st = local_update(MRILSTM().to(DEVICE).state_dict(), loader,
                      N_ROUNDS * LOCAL_EPOCHS, max_batches_per_epoch=1000,
                      val_loader=val_loader)


    return pd.DataFrame([{'regime': 'centralised', 'client': n, 'seed': seed,
                          **evaluate(st, c)} for n, c in clients.items()])


def mmd(A, B, gamma = None, n = 2000, seed = 0):
    """
    This function computes the Maximum Mean Discrepancy (MMD) between
    two datasets using a Gaussian Radial Basis Function (RBF) kernel.

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

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets for each client,
        including training input sequences and target values.

    Returns
    -------
    results : pandas.DataFrame
        Columns: 'pair', 'mmd'.
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


def shapley(clients, seed):
    """
    This function computes the Shapley value of each client in a
    federated learning system. Coalitions of size >= 2 are evaluated
    using plain FedAvg (mu=0.0).

    Parameters
    ----------
    clients : dict
        Dictionary containing the processed datasets, DataLoaders,
        and metadata for each client.

    seed : int
        Random seed used to ensure reproducibility of model training.

    Returns
    -------
    phi : dict
        Shapley value for each client.
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

                final, _, _ = federate(clients, list(coal), mu = 0.0, seed = seed)
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
    # local, transfer (looped over all 3 sources), and fedavg feed the
    # headline constructs (delta_sov, gamma), so they're repeated across
    # seeds to make a reported gap distinguishable from ordinary
    # init-to-init noise.
    #
    # COST NOTE: looping transfer() over 3 sources roughly triples the
    # cost of the transfer stage versus a single-source run (3 from-
    # scratch source trainings + 6 fine-tunes per seed, vs. 1+2 before).
    # Consider running with SEEDS = [0] first to check timing before
    # committing to the full SEEDS list.
    for seed in SEEDS:
        seed_start = time.time()
        print(f'\n===== seed {seed} =====')
        out.append(local_only(clients, seed))

        for source in TRANSFER_SOURCES:
            out.append(transfer(clients, seed, source=source))

        out.append(centralised(clients, seed))

        for wt in ('sample', 'uniform'):
            fa_start = time.time()
            fa, hist, _ = federate(clients, members, mu = 0.0,
                                   weighting = wt, seed = seed, log_rounds = True)
            print(f'  [fedavg_{wt}] took {(time.time()-fa_start)/60:.1f} min')
            out.append(pd.DataFrame([{'regime': 'fedavg', 'client': k,
                                      'weighting': wt, 'seed': seed, **m}
                                     for k, m in fa.items()]))
            hist.to_csv(f'{OUT_DIR}/rounds_fedavg_{wt}_s{seed}.csv', index = False)

        print(f'  === seed {seed} total: {(time.time()-seed_start)/60:.1f} min, '
              f'script elapsed: {(time.time()-script_start)/3600:.2f} hr ===')

    res = pd.concat(out, ignore_index = True)
    res.to_csv(f'{OUT_DIR}/federated_results.csv', index = False)
    print(f'\n[SAVED] federated_results.csv with regimes: '
          f'{sorted(res["regime"].unique())}')

    # Quick console diagnostic: compare each source's 'finetuned' R2
    # against that same client's 'local' R2, to flag whether fine-
    # tuning is converging back to the local optimum regardless of
    # source (see TRANSFER_FINETUNE_EPOCHS notes above).
    transfer_rows = res[(res['regime'] == 'transfer') & (res['mode'] == 'finetuned')]
    local_rows = res[res['regime'] == 'local'][['client', 'seed', 'R2']] \
        .rename(columns={'R2': 'R2_local'})
    diag = transfer_rows.merge(local_rows, on=['client', 'seed'])
    diag['diff_vs_local'] = diag['R2'] - diag['R2_local']
    print('\n[DIAGNOSTIC] finetuned transfer R2 vs. that client\'s own local R2:')
    print(diag.groupby(['source', 'client'])['diff_vs_local'].mean().round(4).to_string())
    print('(values near 0 suggest fine-tuning is re-converging to the local '
          'optimum regardless of source — check the zeroshot rows for '
          'evidence of genuine transfer)')

    phi = shapley(clients, seed = SEEDS[0])
    pd.DataFrame([{'client': k, 'shapley': v} for k, v in phi.items()]) \
      .to_csv(f'{OUT_DIR}/shapley.csv', index = False)
    print('\nShapley:', phi)

    # Constructs use fedavg (sample-weighted) as the headline federated
    # regime 'g'.
    fedavg_sample = res[(res['regime'] == 'fedavg') & (res['weighting'] == 'sample')]
    piv = pd.concat([
        res[res['regime'].isin(['local', 'centralised'])],
        fedavg_sample.drop(columns='weighting')
    ], ignore_index=True).pivot_table(
        index = ['client', 'seed'], columns = 'regime', values = 'R2')
    piv['delta_sov_fedavg'] = piv['centralised'] - piv['fedavg']
    piv['gamma_fedavg']     = piv['fedavg'] - piv['local']
    piv.reset_index().to_csv(f'{OUT_DIR}/constructs.csv', index = False)

    print(f'\nDone -> federated_results.csv, constructs.csv, shapley.csv '
          f'(total {(time.time()-script_start)/3600:.2f} hr)')
