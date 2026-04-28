"""
Tools for modeling malaria through Geospatial 
Artificial Intelligence.

Developed by Bonface Osoro.

March 2026

"""
import pytensor
import warnings
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
pytensor.config.floatX = "float32"

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')  


def pca_weights(df, columns):
    """
    Computes PCA-based weights for malaria risk index (MRI).

    Parameters:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to include in PCA

    Returns:
        dict: {column_name: weight}
    """
    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    
    # Applying PCA (first principal component)
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    
    # Extracting PCA weights for each malaria index. 
    # The sign of the weights is inverted to ensure that higher values 
    # of the indices correspond to higher MRI values.
    weights = -pca.components_[0]
    
    # Mapping the column names to weights
    weight_dict = {col: round(float(w), 4) for col, w in zip(columns, weights)}
    
    return weight_dict


def compute_mri(df, weight_dict):
    """
    Compute Malaria Risk Index (MRI) using a weight dictionary.

    Parameters:
        df (pd.DataFrame): Input dataframe
        weight_dict (dict): {column_name: weight}
        normalize (bool): Whether to normalize MRI to 0–1

    Returns:
        pd.Series: MRI values (or normalized MRI if normalize=True)
    """

    missing_cols = [col for col in weight_dict if col not in df.columns]
    if missing_cols:

        raise ValueError(f'Missing columns in dataframe: {missing_cols}')
    
    # Compute weighted sum (vectorized)
    mri = sum(df[col] * weight for col, weight in weight_dict.items())

    # Normalize MRI to 0–1
    normalize = True  # Set to True to normalize MRI to 0–1
    if normalize:

        mri = (mri - mri.min()) / (mri.max() - mri.min())
    
    
    return mri


def estimate_monthly_mri(input_csv, output_csv_path):
    """
    Estimate monthly MRI values using a Bayesian state-space model in PyMC.

    Parameters
    ----------

        input_csv : str 
            Path to input CSV with columns: year, longitude, latitude, month, 
            precipitation_mm, temperature_C, ndvi, elevation_m, mri_value
        output_csv_path : str
            Path to save output CSV with monthly MRI estimates
    """

    df = pd.read_csv(input_csv)
    df = df.dropna().reset_index(drop=True)
    month_map = {
        'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
        'jul':7, 'aug':8, 'sept':9, 'oct':10, 'nov':11, 'dec':12
    }

    df['month_num'] = df['month'].str.lower().map(month_map)

    groups = list(df.groupby(['year', 'longitude', 'latitude']))
    G = len(groups)

    # --------------
    # BUILD ARRAYS #
    # --------------
    rain = np.zeros((G, 12))
    temp = np.zeros((G, 12))
    ndvi = np.zeros((G, 12))
    elev = np.zeros((G, 12))
    mri = np.zeros(G)

    def standardize(x):

        return (x - x.mean()) / (x.std() + 1e-6)

    def logit(x):

        x = np.clip(x, 1e-6, 1 - 1e-6)

        return np.log(x / (1 - x))

    full_months = np.arange(1, 13)

    for i, ((year, lon, lat), group) in tqdm(enumerate(groups),
        total = len(groups), desc = 'Preparing data'):

        group = group.set_index('month_num').reindex(full_months)
        numeric_cols = group.select_dtypes(include = [np.number]).columns

        # Fill missing months safely
        group.loc[:, numeric_cols] = (group[numeric_cols]
            .interpolate(limit_direction = 'both'))

        rain[i] = standardize(group['precipitation_mm'].values)
        temp[i] = standardize(group['temperature_C'].values)
        ndvi[i] = standardize(group['ndvi'].values)
        elev[i] = standardize(group['elevation_m'].values)
        mri[i] = group['mri_value'].iloc[0]

    # Convert MRI to latent space
    mri_logit = logit(mri)

    # -------------------- #
    # HIERARCHICAL MODEL #
    # -------------------- #
    with pm.Model() as model:

        # Shared coefficients
        alpha = pm.Normal('alpha', 0.7, 0.2)
        beta_rain = pm.Normal('beta_rain', 0, 1)
        beta_temp = pm.Normal('beta_temp', 0, 1)
        beta_ndvi = pm.Normal('beta_ndvi', 0, 1)
        beta_elev = pm.Normal('beta_elev', 0, 1)

        sigma = pm.HalfNormal('sigma', 1.0)

        # -----------------------------
        # NON-CENTERED PARAMETERIZATION
        # -----------------------------
        z0_raw = pm.Normal('z0_raw', 0, 1, shape=G)

        # Initialize in latent (logit) space
        eta0 = pm.Deterministic('eta0', mri_logit + 0.5 * z0_raw)
        eps = pm.Normal('eps', 0, 1, shape=(11,))

        # -----------------
        # LATENT DYNAMICS #
        # -----------------
        eta = [eta0]

        for t in range(1, 12):

            mu_t = (alpha * eta[t-1]
                + beta_rain * rain[:, t]
                + beta_temp * temp[:, t]
                + beta_ndvi * ndvi[:, t]
                + beta_elev * elev[:, t])

            eta_t = pm.Deterministic(f'eta_{t}',
                mu_t + sigma * eps[t-1])
            eta.append(eta_t)
        eta_stack = pm.math.stack(eta, axis=1)

        # -----------------------------
        # APPLY SIGMOID → constrain to (0,1)
        # -----------------------------
        z_stack = pm.Deterministic('z_stack',
            pm.math.sigmoid(eta_stack))

        # ------------------- #
        # OBSERVATION MODEL #
        # ------------------- #
        z_mean = z_stack.mean(axis = 1)

        pm.Normal('annual_obs', mu = z_mean,
            sigma = 0.05,  observed = mri)

        # ----------- #
        # SAMPLING #
        # ----------- #
        trace = pm.sample(draws = 200, tune = 200,
            chains = 2, cores = 1, nuts_sampler = 'nutpie',
            target_accept = 0.95, progressbar = True)

    # ----------------- #
    # EXTRACT RESULTS #
    # ----------------- #
    z_post = trace.posterior['z_stack'].mean(dim = ('chain', 'draw')).values

    # -------------------------#
    # REBUILD OUTPUT DATAFRAME #
    # -------------------------#
    results = []

    for i, ((year, lon, lat), group) in enumerate(groups):

        g = group.copy()
        g = g.set_index('month_num').reindex(full_months).reset_index()

        g['year'] = year
        g['longitude'] = lon
        g['latitude'] = lat

        numeric_cols = g.select_dtypes(include = ['number']).columns
        g[numeric_cols] = (g[numeric_cols].astype('float64')
            .interpolate().bfill().ffill())
        g['monthly_mri'] = z_post[i]
        results.append(g)

    df_out = pd.concat(results)

    df_out.to_csv(output_csv_path, index=False)

    return None


def estimate_month_mri(input_csv, output_csv_path):
    df = pd.read_csv(input_csv)
    df = df.dropna().reset_index(drop=True)

    month_map = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
        'jul':7,'aug':8,'sept':9,'oct':10,'nov':11,'dec':12
    }
    df['month_num'] = df['month'].str.lower().map(month_map)
    groups = list(df.groupby(['year','longitude','latitude']))
    G = len(groups)

    rain = np.zeros((G, 12))
    temp = np.zeros((G, 12))
    ndvi = np.zeros((G, 12))
    elev = np.zeros((G, 12))
    mri  = np.zeros(G)

    def standardize(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    def logit(x):
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))

    full_months = np.arange(1, 13)

    for i, ((year, lon, lat), group) in tqdm(
            enumerate(groups), total=len(groups), desc='Preparing data'):
        group = group.set_index('month_num').reindex(full_months)
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        group.loc[:, numeric_cols] = group[numeric_cols].interpolate(limit_direction='both')
        rain[i] = standardize(group['precipitation_mm'].values)
        temp[i] = standardize(group['temperature_C'].values)
        ndvi[i] = standardize(group['ndvi'].values)
        elev[i] = standardize(group['elevation_m'].values)
        mri[i]  = group['mri_value'].iloc[0]

    mri_logit = logit(mri).astype('float32')
    mri       = mri.astype('float32')

    # Stack covariates: shape (4, G, 12) → pre-cast outside model
    X = np.stack([rain, temp, ndvi, elev], axis=0).astype('float32')

    with pm.Model() as model:

        alpha  = pm.Normal('alpha', 0.7, 0.2)
        beta   = pm.Normal('beta', 0, 1, shape=4)
        sigma  = pm.HalfNormal('sigma', 1.0)

        z0_raw = pm.Normal('z0_raw', 0, 1, shape=G)
        eta0   = mri_logit + 0.5 * z0_raw              # (G,)

        # Covariate term: (G, 12)
        X_pt     = pt.as_tensor_variable(X)            # (4, G, 12)
        cov_term = pt.tensordot(beta, X_pt, axes=[[0],[0]])  # (G, 12)

        # Innovation noise shared across groups
        eps = pm.Normal('eps', 0, 1, shape=11)         # (11,)

        # ── scan with alpha and sigma passed as non_sequences ────────────────
        # This is the fix: RVs must be explicit inputs, not closures
        def transition(cov_t, eps_t, eta_prev, alpha_, sigma_):
            return alpha_ * eta_prev + cov_t + sigma_ * eps_t

        # scan sequences must be (steps, ...) — transpose cov to (11, G)
        cov_seq = cov_term[:, 1:].T   # (11, G)  — time-major

        eta_rest, _ = pytensor.scan(
            fn=transition,
            sequences=[cov_seq, eps],          # stepped over axis-0 (time)
            outputs_info=eta0,                 # initial carry: (G,)
            non_sequences=[alpha, sigma],      # ← pass RVs explicitly here
        )
        # eta_rest: (11, G), prepend eta0 → (12, G) → (G, 12)
        eta_all = pt.concatenate([eta0[None, :], eta_rest], axis=0).T

        z_all  = pm.Deterministic('z_all', pm.math.sigmoid(eta_all))  # (G,12)
        z_mean = z_all.mean(axis=1)                                    # (G,)

        pm.Normal('annual_obs', mu=z_mean, sigma=0.05, observed=mri)

        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=2,
            nuts_sampler='numpyro',
            target_accept=0.85,
            progressbar=True,
        )

    # ── Extract & rebuild output ─────────────────────────────────────────────
    z_post = trace.posterior['z_all'].mean(dim=('chain', 'draw')).values
    # z_post shape: (G, 12)

    results = []
    for i, ((year, lon, lat), group) in enumerate(groups):
        g = group.copy().set_index('month_num').reindex(full_months).reset_index()
        g['year'], g['longitude'], g['latitude'] = year, lon, lat
        numeric_cols = g.select_dtypes(include=['number']).columns
        g[numeric_cols] = g[numeric_cols].astype('float64').interpolate().bfill().ffill()
        g['monthly_mri'] = z_post[i]
        results.append(g)

    pd.concat(results).to_csv(output_csv_path, index=False)
    return None