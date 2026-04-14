"""
Tools for modeling malaria through Geospatial 
Artificial Intelligence.

Developed by Bonface Osoro.

March 2026

"""
import warnings
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

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

    df = pd.read_csv(input_csv)

    month_map = {
        'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
        'jul':7, 'aug':8, 'sept':9, 'oct':10, 'nov':11, 'dec':12
    }

    df['month_num'] = df['month'].str.lower().map(month_map)

    groups = list(df.groupby(['year', 'longitude', 'latitude']))
    G = len(groups)

    # -----------------------------
    # BUILD ARRAYS
    # -----------------------------
    rain = np.zeros((G, 12))
    temp = np.zeros((G, 12))
    ndvi = np.zeros((G, 12))
    elev = np.zeros((G, 12))
    mri = np.zeros(G)

    def standardize(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    full_months = np.arange(1, 13)

    for i, ((year, lon, lat), group) in tqdm(
        enumerate(groups),
        total=len(groups),
        desc="Preparing data"
    ):
        group = group.set_index('month_num').reindex(full_months)

        # fill missing months safely
        group = group.interpolate().bfill().ffill()

        rain[i] = standardize(group['precipitation_mm'].values)
        temp[i] = standardize(group['temperature_C'].values)
        ndvi[i] = standardize(group['ndvi'].values)
        elev[i] = standardize(group['elevation_m'].values)

        mri[i] = group['mri_value'].iloc[0]

    # -----------------------------
    # HIERARCHICAL MODEL (OPTIMIZED)
    # -----------------------------
    with pm.Model() as model:

        # Shared coefficients
        alpha = pm.Normal('alpha', 0.7, 0.2)
        beta_rain = pm.Normal('beta_rain', 0, 1)
        beta_temp = pm.Normal('beta_temp', 0, 1)
        beta_ndvi = pm.Normal('beta_ndvi', 0, 1)
        beta_elev = pm.Normal('beta_elev', 0, 1)

        sigma = pm.HalfNormal('sigma', 1.0)

        # -----------------------------
        # NON-CENTERED PARAMETERIZATION (KEY SPEEDUP)
        # -----------------------------
        z0_raw = pm.Normal('z0_raw', 0, 1, shape=G)
        z0 = pm.Deterministic('z0', mri + 0.5 * z0_raw)

        eps = pm.Normal('eps', 0, 1, shape=(G, 11))

        # -----------------------------
        # LATENT DYNAMICS (VECTORIZED)
        # -----------------------------
        z = [z0]

        for t in range(1, 12):

            mu_t = (
                alpha * z[t-1]
                + beta_rain * rain[:, t]
                + beta_temp * temp[:, t]
                + beta_ndvi * ndvi[:, t]
                + beta_elev * elev[:, t]
            )

            z_t = pm.Deterministic(
                f'z_{t}',
                mu_t + sigma * eps[:, t-1]
            )

            z.append(z_t)

        z_stack = pm.math.stack(z, axis=1)

        # -----------------------------
        # FAST OBSERVATION MODEL
        # -----------------------------
        z_mean = z_stack.mean(axis=1)

        pm.Normal(
            'annual_obs',
            mu=z_mean,
            sigma=0.5,
            observed=mri
        )

        # -----------------------------
        # SAMPLING (OPTIMIZED FOR SCALE)
        # -----------------------------
        trace = pm.sample(
            draws=400,
            tune=600,
            chains=2,
            cores=1,  
            target_accept=0.95,
            progressbar=True
        )

    # -----------------------------
    # EXTRACT RESULTS
    # -----------------------------
    z_post = trace.posterior

    z_est = np.zeros((G, 12))

    z_est[:, 0] = z_post['z0'].mean(dim=('chain', 'draw')).values

    for t in range(1, 12):
        z_est[:, t] = z_post[f'z_{t}'].mean(dim=('chain', 'draw')).values

    # -----------------------------
    # REBUILD OUTPUT DATAFRAME
    # -----------------------------
    results = []

    for i, ((year, lon, lat), group) in enumerate(groups):

        g = group.copy()
        g = g.set_index('month_num').reindex(full_months).reset_index()
        g['year'] = year
        g['longitude'] = lon
        g['latitude'] = lat

        g = g.interpolate().bfill().ffill()
        g['monthly_mri'] = z_est[i]

        results.append(g)

    df_out = pd.concat(results)
    df_out.to_csv(output_csv_path, index=False)

    return None


def estimate_monthly_mri_pymc(input_csv, output_csv_path):
    """
    Estimate monthly MRI values using a Bayesian state-space model in PyMC.

    Parameters
    ----------

        input_csv : str 
            Path to input CSV with columns: year, longitude, latitude, month, 
            precipitation_mm, temperature_C, ndvi, elevation_m, mri_value
        output_csv_path : str
            Path to save output CSV with monthly MRI estimates"""

    df = pd.read_csv(input_csv)
    results = []

    # Month mapping
    month_map = {
        'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
        'jul':7, 'aug':8, 'sept':9, 'oct':10, 'nov':11, 'dec':12
    }

    df['month_num'] = df['month'].str.lower().map(month_map)

    grouped = df.groupby(['year', 'longitude', 'latitude'])

    for (year, lon, lat), group in tqdm(grouped, total=len(grouped), desc="Processing groups"):

        group = group.sort_values('month_num')

        rain = group['precipitation_mm'].values
        temp = group['temperature_C'].values
        ndvi = group['ndvi'].values
        elevation = group['elevation_m'].values

        annual_mri = group['mri_value'].iloc[0]

        def standardize(x):

            return (x - x.mean()) / (x.std() + 1e-6)

        rain_s = standardize(rain)
        temp_s = standardize(temp)
        ndvi_s = standardize(ndvi)
        elev_s = standardize(elevation)

        with pm.Model() as model:

            alpha = pm.Normal('alpha', mu = 0.7, sigma = 0.2)
            beta_rain = pm.Normal('beta_rain', mu = 0, sigma = 1)
            beta_temp = pm.Normal('beta_temp', mu = 0, sigma = 1)
            beta_ndvi = pm.Normal('beta_ndvi', mu = 0, sigma = 1)
            beta_elev = pm.Normal('beta_elev', mu = 0, sigma = 1)

            sigma = pm.HalfNormal('sigma', sigma = 1)
            z0 = pm.Normal('z0', mu = annual_mri, sigma = 0.2)

            # Latent monthly MRI
            z = [z0]

            for t in range(1, 12):

                z_t = pm.Normal(
                    f'z_{t}',
                    mu = (
                        alpha * z[t-1]
                        + beta_rain * rain_s[t]
                        + beta_temp * temp_s[t]
                        + beta_ndvi * ndvi_s[t]
                        + beta_elev * elev_s[t]
                    ),
                    sigma = sigma
                )
                z.append(z_t)

            z_stack = pm.math.stack(z)

            # --- Observation constraint  ---
            pm.Normal(
                'annual_obs',
                mu =pm.math.mean(z_stack),
                sigma = 0.5,  
                observed = annual_mri
            )

            # Sample
            trace = pm.sample(draws = 300, tune = 500, chains = 1, cores = 4, 
                              progressbar = False, target_accept = 0.98)

        # Extract posterior mean of latent states
        z_est = np.array([trace.posterior[f'z_{t}'].mean().values for t in range(1,12)])
        z_est = np.insert(z_est, 0, trace.posterior['z0'].mean().values)

        group = group.copy()
        group['monthly_mri'] = z_est

        results.append(group)

    results_df = pd.concat(results)

    results_df.to_csv(output_csv_path, index = False)

    return None


def create_sequences(df, features, target, look_back):
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

    Returns:
    -------    
        X : np.array
            3D array of shape (num_samples, look_back, num_features)
        y : np.array
            1D array of target values corresponding to each sequence
    """
    
    years = look_back
    X_list, y_list, loc_list = [], [], []

    for (lon, lat), group in df.groupby(['longitude', 'latitude']):

        group = group.sort_values('year')
        data_features = group[features].values
        data_target = group[target].values
        for i in range(len(group) - years):

            X_list.append(data_features[i:i + years])
            y_list.append(data_target[i + years])
            loc_list.append((lon, lat))  
            
    return np.array(X_list), np.array(y_list), loc_list