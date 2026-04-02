"""
Tools for modeling malaria through Geospatial 
Artificial Intelligence.

Developed by Bonface Osoro.

March 2026

"""
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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