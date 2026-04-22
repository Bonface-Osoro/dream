import configparser
import os 
import warnings
import pandas as pd
from dream.pre_geo import (normalize_column)
from dream.malmo import *


pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')       

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')


'''output_years_folder = os.path.join(DATA_RESULTS, 'spatial_temporal_data')
df = pd.read_csv(os.path.join(output_years_folder, 'spatio_temporal_malaria_indices.csv'))
df['normalized_parasite_rate'] = normalize_column(df, 'parasite_rate')
df['normalized_incidence_rate'] = normalize_column(df, 'incidence_rate')
df['normalized_net_use'] = normalize_column(df, 'net_use')
df['normalized_net_access'] = normalize_column(df, 'net_access')
df['normalized_mortality_rate'] = normalize_column(df, 'mortality_rate')
df = df.drop(columns = ['parasite_rate', 'incidence_rate', 'net_use', 
                        'net_access', 'mortality_rate'])
df.to_csv(os.path.join(output_years_folder, 
                       'spatio_temporal_malaria_indices_normalized.csv'), index = False)

cols = ['normalized_parasite_rate', 'normalized_incidence_rate',
        'normalized_net_use', 'normalized_net_access',
        'normalized_mortality_rate']

weights = pca_weights(df, cols)

df['mri_value'] = compute_mri(df, weights)
output_path = os.path.join(DATA_RESULTS, 'mri', 'malaria_risk_index.csv')
os.makedirs(os.path.dirname(output_path), exist_ok = True)
df.to_csv(output_path, index = False)'''

input_mri_file = os.path.join(DATA_RESULTS, 'mri', 'ZWE_ecological_predictors_with_mri.csv')
monthly_mri_output = os.path.join(DATA_RESULTS, 'mri', 'ZWE_malaria_risk_index_monthly.csv')

if __name__ == "__main__":
    
        estimate_monthly_mri(input_mri_file, monthly_mri_output) 