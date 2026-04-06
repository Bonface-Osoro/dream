import configparser
import os 
import warnings
import numpy as np
import pandas as pd
from dream.pre_geo import combine_predictors


pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')       

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')

mri_path = os.path.join(DATA_RESULTS, 'mri', 'malaria_risk_index.csv')
ndvi_path = os.path.join(DATA_PROCESSED, 'combined_data', 'ndvi_combined.csv')


if __name__ == '__main__':

    combine_predictors(mri_path, ndvi_path, DATA_RESULTS)