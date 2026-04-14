import configparser
import glob
import os 
import warnings
import numpy as np
import pandas as pd
from dream.pre_geo import eco_mri, combine_ndvi_monthly, combine_rain_monthly, combine_monthly_temp, merge_eco_datasets


pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')       

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')

mri_path = os.path.join(DATA_RESULTS, 'mri', 'malaria_risk_index.csv')
ndvi_path = os.path.join(DATA_PROCESSED, 'combined_data', 'ndvi_combined.csv')

monthly_ndvi = os.path.join(DATA_RAW, 'monthly_ndvi')
monthly_rain = os.path.join(DATA_RAW, 'monthly_rain')
monthly_temp = os.path.join(DATA_RAW, 'monthly_temperature')
complete_datasets = os.path.join(DATA_PROCESSED, 'combined_data')

comb_ndvi = os.path.join(DATA_PROCESSED, 'combined_data', 'combined_monthly_ndvi.csv')
comb_rain = os.path.join(DATA_PROCESSED, 'combined_data', 'combined_monthly_rain.csv')
comb_temp = os.path.join(DATA_PROCESSED, 'combined_data', 'combined_monthly_temperature.csv')
comb_elev = os.path.join(DATA_PROCESSED, 'combined_data', 'elevation.csv')
combined_output = os.path.join(DATA_PROCESSED, 'combined_data', 'ecological_predictors.csv')

eco_mri_output = os.path.join(DATA_RESULTS, 'mri', 'ecological_predictors_with_mri.csv')


if __name__ == '__main__':

    #combine_ndvi_monthly(monthly_ndvi, complete_datasets)

    #combine_rain_monthly(monthly_rain, complete_datasets)

    #combine_monthly_temp(monthly_temp, complete_datasets)

    #merge_eco_datasets(comb_ndvi, comb_elev, comb_rain, comb_temp, combined_output)


    eco_mri(mri_path, combined_output, eco_mri_output)