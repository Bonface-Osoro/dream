import configparser
import glob
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

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')

mri_path = os.path.join(DATA_RESULTS, 'mri', 'malaria_risk_index.csv')
ndvi_path = os.path.join(DATA_PROCESSED, 'combined_data', 'ndvi_combined.csv')

monthly_ndvi = os.path.join(DATA_RAW, 'monthly_ndvi')
files = glob.glob(os.path.join(monthly_ndvi, '*.xlsx'))
all_dfs = []

for file in files:
    
    df = pd.read_excel(file)
    filename = os.path.basename(file)
    
    # extract month (jan, feb, etc.)
    month = filename.split("_")[-1].split(".")[0].lower()
    df = df.drop(columns = ['OBJECTID', 'rainfall', 'normalized_rain'], errors = 'ignore')

    df = df.rename(columns={'RASTERVALU': 'ndvi'})
    df['month'] = month
    df = df.drop_duplicates(subset=['year', 'longitude', 'latitude'])
    
    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)
final_df = final_df.sort_values(by=['year', 'longitude', 'latitude', 'month'])

output_path = os.path.join(DATA_PROCESSED, 'combined_data', 'combined_monthly_ndvi.csv')
final_df.to_csv(output_path, index=False)

print(f"Done! File saved to: {output_path}")

#if __name__ == '__main__':

    #combine_predictors(mri_path, ndvi_path, DATA_RESULTS)