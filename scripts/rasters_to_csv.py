"""
GeoTIFF Multi-Variable Raster → CSV Converter
-----------------------------------------------
Processes five folders of raster data:
  parasite_rate, incidence_rate, mortality_rate, net_access, net_use

For each folder, all yearly GeoTIFF files are combined into a single CSV:
  year, longitude, latitude, <variable_name>

No per-year CSVs are saved — just one combined file per variable.

Expected filename format:
  202508_Global_Pf_Parasite_Rate_ZWE_2021.tiff
  └─ year = last 4-digit token (1900–2099) in the filename → 2021

CONFIG: set BASE_DIR and OUTPUT_DIR below, then run.
"""

import configparser
import os
import re
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

warnings.filterwarnings("ignore")
CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

BASE_DIR = os.path.join(BASE_PATH, '..', 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_PATH, '..', 'results', 'processed', 'ZWE_raster_csvs')

VARIABLES = {
    'ZWE_parasite_rates':  {'column': 'parasite_rate',  'output': 'parasite_rate_all_years.csv'},
    'ZWE_incidence_rates': {'column': 'incidence_rate', 'output': 'incidence_rate_all_years.csv'},
    'ZWE_mortality_rates': {'column': 'mortality_rate', 'output': 'mortality_rate_all_years.csv'},
    'ZWE_net_access':      {'column': 'net_access',     'output': 'net_access_all_years.csv'},
    'ZWE_net_use':         {'column': 'net_use',        'output': 'net_use_all_years.csv'},
}

NO_DATA = None  

def extract_year(filename):

    """
    Parse year from filename by splitting on non-digit characters and
    finding the last exactly-4-digit token in range 1900–2099.
    '202508_Global_Pf_Parasite_Rate_ZWE_2021.tiff' → 2021

    Parameters
    ----------
        filename : str
            The name of the file to extract the year from.
    Returns
    -------
        int
            The extracted year as an integer.
    """
    stem   = os.path.splitext(filename)[0]
    tokens = re.split(r'[^0-9]+', stem)
    years  = [t for t in tokens if len(t) == 4 and t.isdigit()
              and 1900 <= int(t) <= 2099]
    if not years:

        raise ValueError(
            f"No 4-digit year found in: '{filename}'. "
            f"Tokens found: {tokens}"
        )
    
    return int(years[-1])


def raster_to_df(tif_path, value_col, nodata = None):

    """
    Read first band of a GeoTIFF and return a DataFrame:
      year (added by caller), longitude, latitude, <value_col>
    Nodata and negative pixels are dropped.

    Parameters
    ----------
    tif_path : str
        Path to the GeoTIFF file to read.
    value_col : str
        The name of the column to store pixel values in 
        the resulting DataFrame.
    nodata : numeric, optional
        The nodata value to filter out. If None, uses the 
        nodata value from the GeoTIFF metadata.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing longitude, latitude, and 
        pixel values for valid pixels.
    """
    with rasterio.open(tif_path) as src:

        band      = src.read(1).astype(float)
        nd_val    = nodata if nodata is not None else src.nodata
        transform = src.transform
        height, width = band.shape

        rows, cols = np.meshgrid(
            np.arange(height), np.arange(width), indexing='ij'
        )
        lons, lats = xy(transform, rows.ravel(), cols.ravel(), 
                        offset='center')
        values     = band.ravel()

        df = pd.DataFrame({
            'longitude': np.array(lons),
            'latitude':  np.array(lats),
            value_col:   values,
        })

        if nd_val is not None:
            df = df[df[value_col] != nd_val]

        df = df[np.isfinite(df[value_col])]
        df = df[df[value_col] >= 0]

    return df.reset_index(drop=True)


def process_variable(folder_path, value_col,
                     output_path, nodata,):
    
    """
    Process all TIFFs in one folder and save a single 
    combined CSV.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing the GeoTIFF files for one variable.
    value_col : str
        The name of the column to store pixel values in the resulting CSV.
    output_path : str
        Path to save the combined CSV file.
    nodata : numeric
        The nodata value to filter out from the GeoTIFF files.
    """

    tif_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.tif', '.tiff'))])

    if not tif_files:
        print(f"No .tif/.tiff files found in: {folder_path}")
        return

    print(f"{len(tif_files)} file(s) found")
    all_dfs = []

    for fname in tif_files:

        fpath = os.path.join(folder_path, fname)
        try:

            year = extract_year(fname)
        except ValueError as e:

            print(f"Skipping '{fname}' — {e}")
            continue

        df = raster_to_df(fpath, value_col, nodata = nodata)
        df.insert(0, 'year', year)
        all_dfs.append(df)

    if not all_dfs:

        print(f"  ✗  No data extracted for {value_col}")
        return

    combined = pd.concat(all_dfs, ignore_index = True).sort_values(
        ['year', 'latitude', 'longitude']).reset_index(drop=True)

    combined.to_csv(output_path, index=False)
    print(f"     Years: {sorted(combined['year'].unique().tolist())}")


def run():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for folder_name, cfg in VARIABLES.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        output_path = os.path.join(OUTPUT_DIR, cfg['output'])

        print(f"Output   : {output_path}")

        if not os.path.isdir(folder_path):
            
            print(f"Folder not found — skipping. "
                  f"Check that '{folder_name}' exists inside BASE_DIR.")
            continue

        process_variable(folder_path, cfg['column'], output_path, nodata=NO_DATA)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == '__main__':
    run()
