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

DATA_RAW = os.path.join(BASE_PATH, '..', 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')
NO_DATA    = None   

def extract_year_from_filename(filename):

    """
    This is a helper function for parsing 
    year and month from a filename. Splits 
    on non-digit characters, then picks the 
    last exactly-4-digit token in the range 1900-2099.

    Parameters
    ----------

    filename : str
        name of the csv file

    Return
    ------

    year : int
        year
    """
    stem = os.path.splitext(filename)[0]
    tokens = re.split(r'[^0-9]+', stem)
    years = [t for t in tokens if len(t) == 4 
             and t.isdigit() and 1900 <= int(t) <= 2099]
    
    if not years:

        raise ValueError(
            f"Could not parse a year from filename: '{filename}'\n"
            f"Expected a 4-digit year (1900-2099) as a separate token in the filename."
        )


    return int(years[-1])


def raster_to_dataframe(tif_path):

    """
    This function reads a single-band GeoTIFF 
    and return a DataFrame with columns:
    longitude, latitude, parasite_rate, year
    Pixels with nodata values are dropped.

    Parameters
    ----------
    tif_path : str
        Name of the GeoTIFF raster imagery.

    Returns
    -------
    df : dataframe
        Dataframe
    """

    nodata = None
    with rasterio.open(tif_path) as src:

        band    = src.read(1).astype(float)
        nd_val  = nodata if nodata is not None else src.nodata
        transform = src.transform
        height, width = band.shape

        rows, cols = np.meshgrid(
            np.arange(height), np.arange(width), 
            indexing ='ij')

        lons, lats = xy(transform, rows.ravel(), 
                        cols.ravel(), offset='center')

        values = band.ravel()

        df = pd.DataFrame({
            'longitude':     np.array(lons),
            'latitude':      np.array(lats),
            'parasite_rate': values,
        })

        if nd_val is not None:

            df = df[df['parasite_rate'] != nd_val]

        df = df[np.isfinite(df['parasite_rate'])]
        df = df[df['parasite_rate'] >= 0]
        df = df.reset_index(drop = True)


    return df


def process_folder(raster_dir, output_dir):

    """
    This function processes all single-band GeoTIFF files in a folder,
    converts each raster to a tabular DataFrame, extracts the year from
    the filename, saves an individual CSV for each raster, and creates
    a combined CSV containing data from all years. Pixels with nodata
    values are dropped.

    Parameters
    ----------
    raster_dir : str
        Path to the directory containing the GeoTIFF raster files.

    output_dir : str
        Path to the directory where the output CSV files will be saved.
    """
    
    nodata = None
    os.makedirs(output_dir)

    tif_files = sorted([f for f in os.listdir(raster_dir)
        if f.lower().endswith(('.tif', '.tiff'))])

    if not tif_files:

        print(f'No .tif/.tiff files found in: {raster_dir}')

        return

    print(f'Found {len(tif_files)} raster file(s)\n')

    all_dfs = []

    for fname in tif_files:

        fpath = os.path.join(raster_dir, fname)
        print(f'Processing: {fname}')

        try:

            year = extract_year_from_filename(fname)
        except ValueError as e:

            print(f'Skipping — {e}')

            continue

        print(f'Year parsed: {year}')

        df = raster_to_dataframe(fpath, nodata = nodata)
        df.insert(0, 'year', year)

        out_name = os.path.splitext(fname)[0] + '_tabular.csv'
        out_path = os.path.join(output_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f'Saved {out_path}\n')

        all_dfs.append(df)

    if all_dfs:

        combined = pd.concat(all_dfs, ignore_index = True)
        combined_path = os.path.join(output_dir, 
                                     'all_years_parasite_rate.csv')
        combined.to_csv(combined_path, index = False)

        print(f'Saved {combined_path}')

raster_path = os.path.join(DATA_RAW, 'ZWE_parasite_rates')
output_path = os.path.join(DATA_RESULTS, 'ZWE_parasite_rates_csv')

if __name__ == '__main__':

    process_folder(raster_path, output_path, nodata = NO_DATA)
