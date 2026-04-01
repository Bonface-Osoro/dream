"""
Preprocessing tools for raw satellite data.

Developed by Bonface Osoro.

March 2026

"""
import os
import re
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


def shp_to_csv(data_type, in_folder, output_folder):
    """
    This function reads different shapefiles and convert them 
    into csv files for further processing.

    Parameters
    ----------
    data_type : str
        The type of data to be processed (e.g., 'incidence_rate', 'parasite_rate', 'net_use', 'net_access', 'mortality_rate').
    input_folder : str
        Path to the folder containing the shapefiles.
    output_folder : str
        Path to the folder where the CSV files will be saved.

    """
    input_folder = os.path.join(in_folder, data_type)
    for file in os.listdir(input_folder):

        if file.endswith('.shp'):

            shp_path = os.path.join(input_folder, file)
            
            # Read shapefile
            gdf = gpd.read_file(shp_path)
            base_name = os.path.splitext(file)[0]
            metric_name = re.sub(r'_\d{4}$', '', base_name)
            gdf['metric'] = metric_name 
            gdf = gdf.rename(columns = {'grid_code': 'value'})
            gdf = gdf.drop(columns = [ 'pointid'])
            
            # Check if geometry is Point
            if gdf.geometry.geom_type.unique()[0] == "Point":

                gdf["longitude"] = gdf.geometry.x
                gdf["latitude"] = gdf.geometry.y
                gdf = gdf.drop(columns="geometry")

            else:

                gdf["geometry"] = gdf["geometry"].apply(lambda geom: 
                                    geom.wkt if geom else None)
            
            csv_name = os.path.splitext(file)[0] + '.csv'
            out_folder = os.path.join(output_folder, metric_name)
            os.makedirs(out_folder, exist_ok = True)
            path_out = os.path.join(out_folder, csv_name)
            gdf.to_csv(path_out, index = False)
            
            print(f'{file} converted to {csv_name}')
 
    return None


def combine_csvs(input_folder, output_file):
    """
    This function combines multiple CSV files into a single CSV file.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the CSV files to be combined.
    output_file : str
        Path to the output CSV file where the combined data will be saved.

    """
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    combined_df = pd.DataFrame()

    for csv_file in csv_files:
        csv_path = os.path.join(input_folder, csv_file)
        df = pd.read_csv(csv_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_file, index=False)
    print(f'Combined {len(csv_files)} CSV files into {output_file}')


def select_valid_years(data_type, input_folder, output_folder):
    """This function selects only the coordinates with complete 
    time series across all years for each metric.
    
    Parameters
    ----------
    data_type : str
        The type of data to be processed (e.g., 'incidence_rate', 
        'parasite_rate', 'net_use', 'net_access', 'mortality_rate').
    input_folder : str
        Path to the folder containing the combined CSV file for the specified data type.
    output_folder : str
        Path to the folder where the filtered CSV file with complete time series will be saved.

    """
    csv_1 = os.path.join(input_folder, f'{data_type}_combined.csv')
    df = pd.read_csv(csv_1)

    total_years = df['year'].nunique()

    coord_counts = (df.groupby(['longitude', 'latitude'])
        ['year'].nunique().reset_index(name = 'year_count'))

    valid_coords = coord_counts[coord_counts['year_count'] == total_years]

    filtered_df = df.merge(valid_coords[['longitude', 'latitude']],
                        on = ['longitude', 'latitude'],
                        how = 'inner')

    metric_name = filtered_df['metric'].iloc[0]
    csv_name = f'{metric_name}.csv'
    out_folder = os.path.join(output_folder, 'malaria_indices')
    os.makedirs(out_folder, exist_ok = True)
    path_out = os.path.join(out_folder, csv_name)
    filtered_df.to_csv(path_out, index=False)

    print("Done! Only complete coordinate time series retained.")

    return None