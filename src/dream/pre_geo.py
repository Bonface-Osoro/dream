"""
Preprocessing tools for raw satellite data.

Developed by Bonface Osoro.

March 2026

"""
import os
import glob
import re
import warnings
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


def shp_to_csv(data_type, in_folder, output_folder):
    """
    This function reads different shapefiles and 
    convert them into csv files for further processing.

    Parameters
    ----------
    data_type : str
        The type of data to be processed (e.g., 
        'incidence_rate', 'parasite_rate', 'net_use', 
        'net_access', 'mortality_rate').
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


def combine_year_files(input_folder, output_folder):
    """
    This function combines multiple csv files 
    containing malaria indices of same location 
    and year into a single csv file.

    Parameters
    ----------
    input_folder : str
        Path to folder containing CSV files.
    output_folder : str
        Path to folder where combined CSV will be saved.

    """

    # Get all CSV files
    csv_files = [os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith('.csv')]

    if not csv_files:
        print('No CSV files found in the input folder.')
        return None

    # Read and combine
    df_list = []
    for file in csv_files:

        try:

            df = pd.read_csv(file)
            df_list.append(df)

        except Exception as e:

            print(f'Skipping {file} due to error: {e}')

    if not df_list:

        print('No valid CSV files to combine.')
        return None

    combined_df = pd.concat(df_list, ignore_index = True)

    os.makedirs(output_folder, exist_ok = True)
    output_path = os.path.join(output_folder, 
                               'spatio_temporal_malaria_indices.csv')
    combined_df.to_csv(output_path, index = False)

    print(f'\nDone! Combined CSV saved to:\n{output_path}')

    return combined_df


    import pandas as pd


def normalize_column(df, column_name):
    """
    Normalizes a column in a DataFrame to values between 0 and 1.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of the column to normalize

    Returns:
        pd.Series: Normalized column
    """
    col = df[column_name]
    
    min_val = col.min()
    max_val = col.max()
    
    # Handle edge case where all values are the same
    if min_val == max_val:
        return pd.Series([0.0] * len(col), index=col.index)
    
    normalized = (col - min_val) / (max_val - min_val)
    return normalized


def eco_mri(annual_csv_path, monthly_csv_path, output_csv_path):
    ''''
    This is afunction for merging csv files 
    containing predictors using ['latitude', 
    'longitude', 'year'] as keys.

    Parameters
    ----------

        annual_csv_path : str
            Path to annual MRI CSV file
        monthly_csv_path : str
            Path to monthly predictors CSV file
        output_csv_path : str
            Path where the merged CSV file will be saved

    Returns
    -------
        pd.DataFrame: Merged DataFrame
    '''

    annual_df = pd.read_csv(annual_csv_path)
    monthly_df = pd.read_csv(monthly_csv_path)

    if 'mri_value' not in annual_df.columns:
        
        raise ValueError("annual CSV must contain 'mri_value' column")

    annual_df = annual_df[['year', 'longitude', 'latitude', 'mri_value']]

    annual_df['year'] = annual_df['year'].astype(int)
    monthly_df['year'] = monthly_df['year'].astype(int)

    for col in ['longitude', 'latitude']:

        annual_df[col] = annual_df[col].round(5)
        monthly_df[col] = monthly_df[col].round(5)

    merged_df = pd.merge(monthly_df, annual_df,
        on = ['year', 'longitude', 'latitude'],
        how = 'left')

    missing = merged_df['mri_value'].isna().sum()
    if missing > 0:

        print(f"Warning: {missing} rows did not match and have missing MRI values.")

    merged_df.to_csv(output_csv_path, index = False)

    print(f"Merge complete. File saved to: {output_csv_path}")

    return merged_df


def combine_ndvi_monthly(input_folder, output_folder):
    """
    This function combines multiple monthly NDVI CSV 
    files into a single CSV file.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the monthly NDVI CSV files to be combined.
    output_file : str
        Path to the output CSV file where the combined data will be saved.

    """
    files = glob.glob(os.path.join(input_folder, '*.xlsx'))
    
    all_dfs = []

    for file in tqdm(files, desc = 'Combining monthly NDVI files', 
                     unit = 'file'):

        df = pd.read_excel(file)
        
        filename = os.path.basename(file)
        month = filename.split("_")[-1].split(".")[0].lower()
        
        df = df.drop(columns=['OBJECTID', 'rainfall', 'normalized_rain'], 
                     errors = 'ignore')
        
        df = df.rename(columns = {'RASTERVALU': 'ndvi'})
        df['month'] = month
        df = df.drop_duplicates(subset = ['year', 'longitude', 'latitude'])
        
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sort_values(by = ['year', 'longitude', 'latitude', 'month'])
    os.makedirs(output_folder, exist_ok = True)

    output_path = os.path.join(output_folder, 'combined_monthly_ndvi.csv')
    final_df.to_csv(output_path, index = False)


    return None


def combine_rain_monthly(input_folder, output_folder):
    """
    This function combines multiple monthly rain CSV 
    files into a single CSV file.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the monthly rain CSV files to be combined.
    output_file : str
        Path to the output CSV file where the combined data will be saved.

    """
    files = glob.glob(os.path.join(input_folder, '*.xlsx'))
    
    all_dfs = []

    for file in tqdm(files, desc = 'Combining monthly precipitation files', 
                     unit = 'file'):

        df = pd.read_excel(file)
        
        filename = os.path.basename(file)
        month = filename.split("_")[-1].split(".")[0].lower()
        
        df = df.drop(columns=['OBJECTID', 'rainfall', 'normalized_rain'], 
                     errors = 'ignore')
        
        df = df.rename(columns = {'RASTERVALU': 'precipitation_mm'})
        df['month'] = month
        df = df.drop_duplicates(subset = ['year', 'longitude', 'latitude'])
        
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sort_values(by = ['year', 'longitude', 'latitude', 'month'])
    os.makedirs(output_folder, exist_ok = True)

    output_path = os.path.join(output_folder, 'combined_monthly_rain.csv')
    final_df.to_csv(output_path, index = False)


    return None


def combine_monthly_temp(input_folder, output_folder):
    """
    This function combines multiple monthly temperature CSV 
    files into a single CSV file.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the monthly temperature CSV files to be combined.
    output_file : str
        Path to the output CSV file where the combined data will be saved.

    """
    files = glob.glob(os.path.join(input_folder, '*.xlsx'))
    
    all_dfs = []

    for file in tqdm(files, desc = 'Combining monthly temperature files', 
                     unit = 'file'):

        df = pd.read_excel(file)
        
        filename = os.path.basename(file)
        month = filename.split("_")[-1].split(".")[0].lower()
        
        df = df.drop(columns=['OBJECTID', 'rainfall', 'normalized_rain'], 
                     errors = 'ignore')
        
        df = df.rename(columns = {'RASTERVALU': 'temperature_C'})
        df['temperature_C'] = df['temperature_C'] - 273.15
        df['month'] = month
        df = df.drop_duplicates(subset = ['year', 'longitude', 'latitude'])
        
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sort_values(by = ['year', 'longitude', 'latitude', 'month'])
    os.makedirs(output_folder, exist_ok = True)

    output_path = os.path.join(output_folder, 'combined_monthly_temperature.csv')
    final_df.to_csv(output_path, index = False)


    return None


def merge_eco_datasets(ndvi_path, elev_path, precip_path, temp_path, output_path):
    """This function merges multiple climate datasets (NDVI, elevation, 
    precipitation, temperature)
    
    Parameters
    ----------
    ndvi_path : str
        Path to the NDVI CSV file.
    elev_path : str
        Path to the elevation CSV file.
    precip_path : str
        Path to the precipitation CSV file.
    temp_path : str
        Path to the temperature CSV file.
    output_path : str
        Path to the output CSV file where the merged data will be saved.
    """

    ndvi_df = pd.read_csv(ndvi_path)
    elev_df = pd.read_csv(elev_path)
    precip_df = pd.read_csv(precip_path)
    temp_df = pd.read_csv(temp_path)

    keys = ['year', 'longitude', 'latitude', 'month']

    merged_df = ndvi_df.merge(elev_df, on = keys, how = 'inner')
    merged_df = merged_df.merge(precip_df, on = keys, how = 'inner')
    merged_df = merged_df.merge(temp_df, on = keys, how = 'inner')

    merged_df = merged_df.sort_values(by = keys)

    merged_df.to_csv(output_path, index=False)

    return merged_df