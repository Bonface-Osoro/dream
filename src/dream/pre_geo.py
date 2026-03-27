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