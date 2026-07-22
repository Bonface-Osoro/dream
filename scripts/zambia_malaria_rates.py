"""
Merge malaria indicator datasets (parasite rate, incidence rate, mortality rate,
net access, net use) into one table keyed by location and year.

Each input file is a gridded raster exported to points (year, longitude, latitude,
value, metric). The grids for the five indicators don't sit on exactly the same
pixel centers, so locations are matched using nearest-neighbor spatial join with
a maximum matching distance of 5 km (great-circle / haversine distance). Years
are matched exactly.

Output: one row per (base location, year) with a column for each indicator.
Rows are NaN for an indicator if no grid cell of that dataset was found within
5 km of the base location.
"""
import configparser
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), 'script_config.ini'))
BASE_PATH = CONFIG['file_locations']['base_path']

DATA_RAW = os.path.join(BASE_PATH, 'raw')
DATA_PROCESSED = os.path.join(BASE_PATH, '..', 'results', 'processed')
DATA_RESULTS = os.path.join(BASE_PATH, '..', 'results', 'final')

EARTH_RADIUS_KM = 6371.0088
MAX_DIST_KM = 5.0

INPUT_DIR = os.path.join(DATA_PROCESSED, 'zambia', 'malaria_indices')
FILES = {
    "parasite_rate": f"{INPUT_DIR}/parasite_rate.csv",
    "incidence_rate": f"{INPUT_DIR}/incidence_rate.csv",
    "mortality_rate": f"{INPUT_DIR}/mortality_rate.csv",
    "net_access": f"{INPUT_DIR}/net_access.csv",
    "net_use": f"{INPUT_DIR}/net_use.csv",
}

BASE_METRIC = "parasite_rate"

OUTPUT_PATH = os.path.join(DATA_RESULTS, 'spatial_temporal_data', 
                           'ZMB_spatio_temporal_malaria_indices.csv')


def unique_coords(df):

    """Distinct (longitude, latitude) 
    pairs in a dataset, each given a point_id."""

    u = df[["longitude", "latitude"]].drop_duplicates().reset_index(drop = True)
    u["point_id"] = np.arange(len(u))

    return u


def nearest_neighbor_ids(base_latlon_rad, target_coords,
                          max_dist_km):
    """
    For every base point, find the nearest point in target_coords.
    Returns (matched_point_id array, distance_km array), with point_id = -1
    and distance = NaN where nothing was within max_dist_km.
    """
    tree = BallTree(np.radians(target_coords[["latitude", 
                                              "longitude"]].values),
                     metric = "haversine")
    dist, idx = tree.query(base_latlon_rad, k = 1)

    dist_km = dist[:, 0] * EARTH_RADIUS_KM

    matched_point_id = target_coords["point_id"].values[idx[:, 0]]

    within = dist_km <= max_dist_km
    matched_point_id = np.where(within, matched_point_id, -1)
    dist_km = np.where(within, dist_km, np.nan)


    return matched_point_id, dist_km



def main():
    data = {name: pd.read_csv(path) for name, path in FILES.items()}

    # 1. Build the base grid (unique locations) from the anchor dataset.
    base_coords = unique_coords(data[BASE_METRIC])
    base_rad = np.radians(base_coords[["latitude", 
                                       "longitude"]].values)

    # location table that will become the backbone of the output
    locations = base_coords.rename(columns={"point_id": "location_id"})[
        ["location_id", "longitude", "latitude"]
    ]

    # 2. For every dataset, map each base location to that dataset's nearest
    #    grid point (within MAX_DIST_KM), once (grids repeat identically every year).
    point_id_maps = {}   
    dist_report = {}
    for name, df in data.items():

        coords = unique_coords(df)
        if name == BASE_METRIC:

            matched_pid = base_coords["point_id"].values
            dist_km = np.zeros(len(base_coords))
        else:
            matched_pid, dist_km = nearest_neighbor_ids(
                base_rad, coords, MAX_DIST_KM)

        point_id_maps[name] = pd.Series(matched_pid, 
                                        index = locations["location_id"])
        dist_report[name] = dist_km

        n_matched = np.sum(matched_pid != -1)
        print(f"{name}: {n_matched}/{len(base_coords)} base locations matched "
              f"within {MAX_DIST_KM} km "
              f"(median dist {np.nanmedian(dist_km):.3f} km)")

    # 3. Build a fast (point_id, year) -> value lookup for each dataset.
    value_lookups = {}
    for name, df in data.items():

        coords = unique_coords(df)
        merged = df.merge(coords, on = ["longitude", 
                                        "latitude"], how = "left")
        value_lookups[name] = merged.set_index(
            ["point_id", "year"])["value"]

    # 4. Cross join locations x years, then pull in each indicator's value
    #    via its point_id mapping.
    years = sorted(pd.unique(pd.concat([df["year"] for df in data.values()])))
    out = locations.merge(pd.DataFrame({"year": years}), how = "cross")

    for name in FILES:

        pid_map = point_id_maps[name]
        out_pid = out["location_id"].map(pid_map)
        lookup = value_lookups[name]
        keys = list(zip(out_pid, out["year"]))
        out[name] = [lookup.get(k, np.nan) if k[0] != -1 else np.nan for k in keys]

    out = out.sort_values(["location_id", "year"]).reset_index(drop = True)
    out = out.drop(columns = ["location_id"])
    out.to_csv(OUTPUT_PATH, index = False)


if __name__ == "__main__":

    main()
