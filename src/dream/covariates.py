import os
import glob
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

log = logging.getLogger(__name__)
 
# ArcGIS truncates field names over ten characters, and both exports
# used here (the shapefile attribute join and the raster extraction)
# use -9999 to mark a value that is not available, never a value a
# child or a pixel can legitimately hold.
RAW_COLUMNS = [
    'FID', 'cluster_nu', 'household_', 'hvidx', 'mother_cas',
    'survey_rou', 'LATNUM', 'LONGNUM', 'malaria_ri', 'malaria__1',
    'RASTERVALU',
]
NODATA_SENTINEL = '-9999'
MONTH_ABBREVIATIONS = {
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
    'sept', 'oct', 'nov', 'dec',
}
COVARIATES = {'ndvi', 'precipitation', 'temperature'}

MERGE_KEY = [
    'cluster_number', 'mother_caseid', 'year', 'month', 'longitude', 'latitude']
 
FINAL_COLUMNS = [
    'year', 'longitude', 'latitude', 'ndvi', 'month',
    'precipitation_mm', 'temperature_C',
]
 
COORDINATE_TOLERANCE_DEG = 1e-6
 
 
def _parse_year_month(file_path: str) -> tuple[int, str]:
    """
    Read the year and month out of a <year>_<month>.xlsx file name:
 
    Parameters
    ----------
    file_path : str.
        Path of the Excel file, for example '.../2009_jan.xlsx'.
 
    Returns
    -------
    result : tuple[int, str]
            The year and the lower-case month abbreviation.
 
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    parts = stem.split('_')
    if len(parts) != 2 or not parts[0].isdigit() or len(parts[0]) != 4:
        raise ValueError(
            f"{file_path} is not named '<year>_<month>.xlsx'"
        )
    year, month = int(parts[0]), parts[1].lower()
    if month not in MONTH_ABBREVIATIONS:
        raise ValueError(
            f'{file_path} has an unrecognized month {month!r}'
        )
    return year, month
 
 
def _clear_nodata(df: pd.DataFrame, column: str, file_path: str) -> None:
    """
    Replace the -9999 GIS no-data sentinel with a true missing value:
 
    Parameters
    ----------
    df : pd.DataFrame.
        Modified in place.
    column : str.
        Column to check for the sentinel.
    file_path : str.
        Source file, named in the log message.
 
    Returns
    -------
    None
            Nothing is returned. The count replaced is logged, so a
            file with no-data rows is visible, not silently cleaned.
 
    """
    is_nodata = df[column].astype(str).str.strip() == NODATA_SENTINEL
    if is_nodata.any():
        log.info(
            '%s: %s row(s) have the %s no-data value in %s, set to '
            'missing',
            os.path.basename(file_path), int(is_nodata.sum()),
            NODATA_SENTINEL, column,
        )
        df.loc[is_nodata, column] = pd.NA
 
 
def combine_covariate_monthly(
    input_folder: str, output_folder: str, covariate: str,
) -> str:
    """
    Combine the monthly covariate extraction tables into one CSV:
 
    Parameters
    ----------
    input_folder : str.
        Folder holding the '<year>_<month>.xlsx' files.
    output_folder : str.
        Folder the combined CSV is written to.
    covariate : str.
        Which covariate the files hold, one of 'ndvi',
        'precipitation', or 'temperature'.
 
    Returns
    -------
    result : str
            Path of 'UGA_combined_monthly_<covariate>.csv' inside
            output_folder.
 
    """
    if covariate not in COVARIATES:
        raise ValueError(
            f'covariate must be one of {sorted(COVARIATES)}, got '
            f'{covariate!r}'
        )
 
    files = sorted(glob.glob(os.path.join(input_folder, '*.xlsx')))
    if not files:
        raise FileNotFoundError(f'no .xlsx file found in {input_folder}')
 
    all_dfs = []
    for file_path in tqdm(
        files, desc=f'Combining monthly {covariate} files', unit='file'
    ):
        year, month = _parse_year_month(file_path)
        df = pd.read_excel(file_path)
 
        missing_columns = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing_columns:
            raise ValueError(
                f'{file_path} is missing column(s) {missing_columns}'
            )
 
        _clear_nodata(df, 'mother_cas', file_path)
        _clear_nodata(df, 'RASTERVALU', file_path)
 
        df = df.rename(columns={
            'RASTERVALU': covariate,
            'LONGNUM': 'longitude',
            'LATNUM': 'latitude',
            'mother_cas': 'mother_caseid',
            'cluster_nu': 'cluster_number',
            'household_': 'household_number',
            'survey_rou': 'survey_round',
        })
        df['year'] = year
        df['month'] = month
 
        all_dfs.append(df)
 
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(
        by = ['year', 'longitude', 'latitude', 'month'])
    combined = combined[['cluster_number', 'household_number', 'hvidx', 
                         'mother_caseid', 'year', 'month', 'survey_round', 
                         'longitude', 'latitude', covariate]]
 
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder, f'UGA_combined_monthly_{covariate}.csv'
    )
    combined.to_csv(output_path, index=False)
    log.info(
        'wrote %s with %s row(s) and %s column(s)',
        output_path, len(combined), combined.shape[1],
    )
    return output_path

def _check_coordinates(
    merged: pd.DataFrame, sources: list[str], tolerance: float,
) -> None:
    """
    Confirm every source agrees on each row's coordinates:
 
    Parameters
    ----------
    merged : pd.DataFrame.
        Must have a longitude and latitude column per source, named
        'longitude_<source>' and 'latitude_<source>'.
    sources : list[str].
        Names used to build those column names, in the order the
        files were merged.
    tolerance : float.
        Largest difference, in degrees, treated as the same point
        rather than a disagreement.
 
    Returns
    -------
    None
            Nothing is returned. Raises ValueError naming how many
            rows disagree and against which source, rather than
            silently keeping one source's coordinates and discarding
            the others as if they were known to match.
 
    """
    base = sources[0]
    for other in sources[1:]:
        lon_diff = (
            merged[f'longitude_{base}'] - merged[f'longitude_{other}']
        ).abs()
        lat_diff = (
            merged[f'latitude_{base}'] - merged[f'latitude_{other}']
        ).abs()
        has_other = merged[f'longitude_{other}'].notna()
        disagrees = (
            (lon_diff > tolerance) | (lat_diff > tolerance)
        ) & has_other
        if disagrees.any():
            raise ValueError(
                f'{int(disagrees.sum())} row(s) disagree on '
                f'coordinates between {base!r} and {other!r} by '
                f'more than {tolerance} degrees'
            )
 
 
def merge_monthly_covariates(
    ndvi_csv: str,
    precipitation_csv: str,
    temperature_csv: str,
    output_csv: str,
    overwrite: bool = False,
) -> str:
    """
    Merge the separately extracted monthly covariates into one table:
 
    Parameters
    ----------
    ndvi_csv : str.
        Path of the CSV from combine_covariate_monthly with
        covariate='ndvi'.
    precipitation_csv : str.
        Path of the CSV from combine_covariate_monthly with
        covariate='precipitation'.
    temperature_csv : str.
        Path of the CSV from combine_covariate_monthly with
        covariate='temperature'.
    output_csv : str.
        Path of the CSV file to write.
    overwrite : bool.
        When False and output_csv already exists, the existing file
        is left as is and the inputs are not read again. When True,
        it is regenerated.
 
    Returns
    -------
    result : str
            output_csv.
 
    """
    output_path = Path(output_csv)
    if output_path.exists() and not overwrite:
        log.info('%s already present, nothing to do', output_path)
        return str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    sources = {
        'ndvi': (ndvi_csv, 'ndvi'),
        'precipitation': (precipitation_csv, 'precipitation'),
        'temperature': (temperature_csv, 'temperature'),
    }
    frames = {}
    for name, (path, value_col) in sources.items():
        df = pd.read_csv(path, low_memory=False)
        required = MERGE_KEY + ['longitude', 'latitude', value_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'{path} is missing column(s) {missing}')
        frames[name] = df.rename(columns={
            'longitude': f'longitude_{name}',
            'latitude': f'latitude_{name}',
        })
 
    merged = frames['ndvi']
    for name in ('precipitation', 'temperature'):
        before = len(merged)
        merged = merged.merge(
            frames[name], on=MERGE_KEY, how='outer', indicator=True,
        )
        unmatched = int((merged['_merge'] != 'both').sum())
        if unmatched:
            log.warning(
                '%s of %s row(s) have no match between the running '
                'merge and %s', unmatched, len(merged), sources[name][0],
            )
        merged = merged.drop(columns=['_merge'])
        log.info(
            '%s: %s row(s) before, %s after merging %s',
            name, before, len(merged), sources[name][0],
        )
 
    _check_coordinates(
        merged, ['ndvi', 'precipitation', 'temperature'],
        COORDINATE_TOLERANCE_DEG,
    )
    merged['longitude'] = merged['longitude_ndvi']
    merged['latitude'] = merged['latitude_ndvi']
    merged = merged.rename(columns = {'precipitation': 'precipitation_mm',
        'temperature': 'temperature_C'})
    merged = merged[(merged['longitude'] != -9999) & (merged['latitude'] != -9999)].copy()
 
    result = merged[FINAL_COLUMNS]
    result.to_csv(output_path, index=False)
    log.info(
        'wrote %s with %s row(s) and %s column(s)',
        output_path, len(result), result.shape[1],
    )
    return str(output_path)