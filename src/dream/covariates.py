import os
import glob
import logging
import numpy as np
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

# A child is identified by cluster, household, and line number within a
# survey round. mother_caseid does not identify a child, since siblings
# share it and a child with no linked mother has none. Longitude and
# latitude are left out on purpose: every child in a cluster shares
# them, and they are the values _check_coordinates compares across the
# three files.
MERGE_KEY = [
    'cluster_number', 'household_number', 'hvidx', 'mother_caseid',
    'year', 'month', 'survey_round',
]

CHILD_COLUMNS = [
    'cluster_number', 'household_number', 'hvidx', 'mother_caseid',
    'survey_round',
]

FINAL_COLUMNS = [
    'year', 'longitude', 'latitude', 'ndvi', 'month',
    'precipitation_mm', 'temperature_C',
]

COORDINATE_TOLERANCE_DEG = 1e-6

# What merge_monthly_covariates does with a child that appears twice.
REPEAT_POLICIES = ('raise', 'collapse')


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
        by=['year', 'longitude', 'latitude', 'month']
    )
    combined = combined[[
        'cluster_number', 'household_number', 'hvidx', 'mother_caseid',
        'year', 'month', 'survey_round', 'longitude', 'latitude',
        covariate,
    ]]

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


def _one_row_per_location_month(table: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a child-level table to one row per location and month:

    Parameters
    ----------
    table : pd.DataFrame.
        Must hold year, longitude, latitude, and month plus the
        covariate columns, one row per child and month.

    Returns
    -------
    result : pd.DataFrame
            One row per year, longitude, latitude, and month. The
            covariates are read from a raster at the location, so
            every child at one location carries the same values and
            the repeats carry no information. Raises ValueError when
            two rows at one location and month disagree on a
            covariate, since keeping the first would then discard a
            value silently.

    """
    key = ['year', 'longitude', 'latitude', 'month']
    values = [c for c in table.columns if c not in key]
    spread = table.groupby(key)[values].nunique()
    conflicts = int((spread > 1).any(axis=1).sum())
    if conflicts:
        raise ValueError(
            f'{conflicts} location and month combination(s) hold more '
            'than one value of a covariate'
        )
    result = table.drop_duplicates(subset=key)
    log.info(
        '%s child row(s) collapsed to %s row(s), one per location and '
        'month', len(table), len(result),
    )
    return result


def _collapse_repeated_children(
    df: pd.DataFrame, path: str, value_col: str
) -> pd.DataFrame:
    """
    Keep one row for each child and month that appears more than once:

    Parameters
    ----------
    df : pd.DataFrame.
        One covariate table from combine_covariate_monthly.
    path : str.
        File the table came from, named in messages.
    value_col : str.
        Name of the covariate column in df.

    Returns
    -------
    result : pd.DataFrame
            The table with the first row kept for every child,
            survey_round, year, and month, and the later copies
            skipped. The covariates are read from a raster at the
            child's location, so copies at the same location carry
            the same values. When the copies of a child disagree on
            mother_caseid, the row kept has it set to missing, since
            choosing one would be a guess. The number of rows
            skipped, and the children affected, are logged as a
            warning. Raises ValueError when copies disagree on a
            coordinate or on the covariate value, since skipping one
            would then discard real information.

    """
    key = [
        'cluster_number', 'household_number', 'hvidx', 'survey_round',
        'year', 'month',
    ]
    child = ['cluster_number', 'household_number', 'hvidx', 'survey_round']
    repeated = df.duplicated(subset=key, keep=False)
    if not repeated.any():
        return df

    copies = df[repeated].groupby(key)
    spread = copies[['longitude', 'latitude', value_col]].nunique(
        dropna=False
    )
    conflicts = int((spread > 1).any(axis=1).sum())
    if conflicts:
        raise ValueError(
            f'{path}: {conflicts} child and month combination(s) '
            f'repeat with a different coordinate or {value_col} value'
        )

    mothers = copies['mother_caseid'].nunique(dropna=False)
    ambiguous = mothers[mothers > 1].index
    kept = df.drop_duplicates(subset=key, keep='first').copy()
    unclear = kept.set_index(key).index.isin(ambiguous)
    kept.loc[unclear, 'mother_caseid'] = np.nan

    affected = df.loc[repeated, child].drop_duplicates()
    unclear_children = kept.loc[unclear, child].drop_duplicates()
    log.warning(
        '%s: skipped %s repeated row(s) of %s child(ren); %s of them '
        'had copies with different mother_caseid, now set to '
        'missing. First few: %s',
        Path(path).name, len(df) - len(kept), len(affected),
        len(unclear_children), affected.head(6).to_dict('records'),
    )
    return kept


def merge_monthly_covariates(
    ndvi_csv: str,
    precipitation_csv: str,
    temperature_csv: str,
    output_csv: str,
    overwrite: bool = False,
    collapse_to_locations: bool = False,
    repeats: str = 'raise',
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
    collapse_to_locations : bool.
        When False, every child is kept, one row per child and
        month, including children who share a mother or a
        coordinate. When True, the rows are reduced to one per year,
        longitude, latitude, and month and the child columns are
        dropped.
    repeats : str.
        What to do when a child and month appears more than once in
        a covariate file. 'raise' stops with an error, since a merge
        on a repeated key would multiply rows. 'collapse' keeps the
        first copy and skips the others, with a warning, see
        _collapse_repeated_children.

    Returns
    -------
    result : str
            output_csv. Children are matched across the three files
            on MERGE_KEY. A key present in one file but not another
            has the missing file's covariate as NA, and the count is
            logged. The columns are CHILD_COLUMNS followed by
            FINAL_COLUMNS, or FINAL_COLUMNS alone when
            collapse_to_locations is True. Raises ValueError when a
            file repeats a MERGE_KEY value, since the merge would
            then multiply the other files' rows; when two files
            disagree on a shared key's coordinates by more than
            COORDINATE_TOLERANCE_DEG; or, when collapsing, when one
            location and month holds two different values of a
            covariate. Logs a warning when the files spell the
            months differently, such as sep against sept, since
            those rows then fail to match. Raises ValueError when
            repeats is not 'raise' or 'collapse'.

    """
    if repeats not in REPEAT_POLICIES:
        raise ValueError(
            f'repeats must be one of {REPEAT_POLICIES}, got {repeats!r}'
        )

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

        if repeats == 'collapse':
            df = _collapse_repeated_children(df, path, value_col)

        repeated = df.duplicated(subset=MERGE_KEY)
        if repeated.any():
            raise ValueError(
                f'{path} has {int(repeated.sum())} row(s) that repeat '
                f'a value of MERGE_KEY {MERGE_KEY} already seen in '
                'that file; merging on it would multiply the matching '
                "rows in the other files (repeats='collapse' keeps "
                'one copy of each repeated child)'
            )
        frames[name] = df.rename(columns={
            'longitude': f'longitude_{name}',
            'latitude': f'latitude_{name}',
        })

    month_sets = {name: set(df['month']) for name, df in frames.items()}
    base_name, base_months = next(iter(month_sets.items()))
    for name, months in month_sets.items():
        if months != base_months:
            log.warning(
                '%s has month value(s) %s not seen in %s, and %s has '
                '%s not seen in %s; a spelling difference such as '
                'sep against sept stops those rows matching on the '
                'merge key', name, sorted(months - base_months),
                base_name, base_name, sorted(base_months - months),
                name,
            )

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
    merged = merged.rename(columns={
        'precipitation': 'precipitation_mm',
        'temperature': 'temperature_C',
    })

    no_coordinate = (merged['longitude'] == -9999) | (
        merged['latitude'] == -9999
    )
    if no_coordinate.any():
        log.info(
            'dropping %s row(s) whose coordinates are the -9999 '
            'no-data value', int(no_coordinate.sum()),
        )
        merged = merged.loc[~no_coordinate]

    if collapse_to_locations:
        result = _one_row_per_location_month(merged[FINAL_COLUMNS])
    else:
        result = merged[CHILD_COLUMNS + FINAL_COLUMNS]
    result.to_csv(output_path, index=False)
    log.info(
        'wrote %s with %s row(s) and %s column(s)',
        output_path, len(result), result.shape[1],
    )
    return str(output_path)
