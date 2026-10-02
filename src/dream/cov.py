"""
Role: Combine and merge monthly environmental covariates.

Description: Reads monthly ArcGIS extraction files, standardizes their
columns, combines each covariate into a yearly CSV, and merges NDVI,
precipitation, and temperature using the survey identifiers, year, month,
longitude, and latitude as the exact merge keys.

Author: Bor
"""

import glob
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


log = logging.getLogger(__name__)


RAW_COLUMNS = [
    'FID',
    'cluster_nu',
    'household_',
    'hvidx',
    'mother_cas',
    'survey_rou',
    'LATNUM',
    'LONGNUM',
    'malaria_ri',
    'malaria__1',
    'RASTERVALU',
]

NODATA_SENTINEL = '-9999'

MONTH_ABBREVIATIONS = {
    'jan',
    'feb',
    'mar',
    'apr',
    'may',
    'jun',
    'jul',
    'aug',
    'sep',
    'sept',
    'oct',
    'nov',
    'dec',
}

COVARIATES = {
    'ndvi',
    'precipitation',
    'temperature',
}

MERGE_KEY = [
    'cluster_number',
    'mother_caseid',
    'year',
    'month',
    'longitude',
    'latitude',
]

FINAL_COLUMNS = [
    'year',
    'longitude',
    'latitude',
    'ndvi',
    'month',
    'precipitation_mm',
    'temperature_C',
]

COORDINATE_TOLERANCE_DEG = 1e-6


def _parse_year_month(file_path: str) -> tuple[int, str]:
    """
    Read the year and month from a <year>_<month>.xlsx file name.

    Parameters
    ----------
    file_path : str
        Path of the Excel file.

    Returns
    -------
    tuple[int, str]
        Year and lower-case month abbreviation.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    parts = stem.split('_')

    if (
        len(parts) != 2
        or not parts[0].isdigit()
        or len(parts[0]) != 4
    ):
        raise ValueError(
            f"{file_path} is not named '<year>_<month>.xlsx'"
        )

    year = int(parts[0])
    month = parts[1].lower()

    if month not in MONTH_ABBREVIATIONS:
        raise ValueError(
            f'{file_path} has an unrecognized month {month!r}'
        )

    return year, month


def _clear_nodata(
    df: pd.DataFrame,
    column: str,
    file_path: str,
) -> None:
    """
    Replace the GIS -9999 no-data sentinel with a missing value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame modified in place.
    column : str
        Column to check.
    file_path : str
        Source file used in the log message.

    Returns
    -------
    None
        The DataFrame is modified in place.
    """
    is_nodata = (
        df[column]
        .astype(str)
        .str.strip()
        == NODATA_SENTINEL
    )

    if is_nodata.any():
        log.info(
            '%s: %s row(s) have the %s no-data value in %s, '
            'set to missing',
            os.path.basename(file_path),
            int(is_nodata.sum()),
            NODATA_SENTINEL,
            column,
        )

        df.loc[is_nodata, column] = pd.NA


def combine_covariate_monthly(
    input_folder: str,
    output_folder: str,
    covariate: str,
) -> str:
    """
    Combine monthly covariate extraction tables into one CSV.

    Parameters
    ----------
    input_folder : str
        Folder containing '<year>_<month>.xlsx' files.
    output_folder : str
        Folder where the combined CSV is written.
    covariate : str
        One of 'ndvi', 'precipitation', or 'temperature'.

    Returns
    -------
    str
        Path to the combined CSV.
    """
    if covariate not in COVARIATES:
        raise ValueError(
            f'covariate must be one of {sorted(COVARIATES)}, '
            f'got {covariate!r}'
        )

    files = sorted(
        glob.glob(
            os.path.join(input_folder, '*.xlsx')
        )
    )

    if not files:
        raise FileNotFoundError(
            f'no .xlsx file found in {input_folder}'
        )

    all_dfs = []

    for file_path in tqdm(
        files,
        desc=f'Combining monthly {covariate} files',
        unit='file',
    ):
        year, month = _parse_year_month(file_path)

        df = pd.read_excel(file_path)

        missing_columns = [
            column
            for column in RAW_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f'{file_path} is missing column(s) '
                f'{missing_columns}'
            )

        _clear_nodata(
            df,
            'mother_cas',
            file_path,
        )

        _clear_nodata(
            df,
            'RASTERVALU',
            file_path,
        )

        df = df.rename(
            columns={
                'RASTERVALU': covariate,
                'LONGNUM': 'longitude',
                'LATNUM': 'latitude',
                'mother_cas': 'mother_caseid',
                'cluster_nu': 'cluster_number',
                'household_': 'household_number',
                'survey_rou': 'survey_round',
            }
        )

        df['year'] = year
        df['month'] = month

        all_dfs.append(df)

    combined = pd.concat(
        all_dfs,
        ignore_index=True,
    )

    combined = combined.sort_values(
        by=[
            'year',
            'longitude',
            'latitude',
            'month',
        ]
    )

    combined = combined[
        [
            'cluster_number',
            'household_number',
            'hvidx',
            'mother_caseid',
            'year',
            'month',
            'survey_round',
            'longitude',
            'latitude',
            covariate,
        ]
    ]

    os.makedirs(
        output_folder,
        exist_ok=True,
    )

    output_path = os.path.join(
        output_folder,
        f'UGA_combined_monthly_{covariate}.csv',
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    log.info(
        'wrote %s with %s row(s) and %s column(s)',
        output_path,
        len(combined),
        combined.shape[1],
    )

    return output_path


def _check_coordinates(
    merged: pd.DataFrame,
    sources: list[str],
    tolerance: float,
) -> None:
    """
    Confirm that all source-specific coordinates agree.

    Parameters
    ----------
    merged : pd.DataFrame
        DataFrame containing longitude_<source> and
        latitude_<source> columns.
    sources : list[str]
        Covariate source names.
    tolerance : float
        Maximum coordinate difference in degrees.

    Returns
    -------
    None
        Raises ValueError if source coordinates disagree.
    """
    base = sources[0]

    for other in sources[1:]:
        lon_diff = (
            merged[f'longitude_{base}']
            - merged[f'longitude_{other}']
        ).abs()

        lat_diff = (
            merged[f'latitude_{base}']
            - merged[f'latitude_{other}']
        ).abs()

        has_other = (
            merged[f'longitude_{other}'].notna()
            & merged[f'latitude_{other}'].notna()
        )

        disagrees = (
            (
                (lon_diff > tolerance)
                | (lat_diff > tolerance)
            )
            & has_other
        )

        if disagrees.any():
            raise ValueError(
                f'{int(disagrees.sum())} row(s) disagree on '
                f'coordinates between {base!r} and {other!r} '
                f'by more than {tolerance} degrees'
            )


def merge_monthly_covariates(
    ndvi_csv: str,
    precipitation_csv: str,
    temperature_csv: str,
    output_csv: str,
    overwrite: bool = False,
) -> str:
    """
    Merge NDVI, precipitation, and temperature using exact survey,
    temporal, and coordinate keys.

    Parameters
    ----------
    ndvi_csv : str
        Path to the combined NDVI CSV.
    precipitation_csv : str
        Path to the combined precipitation CSV.
    temperature_csv : str
        Path to the combined temperature CSV.
    output_csv : str
        Path to the merged output CSV.
    overwrite : bool
        Whether to regenerate an existing output file.

    Returns
    -------
    str
        Path to the merged output CSV.
    """
    output_path = Path(output_csv)

    if output_path.exists() and not overwrite:
        log.info(
            '%s already present, nothing to do',
            output_path,
        )
        return str(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = {
        'ndvi': (
            ndvi_csv,
            'ndvi',
        ),
        'precipitation': (
            precipitation_csv,
            'precipitation',
        ),
        'temperature': (
            temperature_csv,
            'temperature',
        ),
    }

    frames: dict[str, pd.DataFrame] = {}

    for name, (path, value_col) in sources.items():
        df = pd.read_csv(
            path,
            low_memory=False,
        )

        required = MERGE_KEY + [
            value_col,
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f'{path} is missing column(s) {missing}'
            )

        # Keep copies of each source's coordinates so the final
        # merged table can verify that all sources agree.
        df[f'longitude_{name}'] = df['longitude']
        df[f'latitude_{name}'] = df['latitude']

        frames[name] = df

        log.info(
            '%s: loaded %s row(s) from %s',
            name,
            len(df),
            path,
        )

    merged = frames['ndvi']

    for name in (
        'precipitation',
        'temperature',
    ):
        before = len(merged)

        merged = merged.merge(
            frames[name],
            on=MERGE_KEY,
            how='outer',
            indicator=True,
            suffixes=(
                '',
                f'_{name}',
            ),
        )

        unmatched = int(
            (
                merged['_merge']
                != 'both'
            ).sum()
        )

        if unmatched:
            log.warning(
                '%s of %s row(s) have no exact match between '
                'the running merge and %s',
                unmatched,
                len(merged),
                sources[name][0],
            )

        merged = merged.drop(
            columns=['_merge']
        )

        log.info(
            '%s: %s row(s) before, %s after merging %s',
            name,
            before,
            len(merged),
            sources[name][0],
        )

    _check_coordinates(
        merged,
        [
            'ndvi',
            'precipitation',
            'temperature',
        ],
        COORDINATE_TOLERANCE_DEG,
    )

    merged = merged.rename(
        columns={
            'precipitation': 'precipitation_mm',
            'temperature': 'temperature_C',
        }
    )

    # Remove rows with missing or invalid coordinates.
    valid_coordinates = (
        merged['longitude'].notna()
        & merged['latitude'].notna()
        & (merged['longitude'] != -9999)
        & (merged['latitude'] != -9999)
    )

    removed_rows = int(
        (~valid_coordinates).sum()
    )

    if removed_rows:
        log.warning(
            'removed %s row(s) with missing or invalid '
            'longitude/latitude',
            removed_rows,
        )

    merged = merged[
        valid_coordinates
    ].copy()

    result = merged[
        FINAL_COLUMNS
    ]

    result.to_csv(
        output_path,
        index=False,
    )

    log.info(
        'wrote %s with %s row(s) and %s column(s)',
        output_path,
        len(result),
        result.shape[1],
    )

    return str(output_path)