"""
Role: Select and merge the malaria index input variables from raw DHS data.
Description: Reads the raw zipped DHS/MIS archives directly, rather than the
harmonized per-recode-type CSVs from dhs.py, because those keep only columns
common to every survey round and several requested variables here exist in
one round only. Selects the researcher-chosen columns from the Individual
and Household recodes for each round, concatenates each recode type across
rounds, then left-joins Household onto Individual, one row per eligible
woman, matched by cluster and household number within the same survey
round. Writes one CSV, ready to merge with the GPS cluster file afterward
on cluster number.
Author: Bonny
"""

import logging
import os
import string
import tempfile
import zipfile
import geopandas as gpd
import pandas as pd
from pathlib import Path
from dream.dhSurvey import (READABLE_EXTENSIONS,
    extract_nested_zips, load_tabular_file,
    recode_type, survey_round_label)
log = logging.getLogger(__name__)


def _numeric_suffix_range(
    base: str, start: int, end: int, sep: str = "_", width: int | None = None
) -> list[str]:
    """
    Build a list of column names with a numeric suffix:

    Parameters
    ----------
    base : str.
        Variable name before the suffix, for example "h22".
    start : int.
        First suffix number, inclusive.
    end : int.
        Last suffix number, inclusive.
    sep : str.
        Separator between the base and the suffix.
    width : int or None.
        Zero-pad the suffix to this many digits, for example 2 for
        "01".."30". No padding when None.

    Returns
    -------
    result : list[str]
            The generated column names, for example ["h22_1", ...,
            "h22_6"].

    """
    names = []
    for i in range(start, end + 1):
        suffix = f"{i:0{width}d}" if width else str(i)
        names.append(f"{base}{sep}{suffix}")
    return names


def _alpha_suffix_range(base: str, start: str, end: str) -> list[str]:
    """
    Build a list of column names with a letter suffix:

    Parameters
    ----------
    base : str.
        Variable name before the suffix, for example "s351".
    start : str.
        First suffix letter, inclusive.
    end : str.
        Last suffix letter, inclusive.

    Returns
    -------
    result : list[str]
            The generated column names, for example ["s351a", ...,
            "s351z"].

    """
    letters = string.ascii_lowercase
    return [f"{base}{c}" for c in letters[letters.index(start) : letters.index(end) + 1]]


# DHS recode files follow the CCRRVVFL naming convention. These are the
# columns that uniquely identify a record, and the columns that join one
# recode type to another. Always kept, on top of whatever is in the
# INDEX_COLUMNS lists below.
ID_COLUMNS: dict[str, list[str]] = {
    "IR": ["v001", "v002", "v003", "caseid"],
    "HR": ["hv001", "hv002", "hhid"]}

GE_COLUMNS = ['DHSID', 'DHSCLUST', 'LATNUM', 'LONGNUM']

NET_USE_COLUMNS = ['ml101']
 
INDIVIDUAL_COLUMNS = ['v005', 'v024', 'v025', 'v040', 'v106', 'v190', 'v191', 
                      'survey_round']
 
HOUSEHOLD_COLUMNS = ['hhid', 'hv227', 'hml1', 'hv228', 'hml2', 'hv005', 
                     'hv024', 'hv025', 'hv040', 'hv270', 'hv271', 'hv009', 
                     'hv014', 'hv216', 'hv213', 'hv214', 'hv215', 'hv206', 
                     'hv226', 'hml4_1']
 
CLUSTER_ID_COLUMNS = ['DHSID', 'GPS_Dataset', 'DHSCC', 'DHSYEAR', 
                      'DHSCLUST', 'SurveyID']
 
COORDINATE_COLUMNS = ['LATNUM', 'LONGNUM']
 
ENVIRONMENTAL_BLOCK_START = 'SurveyID'
ENVIRONMENTAL_BLOCK_END = 'DHSID_ge'
 
COLUMN_RENAME_MAP = {'hv001': 'cluster_number', 'hv002': 'household_number',
    'v003': 'respondent_line_number', 'ml101': 'net_type_last_night',
    'v005': 'women_sample_weight', 'v024': 'region_ir', 'v025': 'residence_type_ir',
    'v040': 'altitude_m_ir', 'v106': 'education_level', 'v190': 'wealth_quintile_ir',
    'v191': 'wealth_score_ir', 'hhid': 'household_id', 'hv227': 'has_mosquito_net',
    'hml1': 'total_nets_owned', 'hv228': 'under5_slept_net_last_night',
    'hml2': 'under5_net_users_count', 'hv005': 'household_sample_weight', 'hv024': 'region_hr',
    'hv025': 'residence_type_hr', 'hv040': 'altitude_m_hr', 'hv270': 'wealth_quintile_hr',
    'hv271': 'wealth_score_hr', 'hv009': 'household_size', 'hv014': 'children_under5_count',
    'hv216': 'sleeping_rooms_count', 'hv213': 'floor_material', 'hv214': 'wall_material',
    'hv215': 'roof_material', 'hv206': 'has_electricity', 'hv226': 'cooking_fuel_type',
    'hml4_1': 'net1_age_months'}

IR_COLUMNS: list[str] = (
    _numeric_suffix_range("h22", 1, 6)
    + _numeric_suffix_range("h47", 1, 6)
    + _numeric_suffix_range("h46a", 1, 6)
    + _numeric_suffix_range("h37f", 1, 6)
    + _numeric_suffix_range("h37h", 1, 6)
    + ["ml101"]
    + _numeric_suffix_range("ml0", 1, 6)
    + _numeric_suffix_range("m49a", 1, 6)
    + _numeric_suffix_range("ml1", 1, 6)
    + _numeric_suffix_range("ml2", 1, 6)
    + _numeric_suffix_range("ml20a", 1, 6)
    + ["ml501"]
    + _alpha_suffix_range("ml501", "a", "z")
    + _alpha_suffix_range("ml503", "a", "x")
    + [f"ml{n}" for n in range(505, 515)]
    + ["v005", "v024", "v025", "v040", "v106", "v190", "v191"]
)

HR_PREFIX_COLUMNS: list[str] = ["hml10_", "hml21_", "hml22_", "hml4_", "hml7_"]

HR_COLUMNS: list[str] = (
    ["hv227", "hml1"]
    + _numeric_suffix_range("hml20", 1, 30, width=2)
    + ["hv228", "hml2"]
    + ["hv253", "hv253a", "hv253b", "hv253c"]
    + ["sh120a", "sh120c"]
    + ["sh120d", "sh120e", "sh120f", "shpmi"]
    + ["hv005", "hv024", "hv025", "hv040", "hv270", "hv271", "hv009", "hv014", "hv216"]
    + ["hv213", "hv214", "hv215", "hv206", "hv223", "hv226"]
    + ["sh131a", "sh131b", "sh131c", "sh131d"]
)

INDEX_COLUMNS: dict[str, list[str]] = {
    "IR": IR_COLUMNS,
    "HR": HR_COLUMNS,
}


def select_index_columns(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    Select the malaria index columns present in one round's data frame:

    Parameters
    ----------
    df : pd.DataFrame.
        One round of one recode type, as loaded from its source file.
    code : str.
        The two-letter recode type, one of ID_COLUMNS's keys.

    Returns
    -------
    result : pd.DataFrame
            df restricted to the ID columns plus the requested index
            columns that are present in df. A requested column absent
            from this round is logged at info level and left out,
            since a survey round not asking a given question is
            expected, not an error.

    """
    wanted = list(dict.fromkeys(ID_COLUMNS[code] + INDEX_COLUMNS[code]))

    if code == "HR":
        prefix_hits = [
            c
            for c in df.columns
            if any(
                c.startswith(prefix) and c[len(prefix) :].isdigit()
                for prefix in HR_PREFIX_COLUMNS
            )
        ]
        wanted = list(dict.fromkeys(wanted + prefix_hits))

    present = [c for c in wanted if c in df.columns]
    missing = sorted(set(wanted) - set(present) - set(HR_PREFIX_COLUMNS))
    if missing:
        log.info(
            "%s: %s requested column(s) not present in this round: %s",
            code,
            len(missing),
            ", ".join(missing),
        )
    return df[present]


def build_malaria_index_table(
    base_dir: str,
    output_path: str,
    zip_glob: str = "*.zip",
    overwrite: bool = False,
) -> str:
    """
    Combine the malaria index columns from the raw DHS archives into one
    CSV, one row per eligible woman recorded in the Individual Recode:

    Parameters
    ----------
    base_dir : str.
        Folder holding the raw, zipped DHS downloads.
    output_path : str.
        Path of the CSV file to write.
    zip_glob : str.
        Glob pattern used to find the zip files in base_dir.
    overwrite : bool.
        When False and output_path already exists, the existing file
        is left as is and the source archives are not read again.
        When True, it is regenerated.

    Returns
    -------
    result : str
            output_path. The Individual Recode is the base of every
            row, left-joined to the household (Household Recode, by
            cluster and household number). A woman whose household
            has no matching Household Recode row, which is not
            expected but is not assumed away either, has the
            household columns as missing on her row rather than being
            dropped from the table. The join is scoped to the same
            survey round, since cluster numbers restart each round
            and are not comparable across rounds. If base_dir holds
            more than one archive covering the same survey round, for
            example a DHS download and a GPS download that both carry
            the full recode files, each recode type is deduplicated
            on its natural key within a round before the join runs,
            and the dropped row count is logged as a warning.

    """
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        log.info("%s already present, nothing to do", output_path)
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    base_dir = os.path.abspath(base_dir)
    zip_paths = sorted(Path(base_dir).glob(zip_glob))
    if not zip_paths:
        raise FileNotFoundError(
            f"no zip file matching {zip_glob} found in {base_dir}"
        )

    frames_by_type: dict[str, list[pd.DataFrame]] = {code: [] for code in ID_COLUMNS}

    with tempfile.TemporaryDirectory() as tmp_root:
        for zip_path in zip_paths:
            extract_dir = os.path.join(tmp_root, zip_path.stem)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
            extract_nested_zips(extract_dir)

            for root, _dirs, files in os.walk(extract_dir):
                for fname in files:
                    ext = Path(fname).suffix.lower()
                    if ext not in READABLE_EXTENSIONS:
                        continue

                    code = recode_type(Path(fname).stem)
                    if code is None or code not in ID_COLUMNS:
                        continue

                    fpath = os.path.join(root, fname)
                    round_folder = os.path.relpath(root, extract_dir).split(
                        os.sep
                    )[0]
                    round_label = survey_round_label(round_folder)

                    try:
                        df = load_tabular_file(fpath)
                    except (ValueError, UnicodeDecodeError, OSError) as exc:
                        log.warning("could not read %s: %s", fpath, exc)
                        continue

                    selected = select_index_columns(df, code).copy()
                    selected["survey_round"] = round_label
                    frames_by_type[code].append(selected)

    per_type: dict[str, pd.DataFrame] = {}
    for code, frames in frames_by_type.items():
        if not frames:
            log.warning("no source file found for recode type %s", code)
            continue
        combined = pd.concat(frames, ignore_index=True)

        key_columns = [c for c in ID_COLUMNS[code] if c in combined.columns] + [
            "survey_round"
        ]
        rows_before = len(combined)
        combined = combined.drop_duplicates(subset=key_columns, keep="first")
        rows_dropped = rows_before - len(combined)
        if rows_dropped:
            log.warning(
                "%s: dropped %s row(s) sharing the same %s within one "
                "round, most likely the same survey round supplied by "
                "more than one source archive in base_dir",
                code,
                rows_dropped,
                key_columns,
            )

        log.info(
            "%s: %s row(s), %s column(s) across %s round(s)",
            code,
            len(combined),
            combined.shape[1],
            len(frames),
        )
        per_type[code] = combined

    if "IR" not in per_type:
        raise ValueError(
            "no Individual Recode (IR) data was found, and the malaria "
            "index table is built with one row per woman from that file"
        )

    merged = per_type["IR"].rename(columns={"v001": "hv001", "v002": "hv002"})

    if "HR" in per_type:
        merged = merged.merge(
            per_type["HR"],
            on=["hv001", "hv002", "survey_round"],
            how="left",
            suffixes=("", "_hr"),
        )
    else:
        log.warning(
            "no Household Recode (HR) data was found, writing Individual "
            "Recode columns only"
        )

    merged.to_csv(output_path, index=False)
    log.info(
        "wrote %s with %s row(s) and %s column(s)",
        output_path,
        len(merged),
        merged.shape[1],
    )
    return output_path


def load_gps_clusters(base_dir, zip_glob: str = '*.zip') -> pd.DataFrame:
    """
    Load and combine the GPS cluster covariates (GC) file across every
    survey round found under a raw folder:
 
    Parameters
    ----------
    base_dir : str.
        Folder holding the raw, zipped DHS or GPS downloads.
    zip_glob : str.
        Glob pattern used to find the zip files in base_dir.
 
    Returns
    -------
    result : pd.DataFrame
            One row per surveyed cluster, with a survey_round column
            added.
 
    """
    base_dir = os.path.abspath(base_dir)
    zip_paths = sorted(Path(base_dir).glob(zip_glob))
    if not zip_paths:
 
        raise FileNotFoundError(
            f'no zip file matching {zip_glob} found in {base_dir}')
 
    frames: list[pd.DataFrame] = []
 
    with tempfile.TemporaryDirectory() as tmp_root:
 
        for zip_path in zip_paths:
 
            extract_dir = os.path.join(tmp_root, zip_path.stem)
            with zipfile.ZipFile(zip_path) as zf:
 
                zf.extractall(extract_dir)
            extract_nested_zips(extract_dir)
 
            for root, _dirs, files in os.walk(extract_dir):
 
                for fname in files:
 
                    ext = Path(fname).suffix.lower()
                    if ext not in READABLE_EXTENSIONS:
 
                        continue
 
                    code = recode_type(Path(fname).stem)
                    if code != 'GC':
 
                        continue
 
                    fpath = os.path.join(root, fname)
                    round_folder = os.path.relpath(root, extract_dir).split(
                        os.sep)[0]
                    round_label = survey_round_label(round_folder)
 
                    try:
 
                        df = load_tabular_file(fpath)
                    except (ValueError, UnicodeDecodeError, OSError) as exc:
 
                        log.warning('could not read %s: %s', fpath, exc)
                        continue
 
                    df = df.copy()
                    df['survey_round'] = round_label
                    frames.append(df)
 
    if not frames:
 
        raise ValueError(f'no GPS cluster (GC) file found under {base_dir}')
 
    combined = pd.concat(frames, ignore_index = True)
 
    key_columns = [c for c in ('DHSCLUST',) if c in combined.columns] + [
        'survey_round']
    rows_before = len(combined)
    combined = combined.drop_duplicates(subset = key_columns, keep = 'first')
    rows_dropped = rows_before - len(combined)
    if rows_dropped:
 
        log.warning(
            'GC: dropped %s row(s) sharing the same %s within one round',
            rows_dropped, key_columns)
 
    log.info('GC: %s cluster(s) across %s round(s)', len(combined), len(frames))

 
    return combined


def load_gps_locations(base_dir: str, zip_glob: str = '*.zip') -> pd.DataFrame | None:
    """
    Load and combine the GPS displacement (GE) shapefile across every
    survey round found under a raw folder:
 
    Parameters
    ----------
    base_dir : str.
        Folder holding the raw, zipped DHS or GPS downloads.
    zip_glob : str.
        Glob pattern used to find the zip files in base_dir.
 
    Returns
    -------
    result : pd.DataFrame.
 
    """
    base_dir = os.path.abspath(base_dir)
    zip_paths = sorted(Path(base_dir).glob(zip_glob))
    if not zip_paths:
 
        raise FileNotFoundError(
            f'no zip file matching {zip_glob} found in {base_dir}')
 
    frames: list[pd.DataFrame] = []
 
    with tempfile.TemporaryDirectory() as tmp_root:
 
        for zip_path in zip_paths:
 
            extract_dir = os.path.join(tmp_root, zip_path.stem)
            with zipfile.ZipFile(zip_path) as zf:
 
                zf.extractall(extract_dir)
            extract_nested_zips(extract_dir)
 
            for root, _dirs, files in os.walk(extract_dir):
 
                for fname in files:
 
                    if not fname.lower().endswith('.shp'):
 
                        continue

                    if Path(fname).stem[2:4].upper() != 'GE':
 
                        continue
 
                    fpath = os.path.join(root, fname)
                    round_folder = os.path.relpath(root, extract_dir).split(
                        os.sep)[0]
                    round_label = survey_round_label(round_folder)
 
                    try:
 
                        gdf = gpd.read_file(fpath)
                    except (ValueError, OSError) as exc:
 
                        log.warning('could not read %s: %s', fpath, exc)
                        continue
 
                    present = [c for c in GE_COLUMNS if c in gdf.columns]
                    missing = sorted(set(GE_COLUMNS) - set(present))
                    if missing:
 
                        log.info(
                            'GE: %s column(s) not present in %s: %s',
                            len(missing), fname, ', '.join(missing))
 
                    df = pd.DataFrame(gdf[present])
                    df['survey_round'] = round_label
                    frames.append(df)
 
    if not frames:
 
        log.warning('no GE shapefile found under %s', base_dir)
        return None
 
    combined = pd.concat(frames, ignore_index = True)
 
    key_columns = [c for c in ('DHSCLUST',) if c in combined.columns] + [
        'survey_round']
    rows_before = len(combined)
    combined = combined.drop_duplicates(subset = key_columns, keep = 'first')
    rows_dropped = rows_before - len(combined)
    if rows_dropped:
 
        log.warning(
            'GE: dropped %s row(s) sharing the same %s within one round',
            rows_dropped, key_columns)
 
    log.info('GE: %s cluster(s) across %s round(s)', len(combined), len(frames))

 
    return combined


def merge_with_gps_clusters(index_path, gps_base_dir,
    output_path, zip_glob: str = '*.zip',
    cluster_column: str = 'hv001', overwrite: bool = False) -> str:
    """
    Left-join a malaria index table to the GPS cluster covariates and,
    when geopandas is installed, the GPS coordinates:
 
    Parameters
    ----------
    index_path : str.
        Path of the malaria index CSV, for example the output of
        build_malaria_index_table. Must have a cluster_column and a
        survey_round column.
    gps_base_dir : str.
        Folder holding the raw, zipped DHS or GPS downloads that
        contain the GPS cluster covariates (GC) file and, for
        coordinates, the GPS displacement (GE) shapefile.
    output_path : str.
        Path of the CSV file to write.
    zip_glob : str.
        Glob pattern used to find the zip files in gps_base_dir.
    cluster_column : str.
        Name of the cluster number column in index_path, for example
        'hv001' or 'v001' depending on which recode the index table
        is anchored on.
    overwrite : bool.
        When False and output_path already exists, the existing file
        is left as is and the raw archives are not read again. When
        True, it is regenerated.
 
    Returns
    -------
    result : str
            output_path.
    """
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
 
        log.info('%s already present, nothing to do', output_path)
 
        return output_path
 
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok = True)
 
    index_df = pd.read_csv(index_path, low_memory = False)
    for required in (cluster_column, 'survey_round'):
 
        if required not in index_df.columns:
 
            raise ValueError(f"{index_path} has no '{required}' column to join on")
 
    index_df = index_df.copy()
    index_df['_join_cluster'] = index_df[cluster_column].astype('Int64')
 
    gps_df = load_gps_clusters(gps_base_dir, zip_glob)
    gps_df['DHSCLUST'] = gps_df['DHSCLUST'].astype('Int64')
 
    merged = index_df.merge(gps_df,
        left_on = ['_join_cluster', 'survey_round'],
        right_on = ['DHSCLUST', 'survey_round'],
        how = 'left', suffixes = ('', '_gc'))
 
    unmatched_gc = merged['DHSCLUST'].isna().sum()
    if unmatched_gc:
 
        log.warning(
            '%s of %s row(s) in %s found no matching GC cluster for their '
            '(%s, survey_round) pair', unmatched_gc, len(merged),
            index_path, cluster_column)
 
    locations_df = load_gps_locations(gps_base_dir, zip_glob)
    if locations_df is not None:
 
        locations_df['DHSCLUST'] = locations_df['DHSCLUST'].astype('Int64')
        merged = merged.merge(locations_df,
            left_on = ['_join_cluster', 'survey_round'],
            right_on = ['DHSCLUST', 'survey_round'],
            how = 'left', suffixes = ('', '_ge'))
 
        unmatched_ge = merged['LATNUM'].isna().sum() if 'LATNUM' in merged else len(merged)
        if unmatched_ge:
 
            log.warning(
                '%s of %s row(s) in %s found no matching GE cluster for '
                'their (%s, survey_round) pair, and are missing LATNUM '
                'and LONGNUM', unmatched_ge, len(merged), index_path,
                cluster_column)
 
    merged = merged.drop(columns = ['_join_cluster'])
 
    merged.to_csv(output_path, index = False)
    log.info('wrote %s with %s row(s) and %s column(s)', output_path,
        len(merged), merged.shape[1])

 
    return output_path


def environmental_columns(df: pd.DataFrame) -> list[str]:
    """
    Read the derived environmental covariate columns out of a merged
    malaria survey table by their position:
 
    Parameters
    ----------
    df : pd.DataFrame.
        A table written by merge_with_gps_clusters.
 
    Returns
    -------
    result : list[str].
 
    """
    columns = list(df.columns)
 
    if ENVIRONMENTAL_BLOCK_START not in columns:

        log.warning(
            "'%s' not found, cannot locate the environmental covariate "
            'block',
            ENVIRONMENTAL_BLOCK_START)
        return []
 
    if ENVIRONMENTAL_BLOCK_END not in columns:

        log.warning(
            "'%s' not found, cannot locate the end of the environmental "
            'covariate block',
            ENVIRONMENTAL_BLOCK_END)
        return []
 
    start = columns.index(ENVIRONMENTAL_BLOCK_START) + 1
    end = columns.index(ENVIRONMENTAL_BLOCK_END)
    return columns[start:end]


def select_model_columns(input_path, output_path, 
                         overwrite: bool = False) -> str:
    """
    Select the final modeling columns from a merged malaria survey table
    and write them to a new CSV:
 
    Parameters
    ----------
    input_path : str.
        Path of the CSV written by merge_with_gps_clusters.
    output_path : str.
        Path of the CSV file to write.
    overwrite : bool.
        When False and output_path already exists, the existing file
        is left as is and input_path is not read again. When True, it
        is regenerated.
 
    Returns
    -------
    result : str
            output_path.
    """
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        log.info('%s already present, nothing to do', output_path)
        return output_path
 
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
 
    df = pd.read_csv(input_path, low_memory = False)
    ID_COLUMNS = ['hv001', 'hv002', 'v003', 'caseid']
 
    requested = (
        ID_COLUMNS
        + NET_USE_COLUMNS
        + INDIVIDUAL_COLUMNS
        + HOUSEHOLD_COLUMNS
        + CLUSTER_ID_COLUMNS
        + COORDINATE_COLUMNS
    )
    environmental = environmental_columns(df)
 
    wanted = list(dict.fromkeys(requested + environmental))
    present = [c for c in wanted if c in df.columns]
    missing = sorted(set(wanted) - set(present))
    if missing:

        log.warning('%s requested column(s) not present in %s: %s',
            len(missing), input_path, ', '.join(missing))
 
    selected = df[present].rename(columns = COLUMN_RENAME_MAP)
    selected.to_csv(output_path, index = False)
    log.info('wrote %s with %s row(s) and %s column(s)',
        output_path, len(selected), selected.shape[1])

    
    return output_path