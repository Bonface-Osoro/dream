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
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

ID_COLUMNS: dict[str, list[str]] = {
    "IR": ["v001", "v002", "v003", "caseid"],
    "HR": ["hv001", "hv002", "hhid"],
    "PR": ["hv001", "hv002", "hvidx", "hhid", "hc60"]}

GE_COLUMNS = ['DHSID', 'DHSCLUST', 'LATNUM', 'LONGNUM']

NET_USE_COLUMNS = ['ml101']
 
INDIVIDUAL_COLUMNS = ['v005', 'v006', 'v007', 'v024', 'v025', 'v040', 'v106',
                      'v190', 'v191', 'survey_round']
 
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
    'v003': 'mother_line_number', 'caseid': 'mother_caseid',
    'ml101': 'mother_net_type_last_night', 'v005': 'mother_sample_weight',
    'v024': 'mother_region', 'v025': 'mother_residence_type',
    'v040': 'mother_altitude_m', 'v106': 'mother_education_level',
    'v190': 'mother_wealth_quintile', 'v191': 'mother_wealth_score',
    'v006': 'interview_month', 'v007': 'interview_year',
    'hhid': 'household_id', 'hv227': 'has_mosquito_net',
    'hml1': 'total_nets_owned', 'hv228': 'under5_slept_net_last_night',
    'hml2': 'under5_net_users_count', 'hv005': 'household_sample_weight', 'hv024': 'region_hr',
    'hv025': 'residence_type_hr', 'hv040': 'altitude_m_hr', 'hv270': 'wealth_quintile_hr',
    'hv271': 'wealth_score_hr', 'hv009': 'household_size', 'hv014': 'children_under5_count',
    'hv216': 'sleeping_rooms_count', 'hv213': 'floor_material', 'hv214': 'wall_material',
    'hv215': 'roof_material', 'hv206': 'has_electricity', 'hv226': 'cooking_fuel_type',
    'hml4_1': 'net1_age_months',
    'hv104': 'child_sex', 'hv105': 'child_age_years', 'hml16a': 'child_age_months',
    'hc64': 'child_birth_order', 'hc63': 'child_preceding_birth_interval',
    'hml35': 'malaria_rdt_result', 'hml32': 'malaria_microscopy_result',
    'hml32a': 'malaria_species_falciparum', 'hml32b': 'malaria_species_malariae',
    'hml32c': 'malaria_species_ovale', 'hml32d': 'malaria_species_vivax',
    'hml33': 'malaria_measurement_status', 'spcr': 'malaria_pcr_result',
    'sspec': 'malaria_pcr_species',
    'hml37a': 'symptom_extreme_weakness', 'hml37b': 'symptom_heart_problems',
    'hml37c': 'symptom_loss_of_consciousness', 'hml37d': 'symptom_rapid_breathing',
    'hml37e': 'symptom_seizures', 'hml37f': 'symptom_abnormal_bleeding',
    'hml37g': 'symptom_jaundice', 'hml37h': 'symptom_dark_urine',
    'hml38': 'first_line_medicine_given', 'hml39': 'first_line_medicine_accepted'}

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
    + ["v005", "v024", "v025", "v040", "v106", "v190", "v191", "v006", "v007"]
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

PR_COLUMNS: list[str] = (
    ["hv104", "hv105", "hml16a", "hc64", "hc63"]
    + ["hml35", "hml32", "hml32a", "hml32b", "hml32c", "hml32d", "hml33", "spcr", "sspec"]
    + _alpha_suffix_range("hml37", "a", "h")
    + ["hml38", "hml39"]
)

INDEX_COLUMNS: dict[str, list[str]] = {
    "IR": IR_COLUMNS,
    "HR": HR_COLUMNS,
    "PR": PR_COLUMNS,
}

MATERIAL_COLUMNS = ['floor_material', 'wall_material', 'roof_material']

PROTECTIVE_COLUMNS = ['has_mosquito_net', 'total_nets_owned', 'net_coverage_score',
    'under5_net_users_count', 'has_electricity', 'wealth_score_hr', 'altitude_m_hr']
 
RISK_CATEGORIES = ['extremely low risk', 'low risk', 'high risk', 'extremely high risk']

NET_COVERAGE_MAP = {3: 0, 0: 1, 2: 2, 1: 3}

CARRIED_COLUMNS = [
    'cluster_number', 'household_number', 'hvidx', 'mother_caseid',
    'survey_round', 'LATNUM', 'LONGNUM',
]

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
            columns that are present in df..

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
    CSV, one row per child aged 0-5 recorded in the Person Recode:

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
            output_path.

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

    if "PR" not in per_type:
        raise ValueError(
            "no Person Recode (PR) data was found, and the malaria "
            "index table is built with one row per child from that file"
        )

    children = per_type["PR"]
    rows_before = len(children)
    children = children[children["hv105"] <= 5]
    log.info(
        "PR: kept %s of %s row(s) as children aged 0-5",
        len(children),
        rows_before,
    )

    merged = children

    if "HR" in per_type:
        merged = merged.merge(
            per_type["HR"],
            on=["hv001", "hv002", "survey_round"],
            how="left",
            suffixes=("", "_hr"),
        )
    else:
        log.warning(
            "no Household Recode (HR) data was found, writing Person "
            "Recode columns only"
        )

    if "IR" in per_type:
        # hc60 uses DHS sentinel codes (993, 995, and similar) for "no
        # mother listed", not real line numbers. No household reaches
        # that size, so they would fail to match on their own, but
        # clearing them here avoids a spurious float/int merge warning
        # and keeps the unmatched count below meaning one thing.
        merged["hc60"] = merged["hc60"].where(merged["hc60"] < 90)

        mothers = per_type["IR"].rename(columns={"v001": "hv001", "v002": "hv002"})
        merged = merged.merge(
            mothers,
            left_on=["hv001", "hv002", "hc60", "survey_round"],
            right_on=["hv001", "hv002", "v003", "survey_round"],
            how="left",
            suffixes=("", "_mother"),
        )
        unmatched_mothers = merged["v003"].isna().sum()
        if unmatched_mothers:
            log.warning(
                "%s of %s child row(s) found no matching mother in the "
                "Individual Recode for their (hc60, survey_round) pair",
                unmatched_mothers,
                len(merged),
            )
    else:
        log.warning(
            "no Individual Recode (IR) data was found, writing without "
            "the mother's columns"
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
    ID_COLUMNS = ['hv001', 'hv002', 'hvidx', 'hc60', 'v003', 'caseid']
 
    requested = (
        ID_COLUMNS
        + PR_COLUMNS
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


def material_risk_tier(series: pd.Series) -> pd.Series:
    """
    Recode a DHS floor, wall, or roof material column to an ordinal risk tier:
 
    Parameters
    ----------
    series : pd.Series.
        Raw floor_material, wall_material, or roof_material values, using
        the DHS tens-digit convention (1x natural, 2x rudimentary, 3x
        finished, 96 other).
 
    Returns
    -------
    result : pd.Series.
 
    """
    tens = (series // 10).where(series.notna())
    tier = tens.map({1: 3, 2: 2, 3: 1})
    unclassified = series.notna() & tier.isna()
    if unclassified.any():
        log.info(
            '%s value(s) outside the natural/rudimentary/finished '
            'convention, treated as missing', int(unclassified.sum()),
        )
    return tier


def net_coverage_score(series: pd.Series) -> pd.Series:
    """
    Recode under5_slept_net_last_night to a genuine coverage scale:
 
    Parameters
    ----------
    series : pd.Series.
        Raw under5_slept_net_last_night (hv228) values.
 
    Returns
    -------
    result : pd.Series.
 
    """
    mapped = series.map(NET_COVERAGE_MAP)
    unclassified = series.notna() & mapped.isna()
    if unclassified.any():
        log.info('%s value(s) outside the 0-3 hv228 coding, treated as '
            'missing', int(unclassified.sum()))

        
    return mapped


def crowding_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Compute people per sleeping room:
 
    Parameters
    ----------
    df : pd.DataFrame.
        Must have household_size and sleeping_rooms_count.
 
    Returns
    -------
    result : pd.Series.
 
    """
    rooms = df['sleeping_rooms_count'].where(df['sleeping_rooms_count'] > 0)
    household_size = df['household_size'] / rooms


    return household_size


def build_risk_index(
    input_path: str, output_path: str, overwrite: bool = False
) -> str:
    """
    Build the composite malaria risk index and write it back
    alongside the input table:
 
    Parameters
    ----------
    input_path : str.
        Path of the model-input CSV, for example the output of
        select_model_columns.
    output_path : str.
        Path of the CSV file to write.
    overwrite : bool.
        When False and output_path already exists, the existing file
        is left as is and input_path is not read again. When True, it
        is regenerated.
 
    Returns
    -------
    result : str
            output_path. cluster_number, household_number, hvidx,
            mother_caseid, survey_round, LATNUM, LONGNUM (carried
            through from input_path unchanged), malaria_risk_score
            (the first principal component, oriented so higher means
            higher risk, then min-max normalized to the 0-1 range),
            and malaria_risk_category (its quartile, one of
            RISK_CATEGORIES). A row missing any risk input, or with
            no linked mother, would have both as missing, but such
            rows are dropped from the output entirely rather than
            written with a blank score, so the row count here is
            usually less than input_path's. The row count dropped
            this way, the row count with a computed score, the
            variance the first component explains, and the raw
            range used for normalization are all logged so none of
            this is a black box.
 
    """
    CARRIED_COLUMNS = [
        'cluster_number', 'household_number', 'hvidx', 'mother_caseid',
        'survey_round', 'LATNUM', 'LONGNUM',
    ]
    OUTPUT_COLUMNS = CARRIED_COLUMNS + [
        'malaria_risk_score', 'malaria_risk_category',
    ]
    
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        log.info('%s already present, nothing to do', output_path)
        return output_path
 
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
 
    df = pd.read_csv(input_path, low_memory=False)
 
    derived = pd.DataFrame(index=df.index)
    for material in MATERIAL_COLUMNS:
        derived[f'{material}_tier'] = material_risk_tier(df[material])
    derived['net_coverage_score'] = net_coverage_score(
        df['under5_slept_net_last_night']
    )
    derived['crowding_ratio'] = crowding_ratio(df)
 
    # Guard against re-running this on a table that already has these
    # derived columns, for example this function's own earlier output,
    # rather than silently producing duplicate column names.
    already_present = [c for c in derived.columns if c in df.columns]
    if already_present:
        log.info(
            '%s derived column(s) already present in %s, replacing '
            'rather than duplicating: %s',
            len(already_present), input_path, ', '.join(already_present),
        )
        df = df.drop(columns=already_present)
 
    df = pd.concat([df, derived], axis=1)
 
    risk_matrix_columns = (
        ['has_mosquito_net', 'total_nets_owned', 'net_coverage_score',
         'under5_net_users_count']
        + [f'{m}_tier' for m in MATERIAL_COLUMNS]
        + ['has_electricity', 'crowding_ratio', 'wealth_score_hr',
           'altitude_m_hr']
    )
 
    oriented = df[risk_matrix_columns].copy()
    for col in PROTECTIVE_COLUMNS:
        oriented[col] = -oriented[col]
 
    complete = oriented.dropna()
    dropped = len(oriented) - len(complete)
    log.info(
        '%s of %s row(s) have every risk input and get a score; %s '
        'excluded for missing input',
        len(complete), len(oriented), dropped,
    )
    if complete.empty:
        raise ValueError('no row has every risk input; cannot fit PCA')
 
    scaler = StandardScaler()
    scaled = scaler.fit_transform(complete)
 
    pca = PCA(n_components=1)
    component = pca.fit_transform(scaled)[:, 0]
    log.info(
        'first principal component explains %.1f%% of variance',
        pca.explained_variance_ratio_[0] * 100,
    )
 
    loadings = pd.Series(
        pca.components_[0], index=risk_matrix_columns
    ).sort_values(key=abs, ascending=False)
    for name, weight in loadings.items():
        log.info('  loading %-28s %+.3f', name, weight)
 
    # PCA's sign is arbitrary. Orient so higher score means higher risk by
    # checking correlation with crowding_ratio, which is already oriented
    # that way and always available wherever the score is.
    anchor = complete['crowding_ratio']
    if np.corrcoef(component, anchor)[0, 1] < 0:
        component = -component
        log.info('flipped component sign so higher score means higher risk')
 
    score_min, score_max = component.min(), component.max()
    normalized = (component - score_min) / (score_max - score_min)
    log.info(
        'normalized to 0-1 using raw component range %.3f to %.3f',
        score_min, score_max,
    )
 
    scored = pd.Series(np.nan, index=df.index, name='malaria_risk_score')
    scored.loc[complete.index] = normalized
 
    category = pd.qcut(scored, q=4, labels=RISK_CATEGORIES)
    category.name = 'malaria_risk_category'
 
    missing_carried = [c for c in CARRIED_COLUMNS if c not in df.columns]
    if missing_carried:
        raise ValueError(
            f'{input_path} has no {missing_carried} column(s) to carry '
            'into the output'
        )
 
    result = pd.concat([df[CARRIED_COLUMNS], scored, category], axis=1)
    result = result[OUTPUT_COLUMNS]
 
    rows_before = len(result)
    result = result.dropna(
        subset=['malaria_risk_score', 'malaria_risk_category']
    )
    log.info(
        'dropped %s row(s) with no malaria_risk_score or '
        'malaria_risk_category',
        rows_before - len(result),
    )
 
    result.to_csv(output_path, index=False)
    log.info(
        'wrote %s with %s row(s) and %s column(s)',
        output_path, len(result), result.shape[1],
    )
    return output_path


def merge_risk_with_outcome(
    risk_path: str,
    refined_path: str,
    output_path: str,
    outcome_columns: list[str] | None = None,
    overwrite: bool = False,
) -> str:
    """
    Join the risk index back onto the model input table, for
    validating the score against a real outcome:
 
    Parameters
    ----------
    risk_path : str.
        Path of the risk index CSV, the output of build_risk_index.
    refined_path : str.
        Path of the model input CSV, the output of
        select_model_columns, holding the outcome columns
        build_risk_index does not carry.
    output_path : str.
        Path of the CSV file to write.
    outcome_columns : list[str] or None.
        Columns to bring in from refined_path. Defaults to
        ['malaria_rdt_result'] when None, the field DHS itself
        reports as headline malaria prevalence.
    overwrite : bool.
        When False and output_path already exists, the existing file
        is left as is and the two inputs are not read again. When
        True, it is regenerated.
 
    Returns
    -------
    result : str
            output_path. Every row of risk_path is kept, matched to
            refined_path on (cluster_number, household_number,
            hvidx, survey_round) together, the child's own identity,
            not mother_caseid, since a mother with more than one
            surveyed child under 5 shares one mother_caseid across
            several rows and would otherwise produce a many-to-many
            join.
 
    """
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        log.info('%s already present, nothing to do', output_path)
        return output_path
 
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
 
    if outcome_columns is None:
        outcome_columns = ['malaria_rdt_result']
 
    risk = pd.read_csv(risk_path, low_memory=False)
    refined = pd.read_csv(refined_path, low_memory=False)
 
    join_key = ['cluster_number', 'household_number', 'hvidx', 'survey_round']
    required = join_key + outcome_columns
    missing = [c for c in required if c not in refined.columns]
    if missing:
        raise ValueError(
            f'{refined_path} has no {missing} column(s) needed for '
            'this join'
        )
 
    merged = risk.merge(
        refined[join_key + outcome_columns],
        on=join_key,
        how='left',
        indicator=True,
    )
 
    unmatched = (merged['_merge'] == 'left_only').sum()
    if unmatched:
        log.warning(
            '%s of %s row(s) in %s found no matching %s in %s',
            unmatched, len(merged), risk_path, join_key, refined_path,
        )
    merged = merged.drop(columns=['_merge'])
 
    outcome_missing = merged[outcome_columns[0]].isna().sum()
    if outcome_missing:
        log.info(
            '%s row(s) joined successfully but have no %s recorded, '
            'for example a child who was not tested',
            outcome_missing, outcome_columns[0],
        )
 
    merged.to_csv(output_path, index=False)
    log.info(
        'wrote %s with %s row(s) and %s column(s)',
        output_path, len(merged), merged.shape[1],
    )
    return output_path