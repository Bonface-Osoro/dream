"""
This script combines zipped DHS and MIS survey archives into one CSV per recode type. It unpacks zip files nested at 
any depth under a raw data folder, reads every Stata, SPSS, or CSV file inside them whose name matches one of the six target 
DHS recode types, and concatenates all survey rounds of a given type into a single CSV. Survey phases add and drop questionnaire
variables every round, so only columns present in every round of a given type are kept, rather than filling the gaps with NaN, 
which also keeps memory use bounded. A file that cannot be read, and a recode type with no matching file at all, are both 
logged as a warning, never silently dropped. Takes its folders as arguments. Does not read configuration itself, so the caller 
decides where the raw and processed folders live.
"""
 
import logging
import os
import tempfile
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
 
log = logging.getLogger(__name__)
 
READABLE_EXTENSIONS = {".dta", ".sav", ".csv"}
 
# DHS recode files follow the CCRRVVFL naming convention, where CC is the
# two-letter country code and RR is the two-letter recode type. These are
# the recode types this pipeline needs, mapped to the label used in the
# output file name.
RECODE_LABELS = {
    "IR": "IndividualRecode",
    "HR": "HouseholdRecode",
    "BR": "BirthRecode",
    "PR": "PersonRecode",
    "KR": "KidsRecode",
    "GC": "GPSClusters",
}
 
def extract_nested_zips(root_dir: str) -> None:
    """
    Extract every zip file nested under a root folder:
 
    Parameters
    ----------
    root_dir : str.
        Folder to search. Searched repeatedly until no zip file
        remains, so a zip file inside another zip file is reached too.
 
    Returns
    -------
    None
            Extraction happens in place. Nothing is returned.
 
    """
    found_one = True
    while found_one:
        found_one = False
        for dirpath, _dirs, files in os.walk(root_dir):
            for fname in files:
                if not fname.lower().endswith(".zip"):
                    continue
                zip_path = os.path.join(dirpath, fname)
                target = os.path.join(dirpath, Path(fname).stem)
                os.makedirs(target, exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(target)
                except zipfile.BadZipFile as exc:
                    log.warning("could not open zip %s: %s", zip_path, exc)
                os.remove(zip_path)
                found_one = True
            if found_one:
                break
    log.info("finished unpacking nested zip files under %s", root_dir)


def recode_type(file_stem: str) -> str | None:
    """
    Read the DHS recode type out of a file name:
 
    Parameters
    ----------
    file_stem : str.
        File name without its extension, for example "UGIR72FL",
        following the CCRRVVFL convention where CC is the two-letter
        country code and RR is the two-letter recode type.
 
    Returns
    -------
    result : str or None
            The two-letter recode type, for example "IR", when it is
            one of the types named in RECODE_LABELS. None otherwise,
            for a file this pipeline does not target.
 
    """
    code = file_stem[2:4].upper()
    return code if code in RECODE_LABELS else None
 
 
def survey_round_label(round_folder_name: str) -> str:
    """
    Read a short survey round label out of a round folder name:
 
    Parameters
    ----------
    round_folder_name : str.
        Name of the folder a per-round zip file was extracted into,
        for example "UG_2014-15_MIS_09212026_1323_150620".
 
    Returns
    -------
    result : str
            The round segment of the name, for example "2014-15", or
            the full folder name when it does not split as expected.
 
    """
    parts = round_folder_name.split("_")
    return parts[1] if len(parts) > 1 else round_folder_name
 
 
def load_tabular_file(file_path: str) -> pd.DataFrame:
    """
    Load a Stata, SPSS, or CSV file into a data frame:
 
    Parameters
    ----------
    file_path : str.
        Path of the file to load.
 
    Returns
    -------
    result : pd.DataFrame
            The loaded data.
 
    """
    ext = Path(file_path).suffix.lower()
 
    if ext == ".dta":
        try:
            return pd.read_stata(file_path, convert_categoricals=False)
        except UnicodeDecodeError:
            return pd.read_stata(
                file_path, convert_categoricals=False, encoding="latin-1"
            )
    elif ext == ".sav":
        return pd.read_spss(file_path)
    elif ext == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"no reader configured for extension {ext}")
 

def convert_dhs_zips_to_csv(base_dir, output_dir, zip_glob: str = "*.zip",
                            overwrite: bool = False,) -> list[str]:
    """
    Combine the DHS recode files under a raw folder into one CSV per
    recode type:
 
    Parameters
    ----------
    base_dir : str.
        Folder holding the raw, zipped DHS downloads. Built and passed
        in by the caller, for example one country's raw data folder.
    output_dir : str.
        Folder the CSV files are written to. Built and passed in by
        the caller, for example one country's processed data folder.
    zip_glob : str.
        Glob pattern used to find the zip files in base_dir.
    overwrite : bool.
        When False, a CSV file already present in output_dir is left
        as is, and its source files are not read again. When True, it
        is regenerated.
 
    Returns
    -------
    written : list[str]
            Paths of the CSV files present in output_dir after the
            call, one per target recode type found in the data,
            whether written now or already there from an earlier run.
            Each file holds only the columns common to every survey
            round of that type, since the DHS questionnaire changes
            variables between rounds and a round-specific column is
            not one shared series. A source file that cannot be read
            is logged as a warning and excluded from the combined
            file, not counted as converted. A recode type with no
            matching file anywhere under base_dir is logged as a
            warning and has no CSV written for it, rather than an
            empty placeholder file.
 
    """
    base_dir = os.path.abspath(base_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
 
    zip_paths = sorted(Path(base_dir).glob(zip_glob))
    if not zip_paths:
        log.warning("no zip file matching %s found in %s", zip_glob, base_dir)
        return []
 
    written: list[str] = []
    pending_codes: set[str] = set()
 
    for code, label in RECODE_LABELS.items():
        out_path = os.path.join(output_dir, f"{label}_{code}.csv")
        if os.path.exists(out_path) and not overwrite:
            written.append(out_path)
        else:
            pending_codes.add(code)
 
    if not pending_codes:
        log.info(
            "all %s target file(s) already present in %s",
            len(RECODE_LABELS),
            output_dir,
        )
        return written
 
    frames_by_type: dict[str, list[pd.DataFrame]] = {
        code: [] for code in pending_codes
    }
    skipped: list[str] = []
 
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
                    if code is None or code not in pending_codes:
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
                        skipped.append(fpath)
                        continue
 
                    df["survey_round"] = round_label
                    frames_by_type[code].append(df)
 
    for code in pending_codes:
        label = RECODE_LABELS[code]
        out_path = os.path.join(output_dir, f"{label}_{code}.csv")
        frames = frames_by_type[code]
 
        if not frames:
            log.warning("no source file found for recode type %s", code)
            continue
 
        combined = pd.concat(frames, ignore_index=True, join="inner")
        union_columns = set()
        for frame in frames:
            union_columns.update(frame.columns)
        dropped_columns = len(union_columns) - len(combined.columns)
        if dropped_columns:
            log.info(
                "%s: kept %s column(s) common to every survey round, "
                "dropped %s round-specific column(s)",
                label,
                len(combined.columns),
                dropped_columns,
            )
 
        rows_before = len(combined)
        combined = combined.drop_duplicates()
        rows_dropped = rows_before - len(combined)
        if rows_dropped:
            log.info(
                "dropped %s duplicate row(s) from %s", rows_dropped, label
            )
 
        combined.to_csv(out_path, index=False)
        written.append(out_path)
        log.info(
            "wrote %s with %s row(s) from %s source file(s)",
            out_path,
            len(combined),
            len(frames),
        )
 
    log.info(
        "%s of %s target file(s) present in %s, skipped %s unreadable file(s)",
        len(written),
        len(RECODE_LABELS),
        output_dir,
        len(skipped),
    )
    return written