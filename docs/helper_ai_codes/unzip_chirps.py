"""
Role: Decompress gzipped CHIRPS rainfall rasters into one folder.
Description: Finds every gzipped raster file in a source folder,
decompresses each with gzip, not zip, since these are single-file
gzip archives rather than zip archives, and writes the resulting
.tif file to a destination folder. Streams each file through a
fixed-size buffer rather than loading it fully into memory, since a
CHIRPS raster can be tens of megabytes. A file that is not valid
gzip is logged as a warning and skipped, never silently dropped from
the count without explanation.
Author: Bonny
"""

import gzip
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def unzip_chirps_rasters(
    source_dir: str,
    output_dir: str,
    pattern: str = '*.tif.gz',
    overwrite: bool = False,
) -> list[str]:
    """
    Decompress every matching gzip file into one output folder:

    Parameters
    ----------
    source_dir : str.
        Folder holding the gzipped raster files, for example a
        Downloads folder.
    output_dir : str.
        Folder the decompressed .tif files are written to.
    pattern : str.
        Glob pattern used to find the gzip files in source_dir.
    overwrite : bool.
        When False, a .tif file already present in output_dir is
        left as is and not regenerated. When True, it is
        regenerated.

    Returns
    -------
    result : list[str]
            Paths of the .tif files present in output_dir after the
            call, one per matching gzip file, named the same as the
            source file with the .gz suffix removed. A source file
            that is not valid gzip is logged as a warning and
            excluded from the result, not counted as converted.

    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gz_paths = sorted(source_path.glob(pattern))
    if not gz_paths:
        log.warning('no file matching %s found in %s', pattern, source_dir)
        return []

    written = []
    skipped = []
    for gz_path in gz_paths:
        tif_path = output_path / gz_path.with_suffix('').name

        if tif_path.exists() and not overwrite:
            log.info('%s already present, nothing to do', tif_path)
            written.append(str(tif_path))
            continue

        try:
            with gzip.open(gz_path, 'rb') as src:
                with open(tif_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
        except gzip.BadGzipFile as exc:
            # copyfileobj can fail partway through, after tif_path was
            # already created, so a half-written file is removed here
            # rather than left on disk looking like a finished one.
            tif_path.unlink(missing_ok=True)
            log.warning('could not decompress %s: %s', gz_path, exc)
            skipped.append(str(gz_path))
            continue

        log.info('wrote %s', tif_path)
        written.append(str(tif_path))

    log.info(
        '%s of %s file(s) decompressed into %s, %s skipped',
        len(written), len(gz_paths), output_dir, len(skipped),
    )
    return written


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    unzip_chirps_rasters(
        source_dir=r'C:\Users\osorobo\Downloads',
        output_dir=r'C:\Users\osorobo\Downloads\chirps_tif',
    )
