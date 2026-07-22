"""
Combine yearly parasite-rate files into a single CSV.

Reads every year-named file (e.g. 2010.xlsx, 2011.xlsx, ...) in the
parasite_rate folder, adds a `year` column from the filename, renames
columns, drops unwanted columns, and writes parasite_rate_combined.csv
"""

import os
import glob
import re
import pandas as pd

# ── Folder containing the yearly files ──────────────────────────────────────
FOLDER = r'C:\GitHub\dream\data\raw\ZMB\parasite_rate'

# ── Find every year-named file (xlsx or csv) ────────────────────────────────
files = glob.glob(os.path.join(FOLDER, '*.xlsx')) + \
        glob.glob(os.path.join(FOLDER, '*.csv'))

frames = []

for path in sorted(files):
    fname = os.path.basename(path)

    # skip the output file if it already exists in the folder
    if fname.startswith('parasite_rate_combined'):
        continue

    # extract the 4-digit year from the filename
    match = re.search(r'(\d{4})', fname)
    if not match:
        print(f'Skipping (no year in name): {fname}')
        continue
    year = int(match.group(1))

    # read the file (xlsx or csv)
    if path.lower().endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # add the year column
    df['year'] = year

    # rename the required columns
    df = df.rename(columns={
        'POINT_X':   'longitude',
        'POINT_Y':   'latitude',
        'grid_code': 'value'
    })

    # drop unwanted columns (ignore if already absent)
    df = df.drop(columns=['OBJECTID', 'pointid'], errors='ignore')

    frames.append(df)
    print(f'{fname}: {len(df):>7d} rows  (year={year})')

# ── Combine and save ────────────────────────────────────────────────────────
combined = pd.concat(frames, ignore_index=True)

# order columns sensibly
cols = ['year', 'longitude', 'latitude', 'value']
combined = combined[[c for c in cols if c in combined.columns]]
combined['metric'] = 'parasite_rate'

out_path = os.path.join(FOLDER, 'parasite_rate_combined.csv')
combined.to_csv(out_path, index=False)

print(f'\nCombined {len(frames)} files -> {len(combined):,} rows')
print(f'Saved: {out_path}')
