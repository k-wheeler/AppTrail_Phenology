import os
import numpy as np
import pandas as pd
import rasterio

from filter_ci_widths import load_ci_widths, MAX_CI_WIDTH

NODATA = -9999.0


def _load_point_estimates(output_dir, year):
    """Load start/middle/end point estimate arrays for one year."""
    points = {}
    for phase in ('start', 'middle', 'end'):
        path = os.path.join(output_dir, f'greendown_{phase}_{year}.tif')
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)
            data[data == NODATA] = np.nan
            points[phase] = data
    return points


def _assign_label(doy, start, middle, end):
    """Assign phenological label based on DOY relative to transition point estimates."""
    if doy < start:
        return 'before'
    elif doy < middle:
        return 'early'
    elif doy < end:
        return 'late'
    else:
        return 'after'


def build_labeled_evi_table(output_dir, years, max_width=MAX_CI_WIDTH):
    """
    For pixel-years where all three CI widths are < max_width days, load the
    EVI time series and label each observation relative to the greendown
    transition point estimates (start, middle, end).

    Returns a DataFrame with columns: year, EVI, label.
    """
    phases = ('start', 'middle', 'end')
    widths = load_ci_widths(output_dir, years)
    rows = []

    for year in years:
        year_widths = widths[year]
        if any(year_widths[p] is None for p in phases):
            print(f'  {year}: skipped (missing CI width GeoTIFF)')
            continue

        # Build mask: pixels with valid CI width < max_width for all phases
        mask = np.ones(year_widths['start'].shape, dtype=bool)
        for phase in phases:
            arr = year_widths[phase]
            mask &= np.isfinite(arr) & (arr < max_width)

        if not mask.any():
            print(f'  {year}: no qualifying pixels')
            continue

        # Load EVI stack and DOYs
        evi_stack = np.load(os.path.join(output_dir, f'hls_evi_stack_{year}.npy'))
        doys      = np.load(os.path.join(output_dir, f'hls_evi_doys_{year}.npy'))
        points    = _load_point_estimates(output_dir, year)

        pixel_rows, pixel_cols = np.where(mask)
        print(f'  {year}: {len(pixel_rows)} qualifying pixels')

        for r, c in zip(pixel_rows, pixel_cols):
            start  = points['start'][r, c]
            middle = points['middle'][r, c]
            end    = points['end'][r, c]

            if not (np.isfinite(start) and np.isfinite(middle) and np.isfinite(end)):
                continue

            for t, doy in enumerate(doys):
                evi = evi_stack[t, r, c]
                if not np.isfinite(evi) or evi <= 0:
                    continue
                rows.append({
                    'year':  year,
                    'EVI':   float(evi),
                    'label': _assign_label(doy, start, middle, end),
                })

    df = pd.DataFrame(rows, columns=['year', 'EVI', 'label'])
    print(f'\nTotal labeled EVI observations: {len(df)}')
    return df
