import datetime
import glob
import math
import os
import re
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import xy

from filter_ci_widths import load_ci_widths
from constants import NODATA, MAX_CI_WIDTH, CROSS_YEAR_MAX_CI_WIDTH
from gridmet_utils import load_cdd_historical, cdd_at_latlon, tmean_at_latlon

_LABEL_ENC = {'before': 0, 'early': 1, 'late': 2, 'after': 3}


def _build_cross_year_transition_lookup(greendown_dir):
    """Load transition point estimates for all years, filtered by CI width.

    Args:
        greendown_dir: Path to directory containing transition and CI width GeoTIFFs.

    Returns:
        Dict of {year: {phase: array(h,w)}} with DOY floats. Values are NaN
        for pixels where any CI width exceeds CROSS_YEAR_MAX_CI_WIDTH, or where
        required GeoTIFFs are missing.
    """
    phases = ('start', 'middle', 'end')
    pattern = re.compile(r'greendown_start_(\d{4})\.tif')
    available_years = sorted(
        int(m.group(1))
        for p in glob.glob(os.path.join(greendown_dir, 'greendown_start_*.tif'))
        if (m := pattern.search(os.path.basename(p)))
    )

    lookup = {}
    for year in available_years:
        required = (
            [os.path.join(greendown_dir, f'greendown_{p}_{year}.tif') for p in phases] +
            [os.path.join(greendown_dir, f'greendown_{p}_ci_width_{year}.tif') for p in phases]
        )
        if not all(os.path.exists(f) for f in required):
            continue

        # Build quality mask: all three CI widths must be finite and <= threshold
        ci_mask = None
        for phase in phases:
            with rasterio.open(os.path.join(greendown_dir, f'greendown_{phase}_ci_width_{year}.tif')) as src:
                w = src.read(1).astype(float)
                w[w == NODATA] = np.nan
            phase_ok = np.isfinite(w) & (w <= CROSS_YEAR_MAX_CI_WIDTH)
            ci_mask = phase_ok if ci_mask is None else (ci_mask & phase_ok)

        year_data = {}
        for phase in phases:
            with rasterio.open(os.path.join(greendown_dir, f'greendown_{phase}_{year}.tif')) as src:
                d = src.read(1).astype(float)
                d[d == NODATA] = np.nan
            d[~ci_mask] = np.nan
            year_data[phase] = d

        lookup[year] = year_data

    return lookup


def _load_point_estimates(greendown_dir, year):
    """Load start/middle/end greendown transition point estimate arrays for one year.

    Args:
        greendown_dir: Path to directory containing transition GeoTIFFs.
        year: Integer year to load.

    Returns:
        Dict of {'start': array, 'middle': array, 'end': array} with NaN for nodata.
    """
    points = {}
    for phase in ('start', 'middle', 'end'):
        path = os.path.join(greendown_dir, f'greendown_{phase}_{year}.tif')
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)
            data[data == NODATA] = np.nan
            points[phase] = data
    return points


def _load_lat_lon_arrays(data_dir, year):
    """Build 2D arrays of WGS84 latitudes and longitudes for every pixel.

    Derives coordinates from the spatial transform and CRS of the reference GeoTIFF.

    Args:
        data_dir: Path to directory containing the reference GeoTIFF.
        year: Integer year whose reference raster is used.

    Returns:
        Tuple (lat_array, lon_array), each shape (h, w), WGS84 decimal degrees.
    """
    ref_path = os.path.join(data_dir, f'hls_indices_ref_{year}.tif')
    if not os.path.exists(ref_path):
        ref_path = os.path.join(data_dir, 'hls_indices_ref_current.tif')
    with rasterio.open(ref_path) as src:
        h, w      = src.height, src.width
        transform = src.transform
        src_crs   = src.crs

    rows_idx, cols_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    xs, ys = xy(transform, rows_idx.ravel(), cols_idx.ravel())

    transformer = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)
    lons, lats = transformer.transform(xs, ys)

    return np.array(lats).reshape(h, w), np.array(lons).reshape(h, w)


def _load_lat_array(data_dir, year):
    """Build a 2D array of WGS84 latitudes for every pixel.

    Args:
        data_dir: Path to directory containing the reference GeoTIFF.
        year: Integer year whose reference raster is used.

    Returns:
        Array of shape (h, w) with WGS84 latitude values.
    """
    lat_array, _ = _load_lat_lon_arrays(data_dir, year)
    return lat_array


def _day_length(doy, lat_deg):
    """Compute day length in hours using an astronomical solar geometry formula.

    Args:
        doy: Day of year (1–365).
        lat_deg: Latitude in decimal degrees.

    Returns:
        Day length in hours.
    """
    lat  = math.radians(lat_deg)
    decl = math.radians(-23.45 * math.cos(math.radians(360 / 365 * (doy + 10))))
    arg  = max(-1.0, min(1.0, -math.tan(lat) * math.tan(decl)))
    return 2 * math.degrees(math.acos(arg)) / 15


def _assign_label(doy, start, middle, end):
    """Assign a phenological label based on DOY relative to transition point estimates.

    Args:
        doy: Day of year for the observation.
        start: DOY of the start of greendown.
        middle: DOY of the middle of greendown.
        end: DOY of the end of greendown.

    Returns:
        One of 'before', 'early', 'late', or 'after'.
    """
    if doy < start:
        return 'before'
    elif doy < middle:
        return 'early'
    elif doy < end:
        return 'late'
    else:
        return 'after'


def _add_mode_label_7day(df):
    """Add a mode_label_7day column: mode label derived from the fitted curve over [doy-7, doy-1].

    Because HLS revisit is ~8 days, actual prior observations within a 7-day window are
    rare. Instead, labels for each of the 7 prior days are computed analytically from the
    fitted transition dates (start, middle, end) using _assign_label, then the mode is
    taken. This produces a non-NaN value for every row except the very start of the season
    before any transition date data exists.

    Args:
        df: DataFrame with columns year, doy, pixel_id, label, and the transition date
            columns doy_minus_avg_start, doy_minus_avg_middle, doy_minus_avg_end plus
            the per-year point estimates embedded via start/middle/end columns.

    Returns:
        DataFrame with an additional mode_label_7day column (float ordinal, NaN-able).
    """
    mode_map = {}
    for (_, _), grp in df.groupby(['year', 'pixel_id'], sort=False):
        grp_sorted = grp.sort_values('doy')
        doys   = grp_sorted['doy'].values
        starts = grp_sorted['transition_start'].values
        middles = grp_sorted['transition_middle'].values
        ends   = grp_sorted['transition_end'].values
        for i, idx in enumerate(grp_sorted.index):
            d = int(doys[i])
            s, m, e = starts[i], middles[i], ends[i]
            prior_labels = [
                _LABEL_ENC[_assign_label(pd, s, m, e)]
                for pd in range(d - 7, d)
            ]
            counts = np.bincount(prior_labels, minlength=4)
            mode_map[idx] = float(np.argmax(counts))
    df = df.copy()
    df['mode_label_7day'] = pd.Series(mode_map)
    return df


def build_feature_table(data_dir, greendown_dir, years, max_width=MAX_CI_WIDTH,
                        retain_pixel_id=False, include_temperature=False,
                        spatial_mask=None):
    """Build a labeled feature table from EVI/NDVI time series for qualifying pixel-years.

    Filters to pixel-years where all three CI widths are < max_width days, then labels
    each observation relative to the greendown transition point estimates.

    Args:
        data_dir: Path to directory containing stacks, DOY arrays, and reference GeoTIFFs.
        greendown_dir: Path to directory containing transition and CI width GeoTIFFs.
        years: List of integer years to process.
        max_width: Maximum CI width in days for a pixel-year to qualify.
        retain_pixel_id: If True, keep the pixel_id column in the returned DataFrame
            (needed for RNN sequence construction). Default False (DT pipeline).
        include_temperature: If True, compute and include cdd_accumulated and
            tmean_recent columns (from gridMET). Default False (DT pipeline).
        spatial_mask: Optional boolean numpy array of shape (h, w) matching the CI-width
            GeoTIFF grid. When provided, only pixels where spatial_mask is True are
            included. Default None (all CI-qualifying pixels included).

    Returns:
        DataFrame with columns: year, date, doy, [pixel_id], EVI, NDVI, evi_delta,
        evi_delta2, ndvi_delta, ndvi_delta2, day_length_hrs, doy_minus_avg_start,
        doy_minus_avg_middle, doy_minus_avg_end, [cdd_accumulated, tmean_recent], label.
    """
    phases = ('start', 'middle', 'end')
    widths = load_ci_widths(greendown_dir, years)
    cross_year_lookup = _build_cross_year_transition_lookup(greendown_dir)
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

        if spatial_mask is not None:
            mask &= spatial_mask

        if not mask.any():
            print(f'  {year}: no qualifying pixels')
            continue

        # Load EVI/NDVI stack (n_images, n_bands, h, w): band 0=EVI, band 1=NDVI
        indices_stack = np.load(os.path.join(data_dir, f'hls_indices_stack_{year}.npy'))
        doys          = np.load(os.path.join(data_dir, f'hls_indices_doys_{year}.npy'))
        points        = _load_point_estimates(greendown_dir, year)
        lat_array, lon_array = _load_lat_lon_arrays(data_dir, year)
        if include_temperature:
            cdd_hist = load_cdd_historical(year, data_dir)

        pixel_rows, pixel_cols = np.where(mask)
        print(f'  {year}: {len(pixel_rows)} qualifying pixels')

        sorted_t = np.argsort(doys)  # chronological order

        for r, c in zip(pixel_rows, pixel_cols):
            start  = points['start'][r, c]
            middle = points['middle'][r, c]
            end    = points['end'][r, c]

            if not (np.isfinite(start) and np.isfinite(middle) and np.isfinite(end)):
                continue

            # Cross-year average transition DOYs (other years with CI width <= threshold).
            # Falls back to the current year's own value when no other years are available
            # (e.g. pixel passes CI filter in only one year).
            avg_doys = {}
            for phase in phases:
                vals = [
                    float(cross_year_lookup[yr][phase][r, c])
                    for yr in cross_year_lookup
                    if yr != year and np.isfinite(cross_year_lookup[yr][phase][r, c])
                ]
                if vals:
                    avg_doys[phase] = float(np.mean(vals))
                elif np.isfinite(points[phase][r, c]):
                    avg_doys[phase] = float(points[phase][r, c])
                else:
                    avg_doys[phase] = float('nan')

            lat = lat_array[r, c]
            lon = lon_array[r, c]
            prev_evi   = None
            prev2_evi  = None
            prev_ndvi  = None
            prev2_ndvi = None
            for t in sorted_t:
                doy  = doys[t]
                evi  = indices_stack[t, 0, r, c]
                ndvi = indices_stack[t, 1, r, c]
                if not np.isfinite(evi) or evi <= 0:
                    continue
                if not np.isfinite(ndvi):
                    continue
                evi_delta   = (float(evi)  - prev_evi)   if prev_evi   is not None else float('nan')
                evi_delta2  = (float(evi)  - prev2_evi)  if prev2_evi  is not None else float('nan')
                ndvi_delta  = (float(ndvi) - prev_ndvi)  if prev_ndvi  is not None else float('nan')
                ndvi_delta2 = (float(ndvi) - prev2_ndvi) if prev2_ndvi is not None else float('nan')
                prev2_evi   = prev_evi
                prev_evi    = float(evi)
                prev2_ndvi  = prev_ndvi
                prev_ndvi   = float(ndvi)
                date = (datetime.date(year, 1, 1) + datetime.timedelta(days=int(doy) - 1)).isoformat()
                # Sample CDD and daily mean temperature for the PREVIOUS day (doy - 1)
                # to match serving, where gridMET lags the prediction date by ~1-2 days.
                if include_temperature:
                    prev_doy = int(doy) - 1
                    cdd   = float(cdd_at_latlon(cdd_hist, prev_doy, year,
                                                np.array([lat]), np.array([lon]))[0])
                    tmean = float(tmean_at_latlon(cdd_hist, prev_doy,
                                                  np.array([lat]), np.array([lon]))[0])
                row = {
                    'year':                      year,
                    'date':                      date,
                    'doy':                       int(doy),
                    'pixel_id':                  f'{r}_{c}',
                    'EVI':                       float(evi),
                    'NDVI':                      float(ndvi),
                    'evi_delta':                 evi_delta,
                    'evi_delta2':                evi_delta2,
                    'ndvi_delta':                ndvi_delta,
                    'ndvi_delta2':               ndvi_delta2,
                    'day_length_hrs':            _day_length(int(doy), lat),
                    'doy_minus_avg_start':  int(doy) - avg_doys['start'],
                    'doy_minus_avg_middle': int(doy) - avg_doys['middle'],
                    'doy_minus_avg_end':    int(doy) - avg_doys['end'],
                    'transition_start':          float(start),
                    'transition_middle':         float(middle),
                    'transition_end':            float(end),
                    'ci_width_start':            float(year_widths['start'][r, c]),
                    'ci_width_middle':           float(year_widths['middle'][r, c]),
                    'ci_width_end':              float(year_widths['end'][r, c]),
                    'label':                     _assign_label(doy, start, middle, end),
                }
                if include_temperature:
                    row['cdd_accumulated'] = cdd
                    row['tmean_recent']    = tmean
                rows.append(row)

    base_cols = [
        'year', 'date', 'doy', 'pixel_id', 'EVI', 'NDVI',
        'evi_delta', 'evi_delta2', 'ndvi_delta', 'ndvi_delta2',
        'day_length_hrs',
        'doy_minus_avg_start', 'doy_minus_avg_middle', 'doy_minus_avg_end',
        'transition_start', 'transition_middle', 'transition_end',
        'ci_width_start', 'ci_width_middle', 'ci_width_end',
        'label',
    ]
    temp_cols = ['cdd_accumulated', 'tmean_recent'] if include_temperature else []
    df = pd.DataFrame(rows, columns=base_cols[:-1] + temp_cols + ['label'])
    print(f'\nTotal labeled phenology observations: {len(df)}')
    df = _add_mode_label_7day(df)
    # When retain_pixel_id=True (RNN path), keep transition_*, ci_width_*, and pixel_id
    # so build_rnn_sequences can compute soft labels. For the DT path, drop everything.
    if not retain_pixel_id:
        df = df.drop(columns=[
            'pixel_id',
            'transition_start', 'transition_middle', 'transition_end',
            'ci_width_start', 'ci_width_middle', 'ci_width_end',
        ])
    return df


def export_prediction_avg_assets(data_dir, greendown_dir):
    """Write the committed assets the live prediction needs for doy_minus_avg_middle.

    The automated Action does not have the per-year transition GeoTIFFs (they are
    gitignored), so it cannot rebuild the CI-filtered cross-year average or the
    global gap-fill scalar on the fly — the values the model was trained on.
    This precomputes both from the local per-year tifs and writes them as small
    committed files:

      - greendown_middle_avg_filtered.tif: per-pixel CI-filtered cross-year mean
        middle-transition DOY (matches the per-pixel average in build_feature_table).
      - greendown_avg_meta.json: {"global_avg_middle": <scalar>}, the gap-fill
        value used for pixels with no confident per-pixel estimate.

    Run this whenever the per-year transition GeoTIFFs are regenerated (i.e. after
    re-fitting greendown curves), then commit the two output files.

    Args:
        data_dir: Path to Data/ containing reference GeoTIFFs for CRS lookup.
        greendown_dir: Path to Greendown_Outputs/ containing per-year transition
            and CI-width GeoTIFFs, and where output files are written.
    """
    import json
    import warnings
    from edit_data_table import _compute_global_avg_middle
    from fit_greendown_curves import _canonical_crs

    lookup = _build_cross_year_transition_lookup(greendown_dir)
    if not lookup:
        print('  No per-year transition tifs found; cannot export prediction avg assets.')
        return

    arrs = [lookup[yr]['middle'] for yr in lookup]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # all-NaN slices
        avg = np.nanmean(np.stack(arrs), axis=0).astype(np.float32)

    ref_path = os.path.join(greendown_dir, 'greendown_middle_avg.tif')
    with rasterio.open(ref_path) as src:
        profile = src.profile
    # Use the authoritative CRS from the reference raster (the avg tifs have
    # historically carried a mislabeled CRS).
    ref_crs = _canonical_crs(data_dir)
    if ref_crs is not None:
        profile.update(crs=ref_crs)
    out = avg.copy()
    out[np.isnan(out)] = NODATA
    tif_path = os.path.join(greendown_dir, 'greendown_middle_avg_filtered.tif')
    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(out, 1)

    global_avg = _compute_global_avg_middle(greendown_dir)
    meta_path = os.path.join(greendown_dir, 'greendown_avg_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({'global_avg_middle': float(global_avg)}, f, indent=2)

    n_valid = int(np.isfinite(avg).sum())
    print(f'  Wrote {tif_path} ({n_valid} valid pixels)')
    print(f'  Wrote {meta_path} (global_avg_middle = {global_avg:.2f})')
