"""Utilities for fetching gridMET temperature data and computing cold degree-days (CDD).

CDD accumulates daily heat deficit below 5°C using daily mean temperature:
    CDD_day = max(0, 5 − (Tmax + Tmin) / 2)
Accumulation starts August 1 each year; CDD = 0 for any date before August 1.

Historical yearly CDD increments are stored locally as gridmet_cdd_{year}.npz.
Current-year accumulated CDD is stored as cdd_state_{year}.npz and committed to GitHub.
"""

import datetime
import os
import numpy as np
import rasterio
from rasterio.transform import Affine

CDD_THRESH_C = 5.0
KELVIN_OFFSET = 273.15


def _aug1_doy(year):
    return datetime.date(year, 8, 1).timetuple().tm_yday


def _doy_to_date(year, doy):
    return datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)


def _parse_transform(transform_arr):
    return Affine(transform_arr[0], transform_arr[1], transform_arr[2],
                  transform_arr[3], transform_arr[4], transform_arr[5])


def _compute_tmean_c(tmmn_k, tmmx_k):
    """Daily mean temperature in °C from Kelvin min/max arrays."""
    return (tmmn_k + tmmx_k) / 2.0 - KELVIN_OFFSET


def _compute_cdd_increments(tmmn_k, tmmx_k):
    """Daily CDD increment: max(0, 5 − (Tmax + Tmin)/2) in Celsius.

    Args:
        tmmn_k: Array of minimum temperatures in Kelvin.
        tmmx_k: Array of maximum temperatures in Kelvin.

    Returns:
        Array of non-negative CDD increments (same shape as inputs).
    """
    tmean_c = _compute_tmean_c(tmmn_k, tmmx_k)
    return np.where(np.isfinite(tmean_c), np.maximum(0.0, CDD_THRESH_C - tmean_c), 0.0)


def _download_gridmet_range(start_date, end_date, route_buffer, output_dir, tag):
    """Download gridMET Tmax+Tmin images for [start_date, end_date) from GEE.

    Args:
        start_date: datetime.date for filter start (inclusive).
        end_date: datetime.date for filter end (exclusive).
        route_buffer: GEE Geometry for the download region.
        output_dir: Directory for temporary files.
        tag: String tag used in temp filenames (e.g., year or 'upd').

    Returns:
        Dict with keys doys (int32 1D), tmmn_stack, tmmx_stack ((n,h,w) float32),
        transform (6-element float64), crs_wkt (str).  None if no images available.
    """
    import ee
    import geemap

    coll = (
        ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
        .filterBounds(route_buffer)
        .filterDate(start_date.isoformat(),
                    (end_date + datetime.timedelta(days=1)).isoformat())
        .select(['tmmn', 'tmmx'])
        .sort('system:time_start')
    )

    timestamps = coll.aggregate_array('system:time_start').getInfo()
    if not timestamps:
        return None

    all_dates = [datetime.datetime.utcfromtimestamp(ts / 1000).date() for ts in timestamps]
    all_doys = [d.timetuple().tm_yday for d in all_dates]
    image_list = coll.toList(len(timestamps))

    tmmn_list, tmmx_list, valid_doys = [], [], []
    transform_arr = crs_wkt = None

    for i, doy in enumerate(all_doys):
        img = ee.Image(image_list.get(i))
        tmp_path = os.path.join(output_dir, f'_tmp_gridmet_{tag}_{i:03d}.tif')

        for attempt in range(3):
            geemap.ee_export_image(img, filename=tmp_path, scale=4638,
                                   crs='EPSG:4326',
                                   region=route_buffer, file_per_band=False)
            if os.path.exists(tmp_path):
                break
            print(f'    Attempt {attempt + 1}/3 failed, retrying...')
        else:
            print(f'    Skipping gridMET DOY {doy} after 3 failed attempts.')
            continue

        with rasterio.open(tmp_path) as src:
            data = src.read().astype(float)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            if transform_arr is None:
                t = src.transform
                transform_arr = np.array([t.a, t.b, t.c, t.d, t.e, t.f])
                crs_wkt = src.crs.to_wkt()

        os.remove(tmp_path)
        tmmn_list.append(data[0])
        tmmx_list.append(data[1])
        valid_doys.append(doy)
        print(f'    Downloaded gridMET DOY {doy}')

    if not valid_doys:
        return None

    return {
        'doys':       np.array(valid_doys, dtype=np.int32),
        'tmmn_stack': np.stack(tmmn_list).astype(np.float32),
        'tmmx_stack': np.stack(tmmx_list).astype(np.float32),
        'transform':  transform_arr,
        'crs_wkt':    crs_wkt,
    }


# ---------------------------------------------------------------------------
# Public API — training (historical)
# ---------------------------------------------------------------------------

def fetch_gridmet_cdd_historical(year, route_buffer, output_dir):
    """Download gridMET for a full historical season and save daily CDD increments.

    Stores greendown_outputs/gridmet_cdd_{year}.npz with keys:
        doys (n,), cdd_daily (n, h, w) float32, transform (6,), crs_wkt str.

    Skips download if the file already exists.

    Args:
        year: Integer year.
        route_buffer: GEE Geometry for the AT corridor.
        output_dir: Local directory for persistent storage.

    Returns:
        Path to the .npz file, or None if no data is available.
    """
    out_path = os.path.join(output_dir, f'gridmet_cdd_{year}.npz')
    if os.path.exists(out_path):
        print(f'  gridMET CDD for {year} already cached at {out_path}')
        return out_path

    print(f'  Downloading gridMET Tmax/Tmin for {year} (Aug 1 – Dec 31)...')
    aug1  = datetime.date(year, 8, 1)
    dec31 = datetime.date(year, 12, 31)
    result = _download_gridmet_range(aug1, dec31, route_buffer, output_dir, str(year))

    if result is None:
        print(f'  No gridMET data available for {year}.')
        return None

    cdd_daily  = _compute_cdd_increments(result['tmmn_stack'], result['tmmx_stack'])
    tmean_daily = _compute_tmean_c(result['tmmn_stack'], result['tmmx_stack'])
    np.savez_compressed(
        out_path,
        doys=result['doys'],
        cdd_daily=cdd_daily.astype(np.float32),
        tmean_daily=tmean_daily.astype(np.float32),
        transform=result['transform'],
        crs_wkt=np.array(str(result['crs_wkt'])),
    )
    print(f'  Saved {out_path} ({len(result["doys"])} days)')
    return out_path


def load_cdd_historical(year, output_dir):
    """Load historical gridMET CDD increments and return cumulative CDD by DOY.

    Args:
        year: Integer year.
        output_dir: Directory containing gridmet_cdd_{year}.npz.

    Returns:
        Dict with keys doys (n,), cdd_cumulative (n, h, w), transform (6-element array).
        Returns None if the file does not exist.
    """
    path = os.path.join(output_dir, f'gridmet_cdd_{year}.npz')
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    cdd_daily = data['cdd_daily'].astype(float)
    tmean_daily = (data['tmean_daily'].astype(float)
                   if 'tmean_daily' in data.files
                   else np.full_like(cdd_daily, np.nan))
    return {
        'doys':           data['doys'],
        'cdd_cumulative': np.cumsum(cdd_daily, axis=0),
        'tmean_daily':    tmean_daily,
        'transform':      data['transform'],
    }


def cdd_at_latlon(cdd_data, target_doy, year, lat_vals, lon_vals):
    """Look up accumulated CDD at (lat, lon) points for a given DOY.

    Args:
        cdd_data: Dict from load_cdd_historical, or None.
        target_doy: Target day-of-year (1–365).
        year: Calendar year (for August 1 threshold).
        lat_vals: 1D float array of WGS84 latitudes.
        lon_vals: 1D float array of WGS84 longitudes.

    Returns:
        1D float array of accumulated CDD values.
        Returns 0 for dates before August 1; NaN if historical data not available.
    """
    n = len(lat_vals)
    if target_doy < _aug1_doy(year):
        return np.zeros(n)
    if cdd_data is None:
        return np.full(n, np.nan)

    doys = cdd_data['doys']
    cdd_cum = cdd_data['cdd_cumulative']  # (n_days, h, w)
    t = cdd_data['transform']

    h_gmet, w_gmet = cdd_cum.shape[1], cdd_cum.shape[2]
    gmet_cols = np.clip(((lon_vals - t[2]) / t[0]).astype(int), 0, w_gmet - 1)
    gmet_rows = np.clip(((lat_vals - t[5]) / t[4]).astype(int), 0, h_gmet - 1)

    idx = int(np.searchsorted(doys, target_doy, side='right')) - 1
    if idx < 0:
        return np.zeros(n)

    return cdd_cum[idx, gmet_rows, gmet_cols].astype(float)


def tmean_at_latlon(cdd_data, target_doy, lat_vals, lon_vals):
    """Look up daily mean temperature (°C) at (lat, lon) points for a given DOY.

    Args:
        cdd_data: Dict from load_cdd_historical, or None.
        target_doy: Target day-of-year (1–365).
        lat_vals: 1D float array of WGS84 latitudes.
        lon_vals: 1D float array of WGS84 longitudes.

    Returns:
        1D float array of T_mean values in °C.  NaN if data not available.
    """
    n = len(lat_vals)
    if cdd_data is None or 'tmean_daily' not in cdd_data:
        return np.full(n, np.nan)

    doys        = cdd_data['doys']
    tmean_daily = cdd_data['tmean_daily']  # (n_days, h, w)
    t           = cdd_data['transform']

    h_gmet, w_gmet = tmean_daily.shape[1], tmean_daily.shape[2]
    gmet_cols = np.clip(((lon_vals - t[2]) / t[0]).astype(int), 0, w_gmet - 1)
    gmet_rows = np.clip(((lat_vals - t[5]) / t[4]).astype(int), 0, h_gmet - 1)

    idx = int(np.searchsorted(doys, target_doy, side='right')) - 1
    if idx < 0:
        return np.full(n, np.nan)

    return tmean_daily[idx, gmet_rows, gmet_cols].astype(float)


# ---------------------------------------------------------------------------
# Public API — production (current year, GitHub-stored state)
# ---------------------------------------------------------------------------

def update_cdd_state(year, route_buffer, output_dir):
    """Incrementally update the current-year accumulated CDD state file.

    Loads cdd_state_{year}.npz, downloads any new gridMET data not yet included
    (DOY > last_doy), adds it to the accumulator, and saves back.  Running this
    multiple times per day is safe — only new dates are fetched.

    Before August 1 the function returns immediately; CDD = 0 before accumulation starts.

    Args:
        year: Integer year.
        route_buffer: GEE Geometry for the AT corridor.
        output_dir: Directory to read/write cdd_state_{year}.npz.

    Returns:
        Path to cdd_state_{year}.npz (file may not yet exist if before Aug 1).
    """
    state_path = os.path.join(output_dir, f'cdd_state_{year}.npz')
    aug1_doy   = _aug1_doy(year)
    today      = datetime.date.today()
    today_doy  = today.timetuple().tm_yday

    if today_doy < aug1_doy:
        print(f'  Before Aug 1; CDD accumulation not yet started for {year}.')
        return state_path

    # Load existing state (if any)
    last_doy  = aug1_doy - 1
    cdd_acc   = None
    tmean_last = None
    transform_arr = crs_wkt = None

    if os.path.exists(state_path):
        state = np.load(state_path, allow_pickle=True)
        last_doy      = int(state['last_doy'])
        cdd_acc       = state['cdd_acc'].copy().astype(float)
        tmean_last    = (state['tmean_last'].copy().astype(float)
                         if 'tmean_last' in state.files else None)
        transform_arr = state['transform'].copy()
        crs_wkt       = str(state['crs_wkt'])
        print(f'  CDD state loaded: last_doy={last_doy}')

    yesterday     = today - datetime.timedelta(days=1)
    yesterday_doy = yesterday.timetuple().tm_yday

    if last_doy >= yesterday_doy:
        print(f'  CDD state is current through DOY {last_doy}.')
        return state_path

    start_date = _doy_to_date(year, max(aug1_doy, last_doy + 1))
    end_date   = min(yesterday, datetime.date(year, 12, 31))

    print(f'  Downloading new gridMET data ({start_date} – {end_date})...')
    result = _download_gridmet_range(start_date, end_date, route_buffer, output_dir, 'upd')

    if result is None:
        print('  No new gridMET data available yet.')
        return state_path

    if cdd_acc is None:
        h, w = result['tmmn_stack'].shape[1], result['tmmn_stack'].shape[2]
        cdd_acc       = np.zeros((h, w), dtype=float)
        transform_arr = result['transform']
        crs_wkt       = result['crs_wkt']

    cdd_increments  = _compute_cdd_increments(result['tmmn_stack'], result['tmmx_stack'])
    tmean_stack     = _compute_tmean_c(result['tmmn_stack'], result['tmmx_stack'])
    for i, doy in enumerate(result['doys']):
        cdd_acc   += cdd_increments[i]
        tmean_last = tmean_stack[i]   # keep only the most recent day's T_mean
        last_doy   = max(last_doy, int(doy))
        print(f'    Accumulated CDD for DOY {doy}')

    np.savez_compressed(
        state_path,
        cdd_acc=cdd_acc.astype(np.float32),
        tmean_last=(tmean_last.astype(np.float32)
                    if tmean_last is not None
                    else np.full(cdd_acc.shape, np.nan, dtype=np.float32)),
        last_doy=np.int32(last_doy),
        transform=transform_arr,
        crs_wkt=np.array(str(crs_wkt)),
    )
    print(f'  Saved CDD state → {state_path} (last_doy={last_doy})')
    return state_path


def tmean_from_state(state_path, lat_vals, lon_vals):
    """Look up the most recent daily mean temperature (°C) from the current-year state file.

    Returns the T_mean grid for the last day that was processed (last_doy).

    Args:
        state_path: Path to cdd_state_{year}.npz.
        lat_vals: 1D float array of WGS84 latitudes.
        lon_vals: 1D float array of WGS84 longitudes.

    Returns:
        1D float array of T_mean values in °C.  NaN if state file or field absent.
    """
    n = len(lat_vals)
    if not os.path.exists(state_path):
        return np.full(n, np.nan)

    state = np.load(state_path, allow_pickle=True)
    if 'tmean_last' not in state.files:
        return np.full(n, np.nan)

    tmean_last = state['tmean_last'].astype(float)
    t = state['transform']

    h_gmet, w_gmet = tmean_last.shape
    gmet_cols = np.clip(((lon_vals - t[2]) / t[0]).astype(int), 0, w_gmet - 1)
    gmet_rows = np.clip(((lat_vals - t[5]) / t[4]).astype(int), 0, h_gmet - 1)

    return tmean_last[gmet_rows, gmet_cols]


def cdd_from_state(state_path, year, target_doy, lat_vals, lon_vals):
    """Look up accumulated CDD from the current-year state file.

    Args:
        state_path: Path to cdd_state_{year}.npz.
        year: Calendar year.
        target_doy: Target day-of-year.
        lat_vals: 1D float array of WGS84 latitudes.
        lon_vals: 1D float array of WGS84 longitudes.

    Returns:
        1D float array of accumulated CDD values.
        Returns 0 for dates before August 1 or if no state file exists yet.
    """
    n = len(lat_vals)
    if target_doy < _aug1_doy(year):
        return np.zeros(n)
    if not os.path.exists(state_path):
        return np.zeros(n)

    state = np.load(state_path, allow_pickle=True)
    cdd_acc = state['cdd_acc'].astype(float)
    t = state['transform']

    h_gmet, w_gmet = cdd_acc.shape
    gmet_cols = np.clip(((lon_vals - t[2]) / t[0]).astype(int), 0, w_gmet - 1)
    gmet_rows = np.clip(((lat_vals - t[5]) / t[4]).astype(int), 0, h_gmet - 1)

    return cdd_acc[gmet_rows, gmet_cols]
