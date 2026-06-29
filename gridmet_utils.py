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

# Historical gridMET is fetched from this date through Dec 31 so that every
# training observation (and its previous day) has a real daily mean temperature.
# CDD still only accumulates from Aug 1; earlier increments are zeroed.
HIST_FETCH_START_MONTH = 5
HIST_FETCH_START_DAY   = 25

# Current-year (serving) gridMET is seeded from the season start — the HLS
# collection start (June 1) — so the cdd_state T_mean series covers every
# observation the RNN may hold in its multi-month window, matching training.
SEASON_START_MONTH = 6
SEASON_START_DAY   = 1


def _aug1_doy(year):
    return datetime.date(year, 8, 1).timetuple().tm_yday


def _season_start_doy(year):
    return datetime.date(year, SEASON_START_MONTH, SEASON_START_DAY).timetuple().tm_yday


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


def _download_gridmet_range(start_date, end_date, route_buffer, data_dir, tag):
    """Download gridMET Tmax+Tmin images for [start_date, end_date) from GEE.

    Args:
        start_date: datetime.date for filter start (inclusive).
        end_date: datetime.date for filter end (exclusive).
        route_buffer: GEE Geometry for the download region.
        data_dir: Directory for temporary files.
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
        tmp_path = os.path.join(data_dir, f'_tmp_gridmet_{tag}_{i:03d}.tif')

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

def _cdd_from_result(result, aug1_doy):
    """Daily CDD increments from a download result, zeroed before Aug 1."""
    cdd_daily = _compute_cdd_increments(result['tmmn_stack'], result['tmmx_stack'])
    cdd_daily[result['doys'] < aug1_doy] = 0.0
    return cdd_daily


def fetch_gridmet_cdd_historical(year, route_buffer, data_dir):
    """Download gridMET for a full historical season and save daily CDD increments.

    Stores greendown_outputs/gridmet_cdd_{year}.npz with keys:
        doys (n,), cdd_daily (n, h, w) float32, tmean_daily (n, h, w) float32,
        transform (6,), crs_wkt str.

    Data is fetched from HIST_FETCH_START_MONTH/DAY through Dec 31 so that every
    training observation has a real previous-day temperature.  CDD increments
    before Aug 1 are zeroed, so accumulated CDD still starts at Aug 1.

    If a cached file exists but does not cover the full early window (e.g. an
    older Aug 1-only download), only the missing early gap is fetched and merged
    with the cached data — the cached Aug–Dec days are not re-downloaded.

    Args:
        year: Integer year.
        route_buffer: GEE Geometry for the AT corridor.
        data_dir: Local directory for persistent storage.

    Returns:
        Path to the .npz file, or None if no data is available.
    """
    out_path  = os.path.join(data_dir, f'gridmet_cdd_{year}.npz')
    aug1_doy  = _aug1_doy(year)
    start     = datetime.date(year, HIST_FETCH_START_MONTH, HIST_FETCH_START_DAY)
    start_doy = start.timetuple().tm_yday
    dec31     = datetime.date(year, 12, 31)

    cached = None
    if os.path.exists(out_path):
        c = np.load(out_path, allow_pickle=True)
        full_coverage = ('tmean_daily' in c.files
                         and int(c['doys'].min()) <= start_doy)
        if full_coverage:
            print(f'  gridMET CDD for {year} already cached at {out_path}')
            return out_path
        if 'tmean_daily' in c.files and 'cdd_daily' in c.files:
            cached = c  # partial (likely Aug 1-only); extend the early gap below

    if cached is not None:
        gap_end = _doy_to_date(year, int(cached['doys'].min()) - 1)
        print(f'  Extending {year} gridMET coverage ({start} – {gap_end})...')
        result = _download_gridmet_range(start, gap_end, route_buffer, data_dir, str(year))
        if result is None:
            print(f'  No earlier gridMET data available for {year}; keeping cache.')
            return out_path

        new_cdd   = _cdd_from_result(result, aug1_doy)
        new_tmean = _compute_tmean_c(result['tmmn_stack'], result['tmmx_stack'])
        cached_cdd, cached_tmean = cached['cdd_daily'], cached['tmean_daily']
        if new_cdd.shape[1:] != cached_cdd.shape[1:]:
            print('  Grid mismatch between gap and cache; re-downloading full year.')
            cached = None
        else:
            doys        = np.concatenate([result['doys'], cached['doys']])
            cdd_daily   = np.concatenate([new_cdd,   cached_cdd],   axis=0)
            tmean_daily = np.concatenate([new_tmean, cached_tmean], axis=0)
            transform   = cached['transform']
            crs_wkt     = str(cached['crs_wkt'])

    if cached is None:
        print(f'  Downloading gridMET Tmax/Tmin for {year} ({start} – Dec 31)...')
        result = _download_gridmet_range(start, dec31, route_buffer, data_dir, str(year))
        if result is None:
            print(f'  No gridMET data available for {year}.')
            return None
        doys        = result['doys']
        cdd_daily   = _cdd_from_result(result, aug1_doy)
        tmean_daily = _compute_tmean_c(result['tmmn_stack'], result['tmmx_stack'])
        transform   = result['transform']
        crs_wkt     = str(result['crs_wkt'])

    np.savez_compressed(
        out_path,
        doys=doys.astype(np.int32),
        cdd_daily=cdd_daily.astype(np.float32),
        tmean_daily=tmean_daily.astype(np.float32),
        transform=transform,
        crs_wkt=np.array(str(crs_wkt)),
    )
    print(f'  Saved {out_path} ({len(doys)} days)')
    return out_path


def load_cdd_historical(year, data_dir):
    """Load historical gridMET CDD increments and return cumulative CDD by DOY.

    Args:
        year: Integer year.
        data_dir: Directory containing gridmet_cdd_{year}.npz.

    Returns:
        Dict with keys doys (n,), cdd_cumulative (n, h, w), transform (6-element array).
        Returns None if the file does not exist.
    """
    path = os.path.join(data_dir, f'gridmet_cdd_{year}.npz')
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

# Days of CDD/T_mean history retained in the current-year cdd_state series.
# Sized to span the whole monitoring season (June 1 → year end, ~215 days) so the
# RNN can look up a real T_mean for any observation in its multi-month window,
# not just the most recent few. (The decision tree's mode feature needs only ~7
# days; the RNN sequence spans the full season.)
CDD_STATE_ROLLING_DAYS = 220


def update_cdd_state(year, route_buffer, data_dir):
    """Incrementally update the current-year CDD/T_mean state file.

    cdd_state_{year}.npz stores a per-DOY series spanning the whole monitoring
    season (seeded from the June 1 HLS collection start, up to
    CDD_STATE_ROLLING_DAYS days), so the Action can look up accumulated CDD and
    daily mean temperature for any observation in the RNN's window — not just the
    latest few:
        recent_doys (K,) int32, recent_cdd_cum (K, h, w) float32 (cumulative CDD
        through each DOY), recent_tmean (K, h, w) float32, last_doy int32,
        transform (6,), crs_wkt str.

    Downloads only gridMET days not yet included (DOY > last_doy); re-running
    multiple times per day is safe. CDD accumulation starts Aug 1 (earlier days
    contribute 0) but T_mean is recorded for the full season.

    Args:
        year: Integer year.
        route_buffer: GEE Geometry for the AT corridor.
        data_dir: Directory to read/write cdd_state_{year}.npz.

    Returns:
        Path to cdd_state_{year}.npz.
    """
    state_path = os.path.join(data_dir, f'cdd_state_{year}.npz')
    aug1_doy   = _aug1_doy(year)
    today      = datetime.date.today()
    today_doy  = today.timetuple().tm_yday
    yesterday  = today - datetime.timedelta(days=1)
    yesterday_doy = yesterday.timetuple().tm_yday

    recent_doys = []   # ascending list of int DOYs
    recent_cum  = []   # list of (h, w) cumulative-CDD grids
    recent_tm   = []   # list of (h, w) T_mean grids
    cdd_acc       = None
    transform_arr = crs_wkt = None
    last_doy      = None

    if os.path.exists(state_path):
        state         = np.load(state_path, allow_pickle=True)
        last_doy      = int(state['last_doy'])
        transform_arr = state['transform'].copy()
        crs_wkt       = str(state['crs_wkt'])
        if 'recent_doys' in state.files:
            recent_doys = [int(d) for d in state['recent_doys']]
            recent_cum  = [g.astype(float) for g in state['recent_cdd_cum']]
            recent_tm   = [g.astype(float) for g in state['recent_tmean']]
            cdd_acc     = recent_cum[-1].copy() if recent_cum else None
        else:
            # Migrate legacy single-day schema (cdd_acc / tmean_last) by seeding
            # the series with that one latest day.
            cdd_acc    = state['cdd_acc'].astype(float)
            tmean_last = (state['tmean_last'].astype(float)
                          if 'tmean_last' in state.files
                          else np.full(cdd_acc.shape, np.nan))
            recent_doys = [last_doy]
            recent_cum  = [cdd_acc.copy()]
            recent_tm   = [tmean_last]
        print(f'  CDD state loaded: last_doy={last_doy}, series len={len(recent_doys)}')
    else:
        # First run: seed from the season start (June 1, the HLS collection start)
        # so the series carries a real T_mean for every observation the RNN may
        # hold in its multi-month window. Cumulative CDD is still gated to Aug 1
        # in the accumulation loop below; June–July days are stored with CDD = 0.
        # Before the season starts there are no observations, so a short recent
        # window suffices.
        season_start_doy = _season_start_doy(year)
        if today_doy >= season_start_doy:
            last_doy = season_start_doy - 1
        else:
            last_doy = yesterday_doy - CDD_STATE_ROLLING_DAYS

    if last_doy >= yesterday_doy:
        print(f'  CDD state is current through DOY {last_doy}.')
        return state_path

    start_date = _doy_to_date(year, last_doy + 1)
    end_date   = min(yesterday, datetime.date(year, 12, 31))

    print(f'  Downloading new gridMET data ({start_date} – {end_date})...')
    result = _download_gridmet_range(start_date, end_date, route_buffer, data_dir, 'upd')

    if result is None:
        print('  No new gridMET data available yet.')
        return state_path

    if cdd_acc is None:
        h, w    = result['tmmn_stack'].shape[1], result['tmmn_stack'].shape[2]
        cdd_acc = np.zeros((h, w), dtype=float)
    if transform_arr is None:
        transform_arr = result['transform']
        crs_wkt       = result['crs_wkt']

    cdd_increments = _compute_cdd_increments(result['tmmn_stack'], result['tmmx_stack'])
    tmean_stack    = _compute_tmean_c(result['tmmn_stack'], result['tmmx_stack'])
    for i, doy in enumerate(result['doys']):
        doy = int(doy)
        if doy >= aug1_doy:
            cdd_acc += cdd_increments[i]
        recent_doys.append(doy)
        recent_cum.append(cdd_acc.copy())
        recent_tm.append(tmean_stack[i].astype(float))
        last_doy = max(last_doy, doy)

    # Keep only the most recent CDD_STATE_ROLLING_DAYS days.
    if len(recent_doys) > CDD_STATE_ROLLING_DAYS:
        recent_doys = recent_doys[-CDD_STATE_ROLLING_DAYS:]
        recent_cum  = recent_cum[-CDD_STATE_ROLLING_DAYS:]
        recent_tm   = recent_tm[-CDD_STATE_ROLLING_DAYS:]

    np.savez_compressed(
        state_path,
        recent_doys=np.array(recent_doys, dtype=np.int32),
        recent_cdd_cum=np.stack(recent_cum).astype(np.float32),
        recent_tmean=np.stack(recent_tm).astype(np.float32),
        last_doy=np.int32(last_doy),
        transform=transform_arr,
        crs_wkt=np.array(str(crs_wkt)),
    )
    print(f'  Saved CDD state → {state_path} (last_doy={last_doy}, series={len(recent_doys)})')
    return state_path


def load_cdd_state(state_path):
    """Load cdd_state_{year}.npz into a dict, normalizing legacy schemas.

    Returns None if the file does not exist. Otherwise a dict with keys:
        doys (K,) int, cdd_cum (K, h, w), tmean (K, h, w), last_doy int,
        transform (6,).  DOYs are ascending.
    """
    if not os.path.exists(state_path):
        return None
    s = np.load(state_path, allow_pickle=True)
    if 'recent_doys' in s.files:
        return {
            'doys':      np.asarray(s['recent_doys']).astype(int),
            'cdd_cum':   s['recent_cdd_cum'].astype(float),
            'tmean':     s['recent_tmean'].astype(float),
            'last_doy':  int(s['last_doy']),
            'transform': s['transform'],
        }
    # Legacy single-day schema.
    cdd_acc    = s['cdd_acc'].astype(float)
    tmean_last = (s['tmean_last'].astype(float)
                  if 'tmean_last' in s.files
                  else np.full(cdd_acc.shape, np.nan))
    last_doy = int(s['last_doy'])
    return {
        'doys':      np.array([last_doy]),
        'cdd_cum':   cdd_acc[None, ...],
        'tmean':     tmean_last[None, ...],
        'last_doy':  last_doy,
        'transform': s['transform'],
    }


def _gmet_rowcol(transform, lat_vals, lon_vals, h_gmet, w_gmet):
    t = transform
    cols = np.clip(((lon_vals - t[2]) / t[0]).astype(int), 0, w_gmet - 1)
    rows = np.clip(((lat_vals - t[5]) / t[4]).astype(int), 0, h_gmet - 1)
    return rows, cols


def cdd_state_cum_at_doys(state, year, doy_vals, lat_vals, lon_vals):
    """Accumulated CDD at each pixel's own target DOY from a loaded cdd_state.

    Args:
        state: Dict from load_cdd_state, or None.
        year: Calendar year (for the Aug 1 threshold).
        doy_vals: 1D array of per-pixel target DOYs.
        lat_vals, lon_vals: 1D WGS84 coordinate arrays (same length).

    Returns:
        1D float array of accumulated CDD. 0 before Aug 1, or where the DOY is
        earlier than every day in the series.
    """
    n = len(lat_vals)
    out = np.zeros(n)
    if state is None:
        return out
    doys = state['doys']
    h, w = state['cdd_cum'].shape[1:]
    rows, cols = _gmet_rowcol(state['transform'], lat_vals, lon_vals, h, w)
    idxs  = np.searchsorted(doys, doy_vals, side='right') - 1
    valid = (np.asarray(doy_vals) >= _aug1_doy(year)) & (idxs >= 0)
    vals  = state['cdd_cum'][np.where(valid, idxs, 0), rows, cols]
    return np.where(valid, vals, 0.0)


def cdd_state_tmean_at_doys(state, doy_vals, lat_vals, lon_vals):
    """Daily mean temperature (°C) at each pixel's own target DOY from cdd_state.

    Returns NaN where the DOY precedes every day in the stored series.
    """
    n = len(lat_vals)
    if state is None:
        return np.full(n, np.nan)
    doys = state['doys']
    h, w = state['tmean'].shape[1:]
    rows, cols = _gmet_rowcol(state['transform'], lat_vals, lon_vals, h, w)
    idxs  = np.searchsorted(doys, doy_vals, side='right') - 1
    valid = idxs >= 0
    vals  = state['tmean'][np.where(valid, idxs, 0), rows, cols]
    return np.where(valid, vals, np.nan)


def cdd_from_state(state_path, year, target_doy, lat_vals, lon_vals):
    """Latest accumulated CDD (current-prediction convenience wrapper).

    Returns 0 before Aug 1 or if no state file exists yet.
    """
    state = load_cdd_state(state_path)
    if state is None:
        return np.zeros(len(lat_vals))
    doy_vals = np.full(len(lat_vals), target_doy)
    return cdd_state_cum_at_doys(state, year, doy_vals, lat_vals, lon_vals)


def tmean_from_state(state_path, lat_vals, lon_vals):
    """Most recent daily mean temperature (current-prediction convenience wrapper)."""
    state = load_cdd_state(state_path)
    if state is None:
        return np.full(len(lat_vals), np.nan)
    doy_vals = np.full(len(lat_vals), state['last_doy'])
    return cdd_state_tmean_at_doys(state, doy_vals, lat_vals, lon_vals)
