import datetime
import json
import os
import warnings
import joblib
import numpy as np
import rasterio

from build_data_table import (
    _load_lat_lon_arrays,
    _day_length,
    _build_cross_year_transition_lookup,
)
from edit_data_table import _compute_global_avg_middle
from gridmet_utils import cdd_from_state, tmean_from_state
from constants import NODATA

PRED_YEAR = datetime.date.today().year
FEATURE_COLS = ['EVI', 'NDVI', 'evi_delta', 'evi_delta2',
                'ndvi_delta', 'ndvi_delta2', 'day_length_hrs', 'doy_minus_avg_middle',
                'mode_label_7day', 'cdd_accumulated', 'tmean_recent']


def _compute_mode_7day(recent_labels, recent_label_doys, doy0, r, c):
    """Compute the ordinal mode of recent labels in the [doy0-7, doy0-1] window.

    Mirrors the training-time feature (build_data_table._add_mode_label_7day):
    for the pixel's most recent observation at DOY doy0, take the ordinal mode of
    the labels of that pixel's prior observations whose DOY falls within the 7-day
    window strictly before doy0. The label history is keyed by observation DOY
    (not by Action run), so only days with a real satellite observation count and
    days without new imagery are not double-counted.

    Args:
        recent_labels: (h, w, 7) int8 array; -1 = empty slot,
            0=before, 1=early, 2=late, 3=after. Slot 0 = most recent observation.
        recent_label_doys: (h, w, 7) int16 array of the observation DOY for each
            label slot; -1 = empty.
        doy0: (h, w) float array of each pixel's most recent observation DOY.
        r, c: Row and column index arrays for forest pixels (length n_px).

    Returns:
        1D float array of length n_px with mode values (0.0–3.0), or 0.0
        (before) where no prior observation exists in the window.
    """
    rl  = recent_labels[r, c, :]        # (n_px, 7)
    rld = recent_label_doys[r, c, :]    # (n_px, 7)
    d0  = doy0[r, c]                     # (n_px,)
    mode_vals = np.zeros(len(r))
    for i in range(len(r)):
        di = d0[i]
        if not np.isfinite(di):
            continue
        labels = rl[i]
        doys   = rld[i]
        window = (labels >= 0) & (doys >= di - 7) & (doys <= di - 1)
        v = labels[window]
        if len(v) > 0:
            counts = np.bincount(v.astype(np.intp), minlength=4)
            mode_vals[i] = float(np.argmax(counts))
    return mode_vals


def _per_pixel_avg_middle(cross_year_lookup, output_dir, h, w, exclude_year=None):
    """Average middle-transition DOY per pixel across historical years.

    Mirrors the cross-year averaging used when building the training table
    (build_feature_table): for each pixel, average the middle-transition DOY
    over all available years except exclude_year. This is the value the model
    was trained on, so prediction must use the same per-pixel average rather
    than a single global constant.

    When the per-year transition GeoTIFFs are not available (e.g. in the
    automated Action environment, where they are gitignored), falls back to the
    committed greendown_middle_avg.tif, which is itself the per-pixel mean
    middle-transition DOY across all years.

    Args:
        cross_year_lookup: Dict {year: {phase: array(h, w)}} from
            _build_cross_year_transition_lookup.
        output_dir: Directory containing greendown_middle_avg.tif (fallback).
        h: Raster height in pixels.
        w: Raster width in pixels.
        exclude_year: Year to omit from the average (the prediction year).

    Returns:
        Array of shape (h, w) with the mean middle-transition DOY per pixel,
        NaN where no value is available.
    """
    arrs = [cross_year_lookup[yr]['middle']
            for yr in cross_year_lookup if yr != exclude_year]
    if arrs:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)  # all-NaN slices
            return np.nanmean(np.stack(arrs), axis=0)

    # Fallback: committed CI-filtered cross-year average tif (Action environment).
    # This is precomputed by build_data_table.export_prediction_avg_assets and
    # matches the per-pixel average used during training (NOT greendown_middle_avg.tif,
    # which is an unfiltered mean and differs by ~14 days).
    avg_path = os.path.join(output_dir, 'greendown_middle_avg_filtered.tif')
    if os.path.exists(avg_path):
        with rasterio.open(avg_path) as src:
            avg = src.read(1).astype(float)
            nd = src.nodata
        if nd is not None:
            avg[avg == nd] = np.nan
        avg[avg == NODATA] = np.nan
        return avg

    return np.full((h, w), np.nan)


def _load_global_avg_middle(output_dir):
    """Global gap-fill middle-transition DOY, with a committed fallback.

    Prefers the live computation from per-year CI GeoTIFFs; if those are absent
    (the Action environment), reads the precomputed value from
    greendown_avg_meta.json written by export_prediction_avg_assets.

    Args:
        output_dir: Path to greendown_outputs.

    Returns:
        Global average middle-transition DOY as a float, or NaN if unavailable.
    """
    val = _compute_global_avg_middle(output_dir)
    if not np.isnan(val):
        return val
    meta_path = os.path.join(output_dir, 'greendown_avg_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return float(json.load(f).get('global_avg_middle', np.nan))
    return val


def predict_phenology(date_str, output_dir):
    """Predict phenological state for every pixel on a given 2025 date.

    Uses the most recent valid EVI/NDVI observation at or before the target date.
    Pixels with no valid observation are labeled 'unknown'.

    Args:
        date_str: ISO-format date string, e.g. '2025-09-15'.
        output_dir: Path to greendown_outputs directory containing the 2025 stack,
            saved model, and norm_stats.json.

    Returns:
        Tuple of (pred_grid, forest_mask, transform, crs) where:
            pred_grid: Array of shape (h, w) with label strings
                ('before', 'early', 'late', 'after', or 'unknown').
            forest_mask: Boolean array of shape (h, w), True for pixels with any
                valid EVI in 2025 (i.e., within the MA forest buffer).
            transform: Affine transform of the prediction raster.
            crs: Coordinate reference system of the prediction raster.
    """
    # Parse date
    date = datetime.date.fromisoformat(date_str)
    target_doy = date.timetuple().tm_yday

    # Load 2025 EVI/NDVI stack (mmap to avoid loading ~2.5 GB into RAM)
    stack = np.load(os.path.join(output_dir, f'hls_indices_stack_{PRED_YEAR}.npy'), mmap_mode='r')
    doys  = np.load(os.path.join(output_dir, f'hls_indices_doys_{PRED_YEAR}.npy'))
    n_imgs, _, h, w = stack.shape

    # Load saved model and normalization statistics
    mdl = joblib.load(os.path.join(output_dir, 'decision_tree_model.joblib'))
    with open(os.path.join(output_dir, 'norm_stats.json')) as f:
        norm_stats = json.load(f)

    # Spatial metadata for mapping
    ref_path = os.path.join(output_dir, f'hls_indices_ref_{PRED_YEAR}.tif')
    if not os.path.exists(ref_path):
        ref_path = os.path.join(output_dir, 'hls_indices_ref_current.tif')
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs       = src.crs

    lat_array, lon_array = _load_lat_lon_arrays(output_dir, PRED_YEAR)

    # Cross-year middle transition DOY lookup (excludes 2025 automatically)
    cross_year_lookup = _build_cross_year_transition_lookup(output_dir)
    global_avg_middle = _load_global_avg_middle(output_dir)

    # Indices of images at or before target_doy, sorted newest-first
    valid_t = np.where(doys <= target_doy)[0]
    valid_t = valid_t[np.argsort(doys[valid_t])[::-1]]  # newest first

    # Build feature matrix row-by-row (vectorized within each time step)
    # Output arrays — NaN = no valid observation
    evi_0  = np.full((h, w), np.nan)
    evi_1  = np.full((h, w), np.nan)
    evi_2  = np.full((h, w), np.nan)
    ndvi_0 = np.full((h, w), np.nan)
    ndvi_1 = np.full((h, w), np.nan)
    ndvi_2 = np.full((h, w), np.nan)
    filled = np.zeros((h, w), dtype=int)  # number of valid obs found per pixel

    for t in valid_t:
        evi_t  = stack[t, 0].astype(float)
        ndvi_t = stack[t, 1].astype(float)
        valid_px = np.isfinite(evi_t) & (evi_t > 0) & np.isfinite(ndvi_t)

        need_0 = valid_px & (filled == 0)
        need_1 = valid_px & (filled == 1)
        need_2 = valid_px & (filled == 2)

        evi_0[need_0]  = evi_t[need_0]
        ndvi_0[need_0] = ndvi_t[need_0]
        evi_1[need_1]  = evi_t[need_1]
        ndvi_1[need_1] = ndvi_t[need_1]
        evi_2[need_2]  = evi_t[need_2]
        ndvi_2[need_2] = ndvi_t[need_2]

        filled[need_0 | need_1 | need_2] += 1

        if (filled >= 3).all():
            break

    # Pixels with at least one valid observation
    has_data = filled >= 1
    rows_idx, cols_idx = np.where(has_data)
    n_px = len(rows_idx)

    if n_px == 0:
        return np.full((h, w), 'unknown', dtype=object), transform, crs

    # Assemble feature matrix
    X = np.zeros((n_px, len(FEATURE_COLS)))
    r, c = rows_idx, cols_idx

    X[:, 0] = evi_0[r, c]                           # EVI
    X[:, 1] = ndvi_0[r, c]                          # NDVI
    X[:, 2] = evi_0[r, c] - evi_1[r, c]             # evi_delta  (NaN if filled<2)
    X[:, 3] = evi_0[r, c] - evi_2[r, c]             # evi_delta2 (NaN if filled<3)
    X[:, 4] = ndvi_0[r, c] - ndvi_1[r, c]           # ndvi_delta
    X[:, 5] = ndvi_0[r, c] - ndvi_2[r, c]           # ndvi_delta2
    X[:, 6] = np.array([_day_length(target_doy, float(lat_array[ri, ci]))
                        for ri, ci in zip(r, c)])    # day_length_hrs

    # doy_minus_avg_middle: per-pixel average of the middle-transition DOY
    # across historical years, gap-filled with the global average.
    avg_middle = _per_pixel_avg_middle(cross_year_lookup, output_dir, h, w,
                                       exclude_year=PRED_YEAR)
    doy_minus = np.where(np.isfinite(avg_middle[r, c]),
                         target_doy - avg_middle[r, c], np.nan)
    # Gap-fill remaining NaNs
    if not np.isnan(global_avg_middle):
        doy_minus = np.where(np.isnan(doy_minus), target_doy - global_avg_middle, doy_minus)
    X[:, 7] = doy_minus
    # No rolling label history in the full-stack path; nan_to_num fills to normalized mean
    X[:, 8] = np.nan
    # CDD and T_mean from current-year state file; nan imputed to mean at normalize step
    cdd_state_path = os.path.join(output_dir, f'cdd_state_{PRED_YEAR}.npz')
    X[:, 9]  = cdd_from_state(cdd_state_path, PRED_YEAR, target_doy,
                               lat_array[r, c], lon_array[r, c])
    X[:, 10] = tmean_from_state(cdd_state_path, lat_array[r, c], lon_array[r, c])

    # Z-score normalization using saved training statistics
    for j, col in enumerate(FEATURE_COLS):
        mean = norm_stats[col]['mean']
        std  = norm_stats[col]['std']
        X[:, j] = (X[:, j] - mean) / std

    # Substitute 0 for NaN delta values (normalized mean) — pixels with < 2/3 prior obs
    X = np.where(np.isnan(X), 0.0, X)

    # Predict
    preds = mdl.predict(X)

    # Map back to spatial grid
    pred_grid = np.full((h, w), 'unknown', dtype=object)
    pred_grid[r, c] = preds

    # Forest mask: pixels with any valid EVI across the full 2025 stack
    forest_mask = np.zeros((h, w), dtype=bool)
    for t in range(n_imgs):
        evi_t = stack[t, 0].astype(float)
        forest_mask |= (np.isfinite(evi_t) & (evi_t > 0))

    return pred_grid, forest_mask, transform, crs


def predict_from_pixel_state(state_path, date_str, output_dir,
                              return_features=False):
    """Predict phenological state using the compact rolling pixel state file.

    Used by the automated daily pipeline (GitHub Action) instead of the full
    year stack. Produces identical predictions to predict_phenology() for the
    same date, as long as the pixel state is current.

    Args:
        state_path: Path to pixel_state_{year}.npz containing per-pixel arrays
            evi_0/1/2, ndvi_0/1/2, doy_0/1/2 (index 0 = most recent valid obs).
        date_str: ISO-format date string, e.g. '2026-09-15'.
        output_dir: Path to greendown_outputs directory containing
            norm_stats.json, decision_tree_model.joblib, and transition GeoTIFFs.
        return_features: If True, return raw (pre-normalization) feature grids
            as an additional dict.

    Returns:
        Tuple of (pred_grid, forest_mask, transform, crs). When return_features
        is True, a fifth element is included: a dict mapping each FEATURE_COLS
        name to a 2D float array of shape (h, w) with raw feature values (NaN
        for non-forest pixels).
    """
    date = datetime.date.fromisoformat(date_str)
    target_doy = date.timetuple().tm_yday
    year = date.year

    state = np.load(state_path)
    evi_0  = state['evi_0'].astype(float)
    evi_1  = state['evi_1'].astype(float)
    evi_2  = state['evi_2'].astype(float)
    ndvi_0 = state['ndvi_0'].astype(float)
    ndvi_1 = state['ndvi_1'].astype(float)
    ndvi_2 = state['ndvi_2'].astype(float)
    h, w = evi_0.shape
    doy_0 = (state['doy_0'].astype(float)
             if 'doy_0' in state.files
             else np.full((h, w), np.nan))
    recent_labels = (state['recent_labels']
                     if 'recent_labels' in state.files
                     else np.full((h, w, 7), -1, dtype=np.int8))
    # Observation DOY for each label slot; absent on legacy (run-indexed) states,
    # in which case the empty (-1) DOYs make the window exclude all old labels and
    # the history rebuilds, observation by observation, with the new semantics.
    recent_label_doys = (state['recent_label_doys']
                         if 'recent_label_doys' in state.files
                         else np.full((h, w, 7), -1, dtype=np.int16))

    mdl = joblib.load(os.path.join(output_dir, 'decision_tree_model.joblib'))
    with open(os.path.join(output_dir, 'norm_stats.json')) as f:
        norm_stats = json.load(f)

    # Use current-year ref raster if available; fall back to stable copy
    ref_path = os.path.join(output_dir, f'hls_indices_ref_{year}.tif')
    if not os.path.exists(ref_path):
        ref_path = os.path.join(output_dir, 'hls_indices_ref_current.tif')
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs       = src.crs

    lat_array, lon_array = _load_lat_lon_arrays(output_dir, year)
    cross_year_lookup = _build_cross_year_transition_lookup(output_dir)
    global_avg_middle = _load_global_avg_middle(output_dir)

    # Forest mask: pixels with at least one valid observation in the state
    forest_mask = np.isfinite(evi_0) & (evi_0 > 0)
    rows_idx, cols_idx = np.where(forest_mask)
    n_px = len(rows_idx)

    if n_px == 0:
        return np.full((h, w), 'unknown', dtype=object), forest_mask, transform, crs

    r, c = rows_idx, cols_idx
    X = np.zeros((n_px, len(FEATURE_COLS)))
    X[:, 0] = evi_0[r, c]
    X[:, 1] = ndvi_0[r, c]
    X[:, 2] = evi_0[r, c] - evi_1[r, c]
    X[:, 3] = evi_0[r, c] - evi_2[r, c]
    X[:, 4] = ndvi_0[r, c] - ndvi_1[r, c]
    X[:, 5] = ndvi_0[r, c] - ndvi_2[r, c]
    X[:, 6] = np.array([_day_length(target_doy, float(lat_array[ri, ci]))
                        for ri, ci in zip(r, c)])

    # doy_minus_avg_middle: per-pixel average of the middle-transition DOY
    # across historical years, gap-filled with the global average.
    avg_middle = _per_pixel_avg_middle(cross_year_lookup, output_dir, h, w,
                                       exclude_year=year)
    doy_minus = np.where(np.isfinite(avg_middle[r, c]),
                         target_doy - avg_middle[r, c], np.nan)
    if not np.isnan(global_avg_middle):
        doy_minus = np.where(np.isnan(doy_minus),
                             target_doy - global_avg_middle, doy_minus)
    X[:, 7] = doy_minus
    X[:, 8] = _compute_mode_7day(recent_labels, recent_label_doys, doy_0, r, c)

    # cdd_accumulated and tmean_recent: look up from current-year state file
    cdd_state_path = os.path.join(output_dir, f'cdd_state_{year}.npz')
    X[:, 9]  = cdd_from_state(cdd_state_path, year, target_doy,
                               lat_array[r, c], lon_array[r, c])
    X[:, 10] = tmean_from_state(cdd_state_path, lat_array[r, c], lon_array[r, c])

    # Capture raw (pre-normalization) feature values for the popup JSON
    X_raw = X.copy()

    for j, col in enumerate(FEATURE_COLS):
        mean = norm_stats[col]['mean']
        std  = norm_stats[col]['std']
        X[:, j] = (X[:, j] - mean) / std
    X = np.where(np.isnan(X), 0.0, X)

    preds = mdl.predict(X)
    pred_grid = np.full((h, w), 'unknown', dtype=object)
    pred_grid[r, c] = preds

    if not return_features:
        return pred_grid, forest_mask, transform, crs

    feature_grids = {}
    for j, col in enumerate(FEATURE_COLS):
        grid = np.full((h, w), np.nan)
        grid[r, c] = X_raw[:, j]
        feature_grids[col] = grid

    return pred_grid, forest_mask, transform, crs, feature_grids
