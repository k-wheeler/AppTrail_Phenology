import datetime
import json
import os
import joblib
import numpy as np
import rasterio

from build_data_table import (
    _load_lat_array,
    _day_length,
    _build_cross_year_transition_lookup,
)
from edit_data_table import _compute_global_avg_middle

PRED_YEAR = datetime.date.today().year
FEATURE_COLS = ['EVI', 'NDVI', 'evi_delta', 'evi_delta2',
                'ndvi_delta', 'ndvi_delta2', 'day_length_hrs', 'doy_minus_avg_middle']


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

    # Latitude array for day-length calculation
    lat_array = _load_lat_array(output_dir, PRED_YEAR)

    # Cross-year middle transition DOY lookup (excludes 2025 automatically)
    cross_year_lookup = _build_cross_year_transition_lookup(output_dir)
    global_avg_middle = _compute_global_avg_middle(output_dir)

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

    # doy_minus_avg_middle: per-pixel cross-year average, gap-fill with global avg
    doy_minus = np.full(n_px, np.nan)
    if PRED_YEAR in cross_year_lookup:
        mid_arr = cross_year_lookup[PRED_YEAR]['middle']
        doy_minus = np.where(np.isfinite(mid_arr[r, c]),
                             target_doy - mid_arr[r, c],
                             np.nan)
    # Gap-fill remaining NaNs
    if not np.isnan(global_avg_middle):
        doy_minus = np.where(np.isnan(doy_minus), target_doy - global_avg_middle, doy_minus)
    X[:, 7] = doy_minus

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


def predict_from_pixel_state(state_path, date_str, output_dir):
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

    Returns:
        Tuple of (pred_grid, forest_mask, transform, crs) — same as
        predict_phenology().
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

    lat_array = _load_lat_array(output_dir, year)
    cross_year_lookup = _build_cross_year_transition_lookup(output_dir)
    global_avg_middle = _compute_global_avg_middle(output_dir)

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

    doy_minus = np.full(n_px, np.nan)
    if year in cross_year_lookup:
        mid_arr = cross_year_lookup[year]['middle']
        doy_minus = np.where(np.isfinite(mid_arr[r, c]),
                             target_doy - mid_arr[r, c], np.nan)
    if not np.isnan(global_avg_middle):
        doy_minus = np.where(np.isnan(doy_minus),
                             target_doy - global_avg_middle, doy_minus)
    X[:, 7] = doy_minus

    for j, col in enumerate(FEATURE_COLS):
        mean = norm_stats[col]['mean']
        std  = norm_stats[col]['std']
        X[:, j] = (X[:, j] - mean) / std
    X = np.where(np.isnan(X), 0.0, X)

    preds = mdl.predict(X)
    pred_grid = np.full((h, w), 'unknown', dtype=object)
    pred_grid[r, c] = preds

    return pred_grid, forest_mask, transform, crs
