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
from gridmet_utils import (
    cdd_from_state, tmean_from_state,
    load_cdd_state, cdd_state_cum_at_doys, cdd_state_tmean_at_doys,
)
from constants import NODATA

PRED_YEAR = datetime.date.today().year
FEATURE_COLS = ['EVI', 'NDVI', 'evi_delta', 'evi_delta2',
                'ndvi_delta', 'ndvi_delta2', 'day_length_hrs', 'doy_minus_avg_middle',
                'mode_label_7day', 'cdd_accumulated', 'tmean_recent']


_LABEL_LIST   = ['before', 'early', 'late', 'after']
_LABEL_TO_INT = {lbl: i for i, lbl in enumerate(_LABEL_LIST)}


def _day_length_vec(doy, lat_deg):
    """Vectorized day length in hours (numpy form of build_data_table._day_length)."""
    lat  = np.radians(lat_deg)
    decl = np.radians(-23.45 * np.cos(np.radians(360.0 / 365.0 * (doy + 10))))
    arg  = np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0)
    return 2.0 * np.degrees(np.arccos(arg)) / 15.0


def _forward_mode(labels_slot, doy_w, slot_valid, k, obs_doy_k):
    """Ordinal mode of already-predicted older observations in [obs_doy_k-7, -1].

    Mirrors the training feature (build_data_table._add_mode_label_7day) but the
    window labels are the labels re-predicted for older observation slots (j > k)
    during this same forward pass, rather than stored predictions.

    Args:
        labels_slot: (n_px, N) int8 predicted labels per slot (-1 = none yet).
        doy_w: (n_px, N) observation DOYs (slot 0 newest).
        slot_valid: (n_px, N) bool, True where the slot holds a real observation.
        k: Current slot index.
        obs_doy_k: (n_px,) observation DOY of slot k (the window anchor).

    Returns:
        (n_px,) float mode values (0.0–3.0); 0.0 where the window is empty.
    """
    n_px, N = labels_slot.shape
    counts = np.zeros((n_px, 4))
    for j in range(k + 1, N):                       # older observations only
        in_win = (slot_valid[:, j] & (labels_slot[:, j] >= 0)
                  & (doy_w[:, j] >= obs_doy_k - 7) & (doy_w[:, j] <= obs_doy_k - 1))
        lbl_j = labels_slot[:, j]
        for ci in range(4):
            counts[:, ci] += in_win & (lbl_j == ci)
    has = counts.sum(axis=1) > 0
    return np.where(has, np.argmax(counts, axis=1), 0).astype(float)


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

    The mode_label_7day feature is computed by re-predicting the phenological
    state of every stored observation in the past-7-day window (oldest first, so
    each re-prediction can use the labels already assigned to older observations),
    then taking the ordinal mode — no predicted labels are persisted.

    Args:
        state_path: Path to pixel_state_{year}.npz containing per-pixel arrays
            evi_w/ndvi_w/doy_w (h, w, OBS_WINDOW; slot 0 = most recent valid obs).
        date_str: ISO-format date string, e.g. '2026-09-15'.
        output_dir: Path to greendown_outputs directory containing
            norm_stats.json, decision_tree_model.joblib, and transition GeoTIFFs.
        return_features: If True, return raw (pre-normalization) feature grids
            and the re-predicted recent-observation history as extra values.

    Returns:
        Tuple of (pred_grid, forest_mask, transform, crs). When return_features
        is True, two more elements follow: a dict mapping each FEATURE_COLS name
        to a 2D float (h, w) grid of raw slot-0 feature values, and a dict
        {'labels': (h, w, N) int8, 'doys': (h, w, N) float} of the re-predicted
        recent-observation labels (slot 0 newest) for the popup history.
    """
    date = datetime.date.fromisoformat(date_str)
    target_doy = date.timetuple().tm_yday
    year = date.year

    state = dict(np.load(state_path))
    h, w = state['evi_0'].shape

    # Observation window (h, w, N). Migrate a legacy 3-slot state if needed.
    if 'evi_w' in state:
        evi_w  = state['evi_w'].astype(float)
        ndvi_w = state['ndvi_w'].astype(float)
        doy_w  = state['doy_w'].astype(float)
    else:
        N0 = 3
        evi_w  = np.full((h, w, N0), np.nan)
        ndvi_w = np.full((h, w, N0), np.nan)
        doy_w  = np.full((h, w, N0), np.nan)
        for k in range(N0):
            evi_w[:, :, k]  = state[f'evi_{k}'].astype(float)
            ndvi_w[:, :, k] = state[f'ndvi_{k}'].astype(float)
            doy_w[:, :, k]  = state[f'doy_{k}'].astype(float)
    N = evi_w.shape[2]

    mdl = joblib.load(os.path.join(output_dir, 'decision_tree_model.joblib'))
    with open(os.path.join(output_dir, 'norm_stats.json')) as f:
        norm_stats = json.load(f)
    norm_mean = np.array([norm_stats[col]['mean'] for col in FEATURE_COLS])
    norm_std  = np.array([norm_stats[col]['std']  for col in FEATURE_COLS])

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
    avg_middle = _per_pixel_avg_middle(cross_year_lookup, output_dir, h, w,
                                       exclude_year=year)
    cdd_state = load_cdd_state(os.path.join(output_dir, f'cdd_state_{year}.npz'))

    # Forest mask: pixels with a valid most-recent observation (slot 0)
    forest_mask = np.isfinite(evi_w[:, :, 0]) & (evi_w[:, :, 0] > 0)
    rows_idx, cols_idx = np.where(forest_mask)
    n_px = len(rows_idx)

    if n_px == 0:
        return np.full((h, w), 'unknown', dtype=object), forest_mask, transform, crs

    r, c = rows_idx, cols_idx
    EW = evi_w[r, c, :]                 # (n_px, N)
    NW = ndvi_w[r, c, :]
    DW = doy_w[r, c, :]
    lat_pix = lat_array[r, c]
    lon_pix = lon_array[r, c]
    avg_mid_pix = avg_middle[r, c]
    slot_valid = np.isfinite(EW) & (EW > 0) & np.isfinite(NW)   # (n_px, N)

    labels_slot = np.full((n_px, N), -1, dtype=np.int8)
    raw_slot0   = None

    # Forward pass: oldest observation slot first so each re-prediction's
    # mode_label_7day can use the labels already assigned to older observations.
    for k in range(N - 1, -1, -1):
        # Anchor for time/temperature features: today for the current slot
        # (do-not-anchor), the observation's own DOY for past slots.
        anchor = np.full(n_px, float(target_doy)) if k == 0 else DW[:, k]
        obs_doy_k = DW[:, k]                       # mode window anchor (observation)

        Xk = np.zeros((n_px, len(FEATURE_COLS)))
        Xk[:, 0] = EW[:, k]
        Xk[:, 1] = NW[:, k]
        Xk[:, 2] = EW[:, k] - (EW[:, k + 1] if k + 1 < N else np.nan)
        Xk[:, 3] = EW[:, k] - (EW[:, k + 2] if k + 2 < N else np.nan)
        Xk[:, 4] = NW[:, k] - (NW[:, k + 1] if k + 1 < N else np.nan)
        Xk[:, 5] = NW[:, k] - (NW[:, k + 2] if k + 2 < N else np.nan)
        Xk[:, 6] = _day_length_vec(anchor, lat_pix)

        doy_minus = np.where(np.isfinite(avg_mid_pix), anchor - avg_mid_pix, np.nan)
        if not np.isnan(global_avg_middle):
            doy_minus = np.where(np.isnan(doy_minus),
                                 anchor - global_avg_middle, doy_minus)
        Xk[:, 7]  = doy_minus
        Xk[:, 8]  = _forward_mode(labels_slot, DW, slot_valid, k, obs_doy_k)
        Xk[:, 9]  = cdd_state_cum_at_doys(cdd_state, year, anchor, lat_pix, lon_pix)
        Xk[:, 10] = cdd_state_tmean_at_doys(cdd_state, anchor, lat_pix, lon_pix)

        if k == 0:
            raw_slot0 = Xk.copy()

        Xn = (Xk - norm_mean) / norm_std
        Xn = np.where(np.isnan(Xn), 0.0, Xn)
        preds_k = mdl.predict(Xn)
        enc = np.array([_LABEL_TO_INT.get(p, -1) for p in preds_k], dtype=np.int8)
        labels_slot[:, k] = np.where(slot_valid[:, k], enc, -1)

    pred_grid = np.full((h, w), 'unknown', dtype=object)
    pred_grid[r, c] = [_LABEL_LIST[v] if v >= 0 else 'unknown'
                       for v in labels_slot[:, 0]]

    if not return_features:
        return pred_grid, forest_mask, transform, crs

    feature_grids = {}
    for j, col in enumerate(FEATURE_COLS):
        grid = np.full((h, w), np.nan)
        grid[r, c] = raw_slot0[:, j]
        feature_grids[col] = grid

    recent_labels_grid = np.full((h, w, N), -1, dtype=np.int8)
    recent_labels_grid[r, c, :] = labels_slot
    recent_doys_grid = np.full((h, w, N), np.nan)
    recent_doys_grid[r, c, :] = DW
    recent_info = {'labels': recent_labels_grid, 'doys': recent_doys_grid}

    return pred_grid, forest_mask, transform, crs, feature_grids, recent_info
