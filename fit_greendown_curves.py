import datetime
import glob
import os
import numpy as np
import ee
import geemap
import rasterio
from scipy.optimize import curve_fit

from constants import NODATA

N_CI_SAMPLES = 200  # Monte Carlo samples for confidence intervals

# Number of most-recent valid observations retained per pixel. The prediction
# only needs 3 for the spectral deltas, but the mode_label_7day feature re-predicts
# every observation in the 7-day window, which (plus the two predecessors each
# needs for its own deltas) can reach back several observations.
OBS_WINDOW = 20


# ----------------------------
# Logistic model
# ----------------------------

def _decreasing_logistic(t, L, k, t_mid, offset):
    """Decreasing logistic: high in July, low in December."""
    return L / (1 + np.exp(k * (t - t_mid))) + offset


def _curvature_extrema_doys(k, t_mid):
    """Compute the three DOYs where d(curvature)/dt = 0 on a logistic curve.

    These correspond analytically to the start, middle, and end of the
    greendown transition.

    Args:
        k: Steepness parameter of the logistic curve.
        t_mid: Inflection point (middle transition DOY).

    Returns:
        Tuple of (start_doy, middle_doy, end_doy).
    """
    delta = np.log(2 + np.sqrt(3)) / k
    return t_mid - delta, t_mid, t_mid + delta


def _fit_pixel(doys, values):
    """Fit a decreasing logistic to one pixel's EVI time series.

    Args:
        doys: Array of day-of-year values.
        values: Corresponding EVI values (NaN or <= 0 are excluded).

    Returns:
        Tuple of (popt, pcov) on success, or None if fit fails. pcov may be
        None if the covariance matrix is non-finite.
    """
    valid = np.isfinite(values) & (values > 0)
    if valid.sum() < 4:
        return None
    t = doys[valid].astype(float)
    y = values[valid].astype(float)
    try:
        popt, pcov = curve_fit(
            _decreasing_logistic, t, y,
            p0=[y.max() - y.min(), 0.1, float(t[len(t) // 2]), y.min()],
            bounds=([0, 0.01, 150, -0.5], [1.5, 1.0, 365, 1.0]),
            maxfev=5000
        )
        # Reject fits with non-finite covariance (poorly constrained)
        if not np.all(np.isfinite(pcov)):
            return popt, None
        return popt, pcov
    except Exception:
        return None


def _make_psd(pcov):
    """Force a covariance matrix to be positive semi-definite.

    Clips negative eigenvalues to zero and reconstructs via eigen-decomposition.

    Args:
        pcov: Square covariance matrix from curve_fit.

    Returns:
        Positive semi-definite version of pcov.
    """
    eigvals, eigvecs = np.linalg.eigh(pcov)
    eigvals = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _sample_params(popt, pcov, n=N_CI_SAMPLES):
    """Draw parameter sets from the multivariate normal defined by a curve_fit result.

    Args:
        popt: Best-fit parameter vector of length 4.
        pcov: Covariance matrix from curve_fit.
        n: Number of Monte Carlo samples to draw.

    Returns:
        Array of shape (m, 4) with valid samples (k > 0), or None if fewer than
        10 valid samples were obtained.
    """
    try:
        pcov_psd = _make_psd(pcov)
        samples  = np.random.multivariate_normal(popt, pcov_psd, size=n, check_valid='ignore')
        # Keep only samples with positive k (physically required)
        samples = samples[samples[:, 1] > 0]
        return samples if len(samples) >= 10 else None
    except (np.linalg.LinAlgError, ValueError):
        return None


def compute_curve_ci(popt, pcov, t):
    """Compute a 95% CI band for the fitted logistic at times t.

    Args:
        popt: Best-fit parameter vector.
        pcov: Covariance matrix from curve_fit, or None for no uncertainty.
        t: Array of DOY values at which to evaluate the curve.

    Returns:
        Tuple of (fitted, lower, upper) arrays at the given times.
    """
    fitted = _decreasing_logistic(t, *popt)
    if pcov is None:
        return fitted, fitted, fitted

    samples = _sample_params(popt, pcov)
    if samples is None:
        return fitted, fitted, fitted

    curves = np.array([_decreasing_logistic(t, *p) for p in samples])
    lower  = np.clip(np.percentile(curves, 2.5,  axis=0), 0, None)
    upper  = np.clip(np.percentile(curves, 97.5, axis=0), 0, None)
    return fitted, lower, upper


def compute_transition_dates_ci(popt, pcov):
    """Compute point estimates and 95% CI for the three greendown transition DOYs.

    Args:
        popt: Best-fit parameter vector from curve_fit.
        pcov: Covariance matrix from curve_fit, or None for no uncertainty.

    Returns:
        Dict with keys 'start', 'middle', 'end', each mapping to a tuple of
        (point_estimate, ci_lower, ci_upper).
    """
    s, m, e = _curvature_extrema_doys(popt[1], popt[2])

    nan = float('nan')
    if pcov is None:
        return {
            'start':  (s, nan, nan),
            'middle': (m, nan, nan),
            'end':    (e, nan, nan),
        }

    samples = _sample_params(popt, pcov)
    if samples is None:
        return {
            'start':  (s, nan, nan),
            'middle': (m, nan, nan),
            'end':    (e, nan, nan),
        }

    all_dates = np.array([_curvature_extrema_doys(p[1], p[2]) for p in samples])
    result = {}
    for i, phase in enumerate(('start', 'middle', 'end')):
        d = all_dates[:, i]
        result[phase] = (
            _curvature_extrema_doys(popt[1], popt[2])[i],  # point estimate
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)),
        )
    return result


# ----------------------------
# Export helpers
# ----------------------------

def _export_stack(collection, ma_forest, route_buffer, year, output_dir):
    """Export EVI and NDVI images from GEE and stack into a numpy array.

    Exports one image at a time (~20 MB each, under the 50 MB limit) and
    caches the result to disk to avoid re-downloading.

    Args:
        collection: GEE ImageCollection with EVI and NDVI bands.
        ma_forest: GEE Image binary forest mask.
        route_buffer: GEE Geometry defining the export region.
        year: Integer year being exported.
        output_dir: Directory to write cached stack files.

    Returns:
        Tuple of (stack, doys, ref_path) where stack is array of shape
        (n_images, 2, h, w), doys is a 1D DOY array, and ref_path is the
        path to the single-band reference GeoTIFF.
    """
    stack_path = os.path.join(output_dir, f'hls_indices_stack_{year}.npy')
    doys_path  = os.path.join(output_dir, f'hls_indices_doys_{year}.npy')
    ref_path   = os.path.join(output_dir, f'hls_indices_ref_{year}.tif')

    if os.path.exists(stack_path) and os.path.exists(doys_path):
        print(f'  Loading cached stack for {year}')
        return np.load(stack_path), np.load(doys_path), ref_path

    timestamps = collection.aggregate_array('system:time_start').getInfo()
    doys = np.array([
        datetime.datetime.utcfromtimestamp(ts / 1000).timetuple().tm_yday
        for ts in timestamps
    ])
    n = len(doys)

    if n < 4:
        raise ValueError(f'Too few images ({n}) for {year}')

    image_list = collection.toList(n)
    arrays  = []
    profile = None

    print(f'  Exporting {n} images for {year} (one at a time)...')
    for i in range(n):
        img = ee.Image(image_list.get(i)).select(['EVI', 'NDVI']).updateMask(ma_forest)
        tmp_path = os.path.join(output_dir, f'_tmp_{year}_{i:03d}.tif')

        for attempt in range(3):
            geemap.ee_export_image(
                img,
                filename=tmp_path,
                scale=30,
                region=route_buffer,
                file_per_band=False
            )
            if os.path.exists(tmp_path):
                break
            print(f'    Download failed (attempt {attempt + 1}/3), retrying...')
        else:
            raise RuntimeError(f'Download failed after 3 attempts: year={year}, image={i + 1}/{n}')

        with rasterio.open(tmp_path) as src:
            data = src.read().astype(float)  # (n_bands, h, w): band 0=EVI, band 1=NDVI
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            arrays.append(data)
            if profile is None:
                profile = src.profile
                ref_profile = src.profile.copy()
                ref_profile.update(count=1)
                with rasterio.open(ref_path, 'w', **ref_profile) as dst:
                    dst.write(src.read(1), 1)  # write EVI band for spatial reference

        os.remove(tmp_path)
        print(f'    {i + 1}/{n}', end='\r')

    print()
    arr = np.stack(arrays, axis=0)  # (n_images, n_bands, height, width): band 0=EVI, band 1=NDVI
    np.save(stack_path, arr)
    np.save(doys_path, doys)

    return arr, doys, ref_path


# ----------------------------
# Per-year fitting
# ----------------------------

def compute_transition_dates(hls_indices_collection, route_buffer, ma_forest, year, output_dir='.'):
    """Fit greendown curves and save transition date GeoTIFFs for one year.

    Exports Jul–Dec EVI and NDVI per image from GEE, fits a decreasing logistic
    per forest pixel using EVI only, and saves 12 GeoTIFFs:
        greendown_{phase}_{year}.tif            — point estimate
        greendown_{phase}_ci_lower_{year}.tif   — 2.5th percentile
        greendown_{phase}_ci_upper_{year}.tif   — 97.5th percentile
        greendown_{phase}_ci_width_{year}.tif   — upper minus lower

    Results are cached; re-running skips already-completed years.

    Args:
        hls_indices_collection: GEE ImageCollection with EVI and NDVI bands.
        route_buffer: GEE Geometry defining the spatial extent.
        ma_forest: GEE Image binary forest mask.
        year: Integer year to process.
        output_dir: Directory for cached stacks and output GeoTIFFs.

    Returns:
        Dict of {'start': path, 'middle': path, 'end': path} for point estimates.
    """
    phases = ('start', 'middle', 'end')
    point_paths = {p: os.path.join(output_dir, f'greendown_{p}_{year}.tif')          for p in phases}
    lower_paths = {p: os.path.join(output_dir, f'greendown_{p}_ci_lower_{year}.tif') for p in phases}
    upper_paths = {p: os.path.join(output_dir, f'greendown_{p}_ci_upper_{year}.tif') for p in phases}
    width_paths = {p: os.path.join(output_dir, f'greendown_{p}_ci_width_{year}.tif') for p in phases}

    all_paths = (list(point_paths.values()) + list(lower_paths.values()) +
                 list(upper_paths.values()) + list(width_paths.values()))
    if all(os.path.exists(p) for p in all_paths):
        print(f'  Using cached results for {year}')
        return point_paths

    collection = hls_indices_collection.sort('system:time_start')
    arr, doys, ref_path = _export_stack(collection, ma_forest, route_buffer, year, output_dir)

    _, _, h, w = arr.shape
    results = {
        phase: {
            'point': np.full((h, w), np.nan, dtype=np.float32),
            'lower': np.full((h, w), np.nan, dtype=np.float32),
            'upper': np.full((h, w), np.nan, dtype=np.float32),
        }
        for phase in phases
    }

    print(f'  Fitting logistic curves with CIs for {year}...')
    for i in range(h):
        for j in range(w):
            fit = _fit_pixel(doys, arr[:, 0, i, j])  # EVI only (band 0) for greendown fitting
            if fit is None:
                continue
            popt, pcov = fit
            ci = compute_transition_dates_ci(popt, pcov)
            for phase in phases:
                results[phase]['point'][i, j] = ci[phase][0]
                results[phase]['lower'][i, j] = ci[phase][1]
                results[phase]['upper'][i, j] = ci[phase][2]

    # Clip CI bounds to the analysis window [Jul 1, Dec 31]
    jul1_doy  = datetime.date(year, 7,  1).timetuple().tm_yday
    dec31_doy = datetime.date(year, 12, 31).timetuple().tm_yday
    full_width = dec31_doy - jul1_doy
    for phase in phases:
        for key in ('lower', 'upper'):
            results[phase][key] = np.where(
                np.isfinite(results[phase][key]),
                np.clip(results[phase][key], jul1_doy, dec31_doy),
                np.nan
            )

    # Nullify pixels where CI spans the full window — CI was clipped on both
    # ends, making it uninformative (equivalent to a failed Monte Carlo fit)
    for phase in phases:
        full_span = (results[phase]['upper'] - results[phase]['lower']) >= full_width
        for key in ('point', 'lower', 'upper'):
            results[phase][key] = np.where(full_span, np.nan, results[phase][key])

    with rasterio.open(ref_path) as src:
        out_profile = src.profile.copy()
    out_profile.update(count=1, dtype='float32', nodata=NODATA)

    for phase in phases:
        for key, path in [('point', point_paths[phase]),
                          ('lower', lower_paths[phase]),
                          ('upper', upper_paths[phase])]:
            data = results[phase][key].copy()
            data[np.isnan(data)] = NODATA
            with rasterio.open(path, 'w', **out_profile) as dst:
                dst.write(data, 1)

        # CI width = upper - lower (NaN where either bound is NaN)
        lower = results[phase]['lower']
        upper = results[phase]['upper']
        width = np.where(np.isfinite(lower) & np.isfinite(upper), upper - lower, np.nan)
        width_data = width.astype(np.float32)
        width_data[np.isnan(width_data)] = NODATA
        with rasterio.open(width_paths[phase], 'w', **out_profile) as dst:
            dst.write(width_data, 1)

    print(f'  Done: {year}')
    return point_paths


# ----------------------------
# Multi-year average
# ----------------------------

def _seed_window(state, prefix, h, w):
    """Build an (h, w, OBS_WINDOW) observation window for a variable.

    Seeds the first three slots from a legacy 3-slot pixel state
    ({prefix}_0/1/2) when present; remaining slots are NaN.
    """
    win = np.full((h, w, OBS_WINDOW), np.nan, dtype=np.float32)
    for k in range(min(3, OBS_WINDOW)):
        key = f'{prefix}_{k}'
        if key in state:
            win[:, :, k] = state[key]
    return win


def update_pixel_state(collection, ma_forest, route_buffer, year, output_dir):
    """Download new HLS images and update the compact rolling pixel state file.

    The pixel state stores the 3 most recent valid EVI/NDVI observations per
    pixel, which is all that is needed to compute prediction features without
    keeping the full year stack. Only images with DOY greater than the maximum
    already in the state are downloaded, making each daily run fast.

    Args:
        collection: GEE ImageCollection with EVI and NDVI bands (Jun–Dec).
        ma_forest: GEE Image binary forest mask.
        route_buffer: GEE Geometry defining the export region.
        year: Integer year being processed.
        output_dir: Directory to read/write pixel_state_{year}.npz and the
            reference GeoTIFF hls_indices_ref_{year}.tif.

    Returns:
        Path to the updated pixel_state_{year}.npz file.
    """
    state_path = os.path.join(output_dir, f'pixel_state_{year}.npz')
    ref_path   = os.path.join(output_dir, f'hls_indices_ref_{year}.tif')

    # Load existing state or initialise blank arrays once we know (h, w)
    existing_doys = set()
    state = None
    if os.path.exists(state_path):
        state = dict(np.load(state_path))
        h, w = state['evi_0'].shape
        existing_doys = set(int(d) for d in state.get('seen_doys', []))
        # evi_w/ndvi_w/doy_w hold the last OBS_WINDOW observations per pixel, used
        # to re-predict recent days for the mode_label_7day feature. Migrate a
        # legacy state (only the 3-slot evi_0/1/2 window, or an older stored-label
        # schema) by seeding the window from those three slots.
        if 'evi_w' not in state:
            state['evi_w']  = _seed_window(state, 'evi',  h, w)
            state['ndvi_w'] = _seed_window(state, 'ndvi', h, w)
            state['doy_w']  = _seed_window(state, 'doy',  h, w)
        # Expand window arrays if OBS_WINDOW grew (e.g. 8 → 20). Pad older slots
        # with NaN so existing observations are preserved at the newest end.
        for key in ('evi_w', 'ndvi_w', 'doy_w'):
            old = state[key]
            if old.shape[2] < OBS_WINDOW:
                expanded = np.full((h, w, OBS_WINDOW), np.nan, dtype=np.float32)
                expanded[:, :, :old.shape[2]] = old
                state[key] = expanded
        # Predictions are no longer persisted; drop any legacy label-history arrays.
        state.pop('recent_labels', None)
        state.pop('recent_label_doys', None)
        print(f'  Loaded existing pixel state ({len(existing_doys)} DOYs already processed)')

    # Fetch all image timestamps from GEE and filter to ones not yet processed
    timestamps = collection.sort('system:time_start').aggregate_array('system:time_start').getInfo()
    all_doys = [
        datetime.datetime.utcfromtimestamp(ts / 1000).timetuple().tm_yday
        for ts in timestamps
    ]
    new_indices = [i for i, d in enumerate(all_doys) if d not in existing_doys]

    if not new_indices:
        print(f'  No new images for {year}; pixel state is current.')
        return state_path

    print(f'  Downloading {len(new_indices)} new image(s) for {year}...')
    image_list = collection.sort('system:time_start').toList(len(timestamps))

    for idx in new_indices:
        doy = all_doys[idx]
        img = ee.Image(image_list.get(idx)).select(['EVI', 'NDVI']).updateMask(ma_forest)
        tmp_path = os.path.join(output_dir, f'_tmp_state_{year}_{idx:03d}.tif')

        for attempt in range(3):
            geemap.ee_export_image(img, filename=tmp_path, scale=30,
                                   region=route_buffer, file_per_band=False)
            if os.path.exists(tmp_path):
                break
            print(f'    Attempt {attempt + 1}/3 failed, retrying...')
        else:
            print(f'    Skipping DOY {doy} after 3 failed attempts.')
            continue

        with rasterio.open(tmp_path) as src:
            data = src.read().astype(float)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan

            # Initialise state arrays on first downloaded image
            if state is None:
                h, w = src.height, src.width
                nan_hw = np.full((h, w), np.nan, dtype=np.float32)
                nan_win = np.full((h, w, OBS_WINDOW), np.nan, dtype=np.float32)
                state = {
                    'evi_0':  nan_hw.copy(), 'evi_1':  nan_hw.copy(), 'evi_2':  nan_hw.copy(),
                    'ndvi_0': nan_hw.copy(), 'ndvi_1': nan_hw.copy(), 'ndvi_2': nan_hw.copy(),
                    'doy_0':  nan_hw.copy(), 'doy_1':  nan_hw.copy(), 'doy_2':  nan_hw.copy(),
                    'evi_w':  nan_win.copy(), 'ndvi_w': nan_win.copy(), 'doy_w': nan_win.copy(),
                }

            # Write reference rasters if not already saved
            if not os.path.exists(ref_path):
                ref_profile = src.profile.copy()
                ref_profile.update(count=1)
                with rasterio.open(ref_path, 'w', **ref_profile) as dst:
                    dst.write(src.read(1), 1)
            current_ref = os.path.join(output_dir, 'hls_indices_ref_current.tif')
            if not os.path.exists(current_ref):
                import shutil
                shutil.copy2(ref_path, current_ref)

        os.remove(tmp_path)

        evi_new  = data[0]
        ndvi_new = data[1]
        valid = np.isfinite(evi_new) & (evi_new > 0) & np.isfinite(ndvi_new)

        # Shift the 3-slot rolling window and insert the new observation at index 0
        state['evi_2'][valid]  = state['evi_1'][valid]
        state['evi_1'][valid]  = state['evi_0'][valid]
        state['evi_0'][valid]  = evi_new[valid].astype(np.float32)
        state['ndvi_2'][valid] = state['ndvi_1'][valid]
        state['ndvi_1'][valid] = state['ndvi_0'][valid]
        state['ndvi_0'][valid] = ndvi_new[valid].astype(np.float32)
        state['doy_2'][valid]  = state['doy_1'][valid]
        state['doy_1'][valid]  = state['doy_0'][valid]
        state['doy_0'][valid]  = np.float32(doy)

        # Shift the OBS_WINDOW-slot observation history (slot 0 = newest) for the
        # mode_label_7day re-prediction. Only valid pixels advance.
        for win, new in (('evi_w', evi_new), ('ndvi_w', ndvi_new), ('doy_w', None)):
            arr = state[win]
            arr[valid, 1:] = arr[valid, :-1]
            arr[valid, 0]  = (np.float32(doy) if new is None
                              else new[valid].astype(np.float32))

        existing_doys.add(doy)
        print(f'    Processed DOY {doy} ({len(valid[valid])} valid pixels)')

    state['seen_doys'] = np.array(sorted(existing_doys), dtype=np.int32)
    np.savez_compressed(state_path, **state)
    print(f'  Saved pixel state → {state_path}')
    return state_path


def _canonical_crs(output_dir):
    """Return the authoritative CRS from a reference raster, or None.

    Guards against transition GeoTIFFs that may carry a mislabeled CRS by
    sourcing the CRS from the hls_indices_ref rasters, which are written
    directly from the GEE download.

    Args:
        output_dir: Directory containing hls_indices_ref_*.tif.

    Returns:
        A rasterio CRS, or None if no reference raster is found.
    """
    candidates = [os.path.join(output_dir, 'hls_indices_ref_current.tif')]
    candidates += sorted(glob.glob(os.path.join(output_dir, 'hls_indices_ref_*.tif')))
    for path in candidates:
        if os.path.exists(path):
            with rasterio.open(path) as src:
                if src.crs is not None:
                    return src.crs
    return None


def compute_average_transition_dates(paths_by_year, output_dir='.'):
    """Compute pixel-wise mean transition DOYs across years.

    Args:
        paths_by_year: List of dicts returned by compute_transition_dates,
            one per year.
        output_dir: Directory to write the averaged GeoTIFFs.

    Returns:
        Dict of {'start': path, 'middle': path, 'end': path} for averaged rasters.
    """
    ref_crs = _canonical_crs(output_dir)
    avg_paths = {}
    for phase in ('start', 'middle', 'end'):
        arrays  = []
        profile = None
        for paths in paths_by_year:
            with rasterio.open(paths[phase]) as src:
                data = src.read(1).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
                arrays.append(data)
                if profile is None:
                    profile = src.profile

        mean = np.nanmean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
        mean[np.isnan(mean)] = NODATA

        # Use the authoritative CRS from the reference raster, not the per-year
        # tif's (which has historically been mislabeled).
        if ref_crs is not None:
            profile.update(crs=ref_crs)

        path = os.path.join(output_dir, f'greendown_{phase}_avg.tif')
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(mean, 1)
        avg_paths[phase] = path

    return avg_paths
