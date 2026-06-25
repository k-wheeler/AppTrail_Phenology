import datetime
import os
import numpy as np
import ee
import geemap
import rasterio
from scipy.optimize import curve_fit

from constants import NODATA

N_CI_SAMPLES = 200  # Monte Carlo samples for confidence intervals


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

def compute_average_transition_dates(paths_by_year, output_dir='.'):
    """Compute pixel-wise mean transition DOYs across years.

    Args:
        paths_by_year: List of dicts returned by compute_transition_dates,
            one per year.
        output_dir: Directory to write the averaged GeoTIFFs.

    Returns:
        Dict of {'start': path, 'middle': path, 'end': path} for averaged rasters.
    """
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

        path = os.path.join(output_dir, f'greendown_{phase}_avg.tif')
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(mean, 1)
        avg_paths[phase] = path

    return avg_paths
