import datetime
import os
import numpy as np
import ee
import geemap
import rasterio
from scipy.optimize import curve_fit


# ----------------------------
# Logistic model
# ----------------------------

def _decreasing_logistic(t, L, k, t_mid, offset):
    """Decreasing logistic: high in July, low in December."""
    return L / (1 + np.exp(k * (t - t_mid))) + offset


def _curvature_extrema_doys(k, t_mid):
    """
    Analytical solution for the three DOYs where d(curvature)/dt = 0
    on a logistic curve, corresponding to the start, middle, and end
    of the greendown transition.
    """
    delta = np.log(2 + np.sqrt(3)) / k
    return t_mid - delta, t_mid, t_mid + delta


def _fit_pixel(doys, values):
    """Fit a decreasing logistic to one pixel's time series. Returns params or None."""
    valid = np.isfinite(values) & (values > 0)
    if valid.sum() < 4:
        return None
    t = doys[valid].astype(float)
    y = values[valid].astype(float)
    try:
        popt, _ = curve_fit(
            _decreasing_logistic, t, y,
            p0=[y.max() - y.min(), 0.1, float(t[len(t) // 2]), y.min()],
            bounds=([0, 0.01, 150, -0.5], [1.5, 1.0, 365, 1.0]),
            maxfev=5000
        )
        return popt
    except Exception:
        return None


# ----------------------------
# Export helpers
# ----------------------------

def _export_stack(collection, ma_forest, route_buffer, year, output_dir):
    """
    Export EVI images one at a time (each ~20MB, under the 50MB limit)
    and stack into a numpy array. Caches results to disk.
    """
    stack_path = os.path.join(output_dir, f'hls_evi_stack_{year}.npy')
    doys_path  = os.path.join(output_dir, f'hls_evi_doys_{year}.npy')
    ref_path   = os.path.join(output_dir, f'hls_evi_ref_{year}.tif')

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
        img = ee.Image(image_list.get(i)).select('EVI').updateMask(ma_forest)
        tmp_path = os.path.join(output_dir, f'_tmp_{year}_{i:03d}.tif')

        geemap.ee_export_image(
            img,
            filename=tmp_path,
            scale=30,
            region=route_buffer,
            file_per_band=False
        )

        with rasterio.open(tmp_path) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            arrays.append(data)
            if profile is None:
                profile = src.profile
                # Save reference GeoTIFF for spatial metadata
                with rasterio.open(ref_path, 'w', **profile) as dst:
                    dst.write(src.read(1), 1)

        os.remove(tmp_path)
        print(f'    {i + 1}/{n}', end='\r')

    print()
    arr = np.stack(arrays, axis=0)  # (n_images, height, width)
    np.save(stack_path, arr)
    np.save(doys_path, doys)

    return arr, doys, ref_path


# ----------------------------
# Per-year fitting
# ----------------------------

def compute_transition_dates(hls_evi_collection, route_buffer, ma_forest, year, output_dir='.'):
    """
    Export Jul-Dec EVI per image from GEE, fit a decreasing logistic per forest
    pixel, and save three GeoTIFFs: start, middle, end of greendown (as DOY).

    Results are cached — re-running skips already-completed years.
    Returns a dict: {'start': path, 'middle': path, 'end': path}
    """
    paths = {
        phase: os.path.join(output_dir, f'greendown_{phase}_{year}.tif')
        for phase in ('start', 'middle', 'end')
    }

    if all(os.path.exists(p) for p in paths.values()):
        print(f'  Using cached results for {year}')
        return paths

    collection = hls_evi_collection.sort('system:time_start')
    arr, doys, ref_path = _export_stack(collection, ma_forest, route_buffer, year, output_dir)

    _, h, w = arr.shape
    t_start  = np.full((h, w), np.nan, dtype=np.float32)
    t_middle = np.full((h, w), np.nan, dtype=np.float32)
    t_end    = np.full((h, w), np.nan, dtype=np.float32)

    print(f'  Fitting logistic curves for {year}...')
    for i in range(h):
        for j in range(w):
            popt = _fit_pixel(doys, arr[:, i, j])
            if popt is not None:
                _, k, t_m, _ = popt
                s, m, e = _curvature_extrema_doys(k, t_m)
                t_start[i, j]  = s
                t_middle[i, j] = m
                t_end[i, j]    = e

    with rasterio.open(ref_path) as src:
        out_profile = src.profile.copy()
    out_profile.update(count=1, dtype='float32', nodata=-9999.0)

    for phase, data in [('start', t_start), ('middle', t_middle), ('end', t_end)]:
        data[np.isnan(data)] = -9999.0
        with rasterio.open(paths[phase], 'w', **out_profile) as dst:
            dst.write(data, 1)

    print(f'  Done: {year}')
    return paths


# ----------------------------
# Multi-year average
# ----------------------------

def compute_average_transition_dates(paths_by_year, output_dir='.'):
    """
    Pixel-wise mean of start/middle/end transition GeoTIFFs across years.

    paths_by_year: list of dicts returned by compute_transition_dates.
    Returns dict: {'start': path, 'middle': path, 'end': path}
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
        mean[np.isnan(mean)] = -9999.0

        path = os.path.join(output_dir, f'greendown_{phase}_avg.tif')
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(mean, 1)
        avg_paths[phase] = path

    return avg_paths
