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
# Per-year fitting
# ----------------------------

def compute_transition_dates(hls_evi_collection, route_buffer, ma_forest, year, output_dir='.'):
    """
    Export Jul-Dec EVI stack from GEE, fit a decreasing logistic per forest
    pixel, and save three GeoTIFFs representing the DOY of the start, middle,
    and end of the greendown transition (extrema of d(curvature)/dt).

    Returns a dict: {'start': path, 'middle': path, 'end': path}
    """
    collection = hls_evi_collection.sort('system:time_start')

    # Get DOY for each image
    timestamps = collection.aggregate_array('system:time_start').getInfo()
    doys = np.array([
        datetime.datetime.utcfromtimestamp(ts / 1000).timetuple().tm_yday
        for ts in timestamps
    ])

    if len(doys) < 4:
        raise ValueError(f'Too few images ({len(doys)}) for {year} — cannot fit logistic')

    # Export masked EVI stack to a local GeoTIFF
    stacked = collection.toBands().updateMask(ma_forest)
    stack_path = os.path.join(output_dir, f'hls_evi_stack_{year}.tif')
    print(f'  Exporting EVI stack for {year} ({len(doys)} images)...')
    geemap.ee_export_image(
        stacked,
        filename=stack_path,
        scale=30,
        region=route_buffer,
        file_per_band=False
    )

    # Read stack
    with rasterio.open(stack_path) as src:
        arr = src.read().astype(float)   # (n_bands, height, width)
        nodata = src.nodata
        profile = src.profile

    if nodata is not None:
        arr[arr == nodata] = np.nan

    n_bands, h, w = arr.shape
    t_start  = np.full((h, w), np.nan, dtype=np.float32)
    t_middle = np.full((h, w), np.nan, dtype=np.float32)
    t_end    = np.full((h, w), np.nan, dtype=np.float32)

    print(f'  Fitting logistic curves for {year} ({h}x{w} pixels)...')
    for i in range(h):
        for j in range(w):
            popt = _fit_pixel(doys, arr[:, i, j])
            if popt is not None:
                _, k, t_m, _ = popt
                s, m, e = _curvature_extrema_doys(k, t_m)
                t_start[i, j]  = s
                t_middle[i, j] = m
                t_end[i, j]    = e

    # Save result GeoTIFFs
    out_profile = profile.copy()
    out_profile.update(count=1, dtype='float32', nodata=-9999.0)

    paths = {}
    for name, data in [('start', t_start), ('middle', t_middle), ('end', t_end)]:
        data[np.isnan(data)] = -9999.0
        path = os.path.join(output_dir, f'greendown_{name}_{year}.tif')
        with rasterio.open(path, 'w', **out_profile) as dst:
            dst.write(data, 1)
        paths[name] = path

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
        arrays = []
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
