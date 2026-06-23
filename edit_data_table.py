import glob
import re
import numpy as np
import rasterio

NODATA = -9999.0
GAP_FILL_MAX_CI_WIDTH = 14.0  # days; threshold for pixels included in global average


def _compute_global_avg_middle(output_dir):
    """
    Compute the mean middle transition DOY across all year-pixel combinations
    where all three CI widths are <= GAP_FILL_MAX_CI_WIDTH days.
    Returns a scalar float (or NaN if no qualifying pixels found).
    """
    phases = ('start', 'middle', 'end')
    pattern = re.compile(r'greendown_start_(\d{4})\.tif')
    available_years = sorted(
        int(m.group(1))
        for p in glob.glob(f'{output_dir}/greendown_start_*.tif')
        if (m := pattern.search(p.split('/')[-1]))
    )

    all_middle_vals = []
    for year in available_years:
        required = (
            [f'{output_dir}/greendown_{p}_{year}.tif' for p in phases] +
            [f'{output_dir}/greendown_{p}_ci_width_{year}.tif' for p in phases]
        )
        if not all(__import__('os').path.exists(f) for f in required):
            continue

        ci_mask = None
        for phase in phases:
            with rasterio.open(f'{output_dir}/greendown_{phase}_ci_width_{year}.tif') as src:
                w = src.read(1).astype(float)
                w[w == NODATA] = np.nan
            phase_ok = np.isfinite(w) & (w <= GAP_FILL_MAX_CI_WIDTH)
            ci_mask = phase_ok if ci_mask is None else (ci_mask & phase_ok)

        with rasterio.open(f'{output_dir}/greendown_middle_{year}.tif') as src:
            middle = src.read(1).astype(float)
            middle[middle == NODATA] = np.nan

        qualifying = middle[ci_mask & np.isfinite(middle)]
        all_middle_vals.extend(qualifying.tolist())

    return float(np.mean(all_middle_vals)) if all_middle_vals else float('nan')


def _gap_fill_doy_minus_avg_middle(feature_df, output_dir):
    """
    Fill NaN values in doy_minus_avg_middle with (row's doy) minus the global
    average middle transition DOY across all year-pixel combinations with all
    three CI widths <= 14 days.
    """
    nan_mask = feature_df['doy_minus_avg_middle'].isna()
    if not nan_mask.any():
        return feature_df

    global_avg_middle = _compute_global_avg_middle(output_dir)
    if np.isnan(global_avg_middle):
        print('  Warning: no qualifying pixels found for gap-fill average; doy_minus_avg_middle left as NaN')
        return feature_df

    print(f'  Gap-filling {nan_mask.sum()} NaN values in doy_minus_avg_middle '
          f'(global avg middle DOY = {global_avg_middle:.1f})')
    feature_df = feature_df.copy()
    feature_df.loc[nan_mask, 'doy_minus_avg_middle'] = (
        feature_df.loc[nan_mask, 'doy'] - global_avg_middle
    )
    return feature_df


def edit_feature_table(feature_df, output_dir):
    #Imputation to fill in missing data (pixels that never have reliable transition date estimates)
    feature_df = _gap_fill_doy_minus_avg_middle(feature_df, output_dir)

    #These NaNs occur at the start of years where there aren't previous indices to compare to in the data set
    #Because these are not really needed (don't need to check for senescence at the very beginning of july) drop instead of gap fill
    feature_df = feature_df.dropna(subset=['evi_delta', 'evi_delta2', 'ndvi_delta', 'ndvi_delta2'])

    #Remove columns not needed for models
    feature_df = feature_df.drop(columns=['doy', 'doy_minus_avg_start', 'doy_minus_avg_end', 'year', 'date'])

    #TODO: Think about outliers

    #Z-score normalize each column separately
    numeric_cols = feature_df.select_dtypes(include='number').columns
    feature_df[numeric_cols] = (
        (feature_df[numeric_cols] - feature_df[numeric_cols].mean())
        / feature_df[numeric_cols].std()
    )

    #TODO: Think about class distribution
    return feature_df
