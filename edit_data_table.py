import glob
import json
import os
import re
import numpy as np
import rasterio

from constants import NODATA, GAP_FILL_MAX_CI_WIDTH


def _compute_global_avg_middle(output_dir):
    """Compute the mean middle transition DOY across qualifying pixel-years.

    A pixel-year qualifies if all three CI widths are <= GAP_FILL_MAX_CI_WIDTH days.

    Args:
        output_dir: Path to directory containing transition and CI width GeoTIFFs.

    Returns:
        Mean middle transition DOY as a float, or NaN if no qualifying pixels found.
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
    """Gap-fill NaN values in the doy_minus_avg_middle column.

    Fills using (row's doy) minus the global average middle transition DOY,
    computed from pixel-years where all three CI widths are <= GAP_FILL_MAX_CI_WIDTH.

    Args:
        feature_df: DataFrame containing doy and doy_minus_avg_middle columns.
        output_dir: Path to directory containing transition and CI width GeoTIFFs.

    Returns:
        DataFrame with NaN values in doy_minus_avg_middle filled where possible.
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


def _balance_classes(feature_df):
    """Undersample majority label classes to match the size of the smallest class.

    Args:
        feature_df: DataFrame with a 'label' column.

    Returns:
        Balanced DataFrame with equal representation of each label.
    """
    min_count = feature_df['label'].value_counts().min()
    return (feature_df
            .groupby('label', group_keys=False)
            .apply(lambda g: g.sample(min_count, random_state=42)))


def edit_feature_table(feature_df, output_dir):
    """Prepare the feature table for model training.

    Gap-fills doy_minus_avg_middle, drops rows with NaN deltas, removes unused
    columns, balances classes by undersampling, and z-score normalizes numeric
    columns. Saves normalization statistics to {output_dir}/norm_stats.json.

    Args:
        feature_df: DataFrame produced by build_feature_table.
        output_dir: Path to directory for reading GeoTIFFs and writing norm_stats.json.

    Returns:
        Cleaned, balanced, and normalized DataFrame ready for model training.
    """
    #Imputation to fill in missing data (pixels that never have reliable transition date estimates)
    feature_df = _gap_fill_doy_minus_avg_middle(feature_df, output_dir)

    # Fill mode_label_7day NaNs with 0 ('before'): occurs when no prior observation
    # exists within the 7-day window (e.g. sparse imagery early in the season).
    if 'mode_label_7day' in feature_df.columns:
        feature_df['mode_label_7day'] = feature_df['mode_label_7day'].fillna(0.0)

    # Fill cdd_accumulated NaNs with 0: occurs for dates before Jul 1 or when
    # gridMET data was not downloaded for a given training year.
    if 'cdd_accumulated' in feature_df.columns:
        feature_df['cdd_accumulated'] = feature_df['cdd_accumulated'].fillna(0.0)

    # Fill tmean_recent NaNs with column mean: temperature can be negative so 0
    # is not a safe default; mean is a neutral imputation before z-scoring.
    if 'tmean_recent' in feature_df.columns:
        feature_df['tmean_recent'] = feature_df['tmean_recent'].fillna(
            feature_df['tmean_recent'].mean()
        )

    #These NaNs occur at the start of years where there aren't previous indices to compare to in the data set
    #Because these are not really needed (don't need to check for senescence at the very beginning of july) drop instead of gap fill
    feature_df = feature_df.dropna(subset=['evi_delta', 'evi_delta2', 'ndvi_delta', 'ndvi_delta2'])

    #Remove columns not needed for models
    feature_df = feature_df.drop(columns=['doy', 'doy_minus_avg_start', 'doy_minus_avg_end', 'year', 'date'])

    #TODO: Think about outliers (skipping for now because going to build a decision tree first and those are less sensitive to outliers)

    #Undersample majority classes to match the size of the smallest class
    labels = feature_df['label']
    feature_df = _balance_classes(feature_df)

    #Z-score normalize each column separately
    numeric_cols = feature_df.select_dtypes(include='number').columns
    col_means = feature_df[numeric_cols].mean()
    col_stds  = feature_df[numeric_cols].std()
    feature_df[numeric_cols] = (feature_df[numeric_cols] - col_means) / col_stds
    feature_df['label'] = labels.loc[feature_df.index]

    # Save normalization statistics so prediction code can apply the same scaling
    norm_stats = {col: {'mean': float(col_means[col]), 'std': float(col_stds[col])}
                  for col in numeric_cols}
    with open(os.path.join(output_dir, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)

    return feature_df
