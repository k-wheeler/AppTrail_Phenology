import os
import numpy as np
import rasterio

from constants import NODATA, MAX_CI_WIDTH


def load_ci_widths(greendown_dir, years):
    """Load CI width GeoTIFFs for all years and phases.

    Args:
        greendown_dir: Path to directory containing CI width GeoTIFFs.
        years: Iterable of integer years to load.

    Returns:
        Dict of {year: {'start': array, 'middle': array, 'end': array}}.
        Arrays are float with NaN for nodata or missing files.
    """
    phases = ('start', 'middle', 'end')
    widths = {}
    for year in years:
        year_widths = {}
        for phase in phases:
            path = os.path.join(greendown_dir, f'greendown_{phase}_ci_width_{year}.tif')
            if not os.path.exists(path):
                year_widths[phase] = None
            else:
                with rasterio.open(path) as src:
                    data = src.read(1).astype(float)
                    data[data == NODATA] = np.nan
                    year_widths[phase] = data
        widths[year] = year_widths
    return widths


def count_narrow_ci_pixel_years(greendown_dir, years, max_width=MAX_CI_WIDTH):
    """Count pixel-year combinations where all three CI widths are < max_width days.

    Prints a per-year and aggregate summary.

    Args:
        greendown_dir: Path to directory containing CI width GeoTIFFs.
        years: Iterable of integer years to evaluate.
        max_width: CI width threshold in days. Defaults to MAX_CI_WIDTH.

    Returns:
        Integer count of qualifying pixel-year combinations.
    """
    widths = load_ci_widths(greendown_dir, years)
    phases = ('start', 'middle', 'end')

    phase_totals = {p: 0 for p in phases}
    total = 0

    for year in years:
        year_widths = widths[year]
        if any(year_widths[p] is None for p in phases):
            print(f'  {year}: skipped (missing CI width GeoTIFF)')
            continue

        arrays = {p: year_widths[p] for p in phases}
        phase_counts = {p: int((np.isfinite(arrays[p]) & (arrays[p] < max_width)).sum())
                        for p in phases}

        # Pixel must have valid, narrow CI for all three phases
        all_valid = np.ones(list(arrays.values())[0].shape, dtype=bool)
        for arr in arrays.values():
            all_valid &= np.isfinite(arr) & (arr < max_width)
        count = int(all_valid.sum())

        total += count
        for p in phases:
            phase_totals[p] += phase_counts[p]

        print(f'  {year}: {count} pixels (all phases)  |  '
              f'start: {phase_counts["start"]}  '
              f'middle: {phase_counts["middle"]}  '
              f'end: {phase_counts["end"]}')

    print(f'\nTotal pixel-years with all CI widths < {max_width} days: {total}')
    print(f'  By phase — start: {phase_totals["start"]}  '
          f'middle: {phase_totals["middle"]}  '
          f'end: {phase_totals["end"]}')
    return total
