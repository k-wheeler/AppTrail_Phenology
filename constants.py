"""Shared constants used across the phenology pipeline."""

# Raster nodata value used in all GeoTIFF outputs
NODATA = -9999.0

# Output directory for all GeoTIFFs, numpy stacks, and model files
OUTPUT_DIR = './greendown_outputs'

# CI width thresholds (days)
MAX_CI_WIDTH = 15             # pixels must have all CI widths < this to enter the feature table
CROSS_YEAR_MAX_CI_WIDTH = 30  # threshold for including a pixel-year in cross-year DOY averages
GAP_FILL_MAX_CI_WIDTH = 14    # threshold for pixels used to compute the global avg middle DOY gap-fill

# Phenological stage labels, ordered from earliest to latest
LABEL_ORDER = ['before', 'early', 'late', 'after', 'unknown']

# Display colors for each phenological stage
LABEL_COLORS = {
    'before':  'steelblue',
    'early':   'green',
    'late':    'orange',
    'after':   'red',
    'unknown': 'lightgray',
}
