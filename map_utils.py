import numpy as np
import matplotlib.colors as mcolors
from pyproj import Transformer
import rasterio.transform as rio_transform

from constants import LABEL_COLORS


def _pred_grid_to_rgba(pred_grid, forest_mask, opacity=0.65):
    """Convert a label grid to an RGBA image array.

    Args:
        pred_grid: 2D array of label strings with shape (h, w).
        forest_mask: Boolean array of shape (h, w); non-forest pixels are transparent.
        opacity: Alpha channel value in [0, 1] for forest pixels.

    Returns:
        RGBA uint8 array of shape (h, w, 4).
    """
    h, w = pred_grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for label, hex_color in LABEL_COLORS.items():
        r, g, b, _ = mcolors.to_rgba(hex_color)
        mask = (pred_grid == label) & forest_mask
        rgba[mask] = [int(r * 255), int(g * 255), int(b * 255), int(opacity * 255)]
    return rgba


def _get_wgs84_bounds(transform, crs, h, w):
    """Return the WGS84 bounding box of a raster.

    Args:
        transform: Affine transform of the raster.
        crs: Coordinate reference system of the raster.
        h: Raster height in pixels.
        w: Raster width in pixels.

    Returns:
        List [[south, west], [north, east]] in WGS84 decimal degrees.
    """
    xs, ys = rio_transform.xy(transform,
                               [0, 0, h - 1, h - 1],
                               [0, w - 1, 0, w - 1])
    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    return [[min(lats), min(lons)], [max(lats), max(lons)]]
