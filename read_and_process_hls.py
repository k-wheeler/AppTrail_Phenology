import ee


def _add_indices(img):
    """Compute EVI and NDVI from HLS bands and return a two-band image clamped to [0, 1]."""
    evi = img.expression(
        '2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))',
        {'nir': img.select('B5'), 'red': img.select('B4'), 'blue': img.select('B2')}
    ).rename('EVI').clamp(0, 1)
    ndvi = img.expression(
        '(nir - red) / (nir + red)',
        {'nir': img.select('B5'), 'red': img.select('B4')}
    ).rename('NDVI').clamp(0, 1)
    return evi.addBands(ndvi).copyProperties(img, ['system:time_start'])


def compute_hls_indices(route_buffer, ma_forest, year):
    """Filter the HLS Landsat 30m collection and compute EVI and NDVI.

    Args:
        route_buffer: GEE Geometry defining the spatial filter.
        ma_forest: GEE Image binary forest mask (unused in filtering but passed
            for API consistency).
        year: Integer year; images from Jul 1 to Dec 31 are included.

    Returns:
        GEE ImageCollection with EVI and NDVI bands for Jul–Dec of the given year.
    """
    return (
        ee.ImageCollection("NASA/HLS/HLSL30/v002")
        .filterBounds(route_buffer)
        .filterDate(f'{year}-07-01', f'{year}-12-31')
        .select(['B5', 'B4', 'B2'])
        .map(_add_indices)
    )
