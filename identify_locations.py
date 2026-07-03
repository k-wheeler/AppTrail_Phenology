import ee


def _get_at_route():
    """Load the full Appalachian Trail route as a GEE FeatureCollection."""
    return ee.FeatureCollection("projects/turnkey-lacing-391919/assets/AT_Trail")


def _get_ma_boundary():
    """Load the Massachusetts state boundary from the TIGER dataset."""
    return ee.FeatureCollection('TIGER/2018/States') \
        .filter(ee.Filter.inList('NAME', ['Massachusetts']))


def _compute_ma_route():
    """Clip the AT route to the Massachusetts boundary and return the geometry."""
    at_route = _get_at_route()
    ma_boundary = _get_ma_boundary()
    return at_route.geometry().intersection(
        ma_boundary.geometry(),
        ee.ErrorMargin(1)
    )


def _compute_route_buffer(buffer_m=50):
    """Return a buffer around the simplified MA AT route geometry.

    Args:
        buffer_m: Buffer radius in metres. Default 50 (serving). Use 100 for training.
    """
    return _compute_ma_route().simplify(10).buffer(buffer_m)


def _compute_forest_mask(route_buffer):
    """Create a binary forest mask clipped to the route buffer.

    Uses NLCD 2021 land cover classes 41 (deciduous) and 43 (mixed forest).

    Args:
        route_buffer: GEE Geometry defining the clipping boundary.

    Returns:
        GEE Image with 1 for forest pixels and 0 elsewhere, clipped to the buffer.
    """
    nlcd = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD") \
        .filter(ee.Filter.eq('system:index', '2021')) \
        .first()
    landcover = nlcd.select('landcover')
    forest = landcover.eq(41).Or(landcover.eq(43))
    return forest.clip(route_buffer)


def identify_route_buffer(buffer_m=50):
    """Return the buffered MA AT route geometry for use as a spatial filter.

    Args:
        buffer_m: Buffer radius in metres. Default 50 (serving). Use 100 for training.
    """
    return _compute_route_buffer(buffer_m)


def identify_forests(buffer_m=50):
    """Return a binary GEE Image masking deciduous and mixed forest pixels within the route buffer.

    Args:
        buffer_m: Buffer radius in metres. Default 50 (serving). Use 100 for training.
    """
    route_buffer = _compute_route_buffer(buffer_m)
    return _compute_forest_mask(route_buffer)


def identify_maroute():
    """Return the AT route geometry clipped to Massachusetts."""
    return _compute_ma_route()
