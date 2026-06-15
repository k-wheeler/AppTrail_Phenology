import ee


def _get_at_route():
    return ee.FeatureCollection("projects/turnkey-lacing-391919/assets/AT_Trail")


def _get_ma_boundary():
    return ee.FeatureCollection('TIGER/2018/States') \
        .filter(ee.Filter.inList('NAME', ['Massachusetts']))


def _compute_ma_route():
    at_route = _get_at_route()
    ma_boundary = _get_ma_boundary()
    return at_route.geometry().intersection(
        ma_boundary.geometry(),
        ee.ErrorMargin(1)
    )


def _compute_route_buffer():
    return _compute_ma_route().simplify(10).buffer(50)


def _compute_forest_mask(route_buffer):
    nlcd = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD") \
        .filter(ee.Filter.eq('system:index', '2021')) \
        .first()
    landcover = nlcd.select('landcover')
    forest = landcover.eq(41).Or(landcover.eq(43))
    return forest.clip(route_buffer)


def identify_route_buffer():
    return _compute_route_buffer()


def identify_forests():
    route_buffer = _compute_route_buffer()
    return _compute_forest_mask(route_buffer)


def identify_maroute():
    return _compute_ma_route()
