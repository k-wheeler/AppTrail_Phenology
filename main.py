import datetime
import ee
import geemap

from identify_locations import identify_forests, identify_route_buffer
from read_and_process_hls import compute_hls_evi
from calculate_greendown_timing import compute_greendown_date

ee.Initialize(project='turnkey-lacing-391919')

ma_forest    = identify_forests()
route_buffer = identify_route_buffer()

previous_year = datetime.datetime.now().year - 1
start_year    = max(previous_year - 14, 2013)  # HLS data begins 2013

# Previous year greendown
hls_evi_prev       = compute_hls_evi(route_buffer, ma_forest, previous_year)
greendown_prev_year = compute_greendown_date(hls_evi_prev, route_buffer, previous_year)

# 15-year average greendown
greendown_images = [
    compute_greendown_date(
        compute_hls_evi(route_buffer, ma_forest, y),
        route_buffer,
        y
    )
    for y in range(start_year, previous_year + 1)
]

greendown_avg = (
    ee.ImageCollection(greendown_images)
    .mean()
    .rename('Greendown_DOY_Avg')
    .clip(route_buffer)
)

# Visualization
vis_params = {
    'min': 260,
    'max': 310,
    'palette': ['darkgreen', 'yellow', 'orange', 'red']
}

Map = geemap.Map()
Map.addLayer(greendown_avg,      vis_params, f'{start_year}–{previous_year} Avg Greendown DOY')
Map.addLayer(greendown_prev_year, vis_params, f'{previous_year} Greendown DOY')

Map.add_colorbar(
    vis_params,
    label='Greendown DOY (260 = ~Sep 17, 310 = ~Nov 6)',
    orientation='horizontal'
)

Map
