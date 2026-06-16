import datetime
import os
import ee
import geemap

from identify_locations import identify_forests, identify_route_buffer
from read_and_process_hls import compute_hls_evi
from fit_greendown_curves import compute_transition_dates, compute_average_transition_dates

ee.Initialize(project='turnkey-lacing-391919')

OUTPUT_DIR = './greendown_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

ma_forest    = identify_forests()
route_buffer = identify_route_buffer()

previous_year = datetime.datetime.now().year - 1
start_year    = max(previous_year - 14, 2013)  # HLS data begins 2013

# ----------------------------
# Fit logistic curves for each year
# ----------------------------
all_year_paths = []
for y in range(start_year, previous_year + 1):
    print(f'Processing {y}...')
    hls   = compute_hls_evi(route_buffer, ma_forest, y)
    paths = compute_transition_dates(hls, route_buffer, ma_forest, y, output_dir=OUTPUT_DIR)
    all_year_paths.append(paths)

prev_year_paths = all_year_paths[-1]   # most recent year

# ----------------------------
# Average transition dates across all years
# ----------------------------
print('Computing averages...')
avg_paths = compute_average_transition_dates(all_year_paths, output_dir=OUTPUT_DIR)

# ----------------------------
# Map all 6 layers
# ----------------------------
vis_kwargs = dict(colormap='RdYlGn_r', vmin=250, vmax=320, nodata=-9999.0, opacity=0.9)

Map = geemap.Map()

# Most recent year
Map.add_raster(prev_year_paths['start'],  layer_name=f'{previous_year} Greendown Start',  **vis_kwargs)
Map.add_raster(prev_year_paths['middle'], layer_name=f'{previous_year} Greendown Middle', **vis_kwargs)
Map.add_raster(prev_year_paths['end'],    layer_name=f'{previous_year} Greendown End',    **vis_kwargs)

# Multi-year averages
Map.add_raster(avg_paths['start'],  layer_name=f'{start_year}–{previous_year} Avg Start',  **vis_kwargs)
Map.add_raster(avg_paths['middle'], layer_name=f'{start_year}–{previous_year} Avg Middle', **vis_kwargs)
Map.add_raster(avg_paths['end'],    layer_name=f'{start_year}–{previous_year} Avg End',    **vis_kwargs)

Map.add_colorbar(
    {'min': 250, 'max': 320, 'palette': ['darkgreen', 'yellow', 'orange', 'red']},
    label='Greendown Transition DOY  (250 ≈ Sep 7  →  320 ≈ Nov 16)',
    orientation='horizontal'
)

Map
