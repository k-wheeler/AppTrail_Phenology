"""Daily script run by the GitHub Action to generate static web outputs.

Usage:
    python generate_web_outputs.py \
        --output-dir ./greendown_outputs \
        --web-dir ./web_outputs

Authenticates with GEE via the GEE_SERVICE_ACCOUNT_KEY environment variable
(JSON key file contents), downloads any new HLS images for the current year,
runs the phenology prediction for today, and writes a self-contained HTML page
plus supporting assets to --web-dir for publishing on GitHub Pages.
"""

import argparse
import base64
import datetime
import io
import json
import os

import ee
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

from constants import OUTPUT_DIR, LABEL_COLORS, LABEL_ORDER
from map_utils import _pred_grid_to_rgba, _get_wgs84_bounds
from fit_greendown_curves import update_pixel_state
from identify_locations import identify_route_buffer, identify_forests
from predict_for_date import predict_from_pixel_state


# ---------------------------------------------------------------------------
# GEE authentication
# ---------------------------------------------------------------------------

def _init_gee():
    """Authenticate with GEE using the service account key in the environment."""
    key_json = os.environ.get('GEE_SERVICE_ACCOUNT_KEY')
    if key_json:
        key = json.loads(key_json)
        creds = ee.ServiceAccountCredentials(key['client_email'],
                                             key_data=json.dumps(key))
        ee.Initialize(creds)
        print('GEE authenticated via service account.')
    else:
        # Fall back to interactive auth for local testing
        ee.Authenticate()
        ee.Initialize()
        print('GEE authenticated interactively.')


# ---------------------------------------------------------------------------
# Average transition DOY maps
# ---------------------------------------------------------------------------

def _render_avg_doy_png(tif_path, out_path, title, vmin=240, vmax=320):
    """Render an average transition DOY GeoTIFF to a coloured RGBA PNG.

    Args:
        tif_path: Path to the GeoTIFF with DOY values (NODATA = -9999).
        out_path: Path to write the output PNG.
        title: Title string for the colourbar image (saved separately as a
            legend PNG next to the map).
        vmin: Minimum DOY for the colour scale.
        vmax: Maximum DOY for the colour scale.
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan

    cmap = plt.get_cmap('RdYlGn_r')
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(data), bytes=True)          # (h, w, 4) uint8
    rgba[np.isnan(data)] = [0, 0, 0, 0]         # transparent where nodata

    Image.fromarray(rgba, mode='RGBA').save(out_path)
    print(f'  Saved {out_path}')

    # Save a colourbar legend as a separate PNG with calendar date tick labels
    legend_path = out_path.replace('.png', '_legend.png')
    fig, ax = plt.subplots(figsize=(5, 0.6))
    fig.subplots_adjust(bottom=0.5)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cb.set_label(title)

    # Pick ~5 evenly spaced DOY ticks and label them as calendar dates (non-leap year)
    tick_doys = np.linspace(vmin, vmax, 5).astype(int)
    tick_labels = [
        (datetime.date(2001, 1, 1) + datetime.timedelta(days=int(d) - 1)).strftime('%b %-d')
        for d in tick_doys
    ]
    cb.set_ticks(tick_doys)
    cb.set_ticklabels(tick_labels)

    plt.savefig(legend_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f'  Saved {legend_path}')


def _ensure_avg_pngs(output_dir, web_dir):
    """Generate average transition DOY PNGs if they don't already exist.

    Args:
        output_dir: Directory containing greendown_*_avg.tif files.
        web_dir: Directory to write output PNGs.
    """
    phases = {
        'start':  'Avg Greendown Start',
        'middle': 'Avg Greendown Middle',
        'end':    'Avg Greendown End',
    }
    for phase, title in phases.items():
        tif_path = os.path.join(output_dir, f'greendown_{phase}_avg.tif')
        out_path = os.path.join(web_dir, f'avg_{phase}.png')
        if not os.path.exists(tif_path):
            print(f'  Skipping avg {phase} PNG — GeoTIFF not found.')
            continue
        _render_avg_doy_png(tif_path, out_path, title)


# ---------------------------------------------------------------------------
# PNG bounds helper
# ---------------------------------------------------------------------------

def _bounds_to_leaflet(transform, crs, h, w):
    """Return Leaflet-style bounds [[south, west], [north, east]].

    Args:
        transform: Affine transform of the raster.
        crs: CRS of the raster.
        h: Raster height in pixels.
        w: Raster width in pixels.

    Returns:
        List [[south, west], [north, east]] in WGS84 decimal degrees.
    """
    return _get_wgs84_bounds(transform, crs, h, w)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _render_html(web_dir, meta):
    """Render the static index.html page for the GitHub Pages site.

    Args:
        web_dir: Directory to write index.html.
        meta: Dict loaded from current_meta.json.
    """
    date_str = meta['date']
    bounds   = meta['bounds']           # [[s, w], [n, e]]
    areas    = meta.get('areas_sqmi', {})
    center   = [(bounds[0][0] + bounds[1][0]) / 2,
                (bounds[0][1] + bounds[1][1]) / 2]

    # Build legend HTML rows
    legend_rows = ''.join(
        f'<span class="swatch" style="background:{LABEL_COLORS[l]}"></span>'
        f'<span>{l.capitalize()}</span>'
        for l in LABEL_ORDER if l != 'unknown'
    )

    # Build histogram bars
    total_area = sum(v for k, v in areas.items() if k != 'unknown') or 1
    hist_bars = ''.join(
        f'<div class="bar-wrap">'
        f'<div class="bar" style="height:{int(areas.get(l,0)/total_area*160)}px;'
        f'background:{LABEL_COLORS[l]}"></div>'
        f'<div class="bar-label">{l.capitalize()}<br>{areas.get(l,0):.1f} mi²</div>'
        f'</div>'
        for l in LABEL_ORDER if l != 'unknown'
    )

    # Check which avg maps are available
    avg_tabs = ''
    for phase, title in [('start', 'Avg Start'), ('middle', 'Avg Middle'), ('end', 'Avg End')]:
        if os.path.exists(os.path.join(web_dir, f'avg_{phase}.png')):
            avg_tabs += f'''
            <div class="avg-panel">
              <h3>{title}</h3>
              <div id="map-{phase}" class="avg-map"></div>
              <img src="avg_{phase}_legend.png" class="legend-img">
            </div>'''

    avg_js = ''
    for phase in ('start', 'middle', 'end'):
        if os.path.exists(os.path.join(web_dir, f'avg_{phase}.png')):
            avg_js += f'''
            window['map{phase}'] = L.map('map-{phase}', {{zoomControl: false}})
                .setView({center}, 10);
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
                {{attribution: '&copy; OpenStreetMap &copy; CARTO'}}).addTo(window['map{phase}']);
            L.imageOverlay('avg_{phase}.png', {bounds}, {{opacity: 0.85}}).addTo(window['map{phase}']);
            '''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AT Phenology — Massachusetts</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: sans-serif; background: #f5f5f5; color: #222; }}
  header {{ background: #2c6b3f; color: white; padding: 14px 20px; }}
  header h1 {{ font-size: 1.3rem; }}
  header p  {{ font-size: 0.85rem; opacity: 0.85; }}
  .tabs {{ display: flex; gap: 4px; padding: 12px 20px 0; background: #fff;
            border-bottom: 2px solid #ddd; }}
  .tab  {{ padding: 8px 18px; cursor: pointer; border-radius: 4px 4px 0 0;
            background: #eee; font-size: 0.9rem; }}
  .tab.active {{ background: #2c6b3f; color: white; }}
  .panel {{ display: none; padding: 20px; }}
  .panel.active {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  #map-today {{ width: 620px; height: 480px; border-radius: 6px;
                border: 1px solid #ccc; }}
  .sidebar {{ display: flex; flex-direction: column; gap: 16px; }}
  .legend {{ background: white; border: 1px solid #ccc; border-radius: 6px;
              padding: 12px 16px; }}
  .legend h3 {{ margin-bottom: 8px; font-size: 0.9rem; color: #444; }}
  .swatch {{ display: inline-block; width: 14px; height: 14px;
              border-radius: 2px; margin-right: 8px; vertical-align: middle; }}
  .legend span {{ font-size: 0.88rem; }}
  .legend > * {{ display: block; margin-bottom: 4px; }}
  .histogram {{ display: flex; align-items: flex-end; gap: 8px;
                background: white; border: 1px solid #ccc; border-radius: 6px;
                padding: 16px; }}
  .bar-wrap {{ text-align: center; }}
  .bar {{ width: 44px; border-radius: 3px 3px 0 0; min-height: 2px; }}
  .bar-label {{ font-size: 0.75rem; margin-top: 4px; color: #555; }}
  .avg-map {{ width: 380px; height: 320px; border-radius: 6px;
              border: 1px solid #ccc; }}
  .avg-panel {{ display: flex; flex-direction: column; gap: 8px; }}
  .avg-panel h3 {{ font-size: 0.95rem; color: #444; }}
  .legend-img {{ max-width: 300px; border-radius: 4px; }}
  .leaflet-image-layer {{ image-rendering: pixelated; }}
</style>
</head>
<body>
<header>
  <h1>Appalachian Trail Fall Phenology — Massachusetts</h1>
  <p>Prediction for <strong>{date_str}</strong> &nbsp;|&nbsp;
     Updated daily using NASA HLS 30 m satellite imagery</p>
</header>
<div class="tabs">
  <div class="tab active" onclick="showTab('today', this)">Today's Prediction</div>
  <div class="tab" onclick="showTab('history', this)">Historical Averages</div>
</div>
<div id="panel-today" class="panel active">
  <div id="map-today"></div>
  <div class="sidebar">
    <div class="legend">
      <h3>Phenological State</h3>
      {legend_rows}
    </div>
    <div class="histogram">
      {hist_bars}
    </div>
  </div>
</div>
<div id="panel-history" class="panel">
  {avg_tabs if avg_tabs else '<p style="color:#777">Average transition maps not yet available.</p>'}
</div>
<script>
function showTab(name, el) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  el.classList.add('active');
  // Force Leaflet to recalculate map size after the container becomes visible
  if (name === 'history') {{
    ['start', 'middle', 'end'].forEach(function(p) {{
      var m = window['map' + p];
      if (m) {{ m.invalidateSize(); }}
    }});
  }}
}}

// Today's prediction map
var mapToday = L.map('map-today').setView({center}, 10);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{attribution: '&copy; OpenStreetMap contributors &copy; CARTO'}}).addTo(mapToday);
L.imageOverlay('current_pred.png', {bounds}, {{opacity: 1.0}}).addTo(mapToday);

// Pixel-click popup: each pixel stores its WGS84 centre lat/lon so lookup
// is a simple nearest-neighbour search — no coordinate transform needed.
var pixelArr = null;
var FEAT_LABELS = {{
  'EVI':                  'EVI',
  'NDVI':                 'NDVI',
  'evi_delta':            'EVI Δ1 (vs prior obs)',
  'evi_delta2':           'EVI Δ2 (vs 2nd prior obs)',
  'ndvi_delta':           'NDVI Δ1 (vs prior obs)',
  'ndvi_delta2':          'NDVI Δ2 (vs 2nd prior obs)',
  'day_length_hrs':       'Day length (hrs)',
  'doy_minus_avg_middle': 'Days from avg mid-transition'
}};
var LABEL_COLORS = {json.dumps(LABEL_COLORS)};

fetch('pixel_features.json')
  .then(function(r) {{ return r.json(); }})
  .then(function(d) {{
    pixelArr = d.pixels;
    console.log('pixel_features.json loaded. type:', Object.prototype.toString.call(pixelArr),
                'length:', pixelArr && pixelArr.length,
                'sample:', pixelArr && pixelArr[0]);
  }})
  .catch(function(e) {{ console.warn('pixel_features.json failed to load:', e); }});

// ~2 pixel snap radius (30 m pixel ≈ 0.00027° lat)
var SNAP_SQ = 0.0006 * 0.0006;

mapToday.on('click', function(e) {{
  if (!pixelArr) {{ console.log('click ignored — pixelArr not loaded yet'); return; }}
  var lat = e.latlng.lat, lon = e.latlng.lng;
  var best = null, bestDist = Infinity;
  for (var i = 0; i < pixelArr.length; i++) {{
    var px = pixelArr[i];
    var d = (px.lat - lat) * (px.lat - lat) + (px.lon - lon) * (px.lon - lon);
    if (d < bestDist) {{ bestDist = d; best = px; }}
  }}
  console.log('click', lat.toFixed(5), lon.toFixed(5),
              '| nearest dist:', Math.sqrt(bestDist).toFixed(6),
              '| threshold:', Math.sqrt(SNAP_SQ).toFixed(6),
              '| nearest:', best);
  var p = (best && bestDist <= SNAP_SQ) ? best : null;
  if (!p) return;
  var labelColor = LABEL_COLORS[p.label] || '#888';
  var rows = Object.keys(FEAT_LABELS).map(function(k) {{
    var val = p[k] === null || p[k] === undefined ? '&mdash;' : p[k].toFixed(4);
    return '<tr><td style="padding:2px 10px 2px 0;color:#555">' + FEAT_LABELS[k] + '</td>' +
           '<td style="text-align:right;font-variant-numeric:tabular-nums">' + val + '</td></tr>';
  }}).join('');
  var html = '<div style="font-size:13px;min-width:230px">' +
    '<b>Predicted: <span style="color:' + labelColor + '">' +
    p.label.charAt(0).toUpperCase() + p.label.slice(1) + '</span></b>' +
    '<table style="margin-top:6px;width:100%;border-collapse:collapse">' +
    rows + '</table></div>';
  L.popup().setLatLng(e.latlng).setContent(html).openOn(mapToday);
}});

// Average transition maps
{avg_js}
</script>
</body>
</html>
'''

    out_path = os.path.join(web_dir, 'index.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'  Saved {out_path}')


def _render_placeholder_html(web_dir,
                              message='Satellite monitoring runs June 1 – December 31. '
                                      'Check back in the summer to see live fall foliage predictions.'):
    """Write a placeholder page when no prediction is available.

    Args:
        web_dir: Directory to write index.html.
        message: Body text to display on the placeholder page.
    """
    html = f'''<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>AT Phenology</title>
<style>body{{font-family:sans-serif;text-align:center;padding:60px;color:#444;}}
h1{{color:#2c6b3f;}}</style></head>
<body>
<h1>Appalachian Trail Fall Phenology — Massachusetts</h1>
<p>{message}</p>
</body></html>'''
    out_path = os.path.join(web_dir, 'index.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'  Wrote off-season placeholder → {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate static phenology web outputs.')
    parser.add_argument('--output-dir', default='./greendown_outputs')
    parser.add_argument('--web-dir',    default='./web_outputs')
    args = parser.parse_args()

    os.makedirs(args.web_dir, exist_ok=True)

    today     = datetime.date.today()
    today_str = today.isoformat()
    year      = today.year
    doy       = today.timetuple().tm_yday
    season_start = datetime.date(year, 6, 1).timetuple().tm_yday
    season_end   = datetime.date(year, 12, 31).timetuple().tm_yday

    # Off-season guard
    if doy < season_start or doy > season_end:
        print(f'DOY {doy} is outside monitoring season (Jun 1–Dec 31). Writing placeholder.')
        _render_placeholder_html(args.web_dir,
                                 message='Satellite monitoring runs June 1 – December 31. '
                                         'Check back in the summer to see live fall foliage predictions.')
        return

    _init_gee()

    # GEE objects — collection starts Jun 1 to cover the full monitoring season
    route_buffer = identify_route_buffer()
    ma_forest    = identify_forests()
    collection   = (
        ee.ImageCollection("NASA/HLS/HLSL30/v002")
        .filterBounds(route_buffer)
        .filterDate(f'{year}-06-01', f'{year}-12-31')
        .select(['B5', 'B4', 'B2'])
        .map(lambda img: img.expression(
            '2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))',
            {'nir': img.select('B5'), 'red': img.select('B4'), 'blue': img.select('B2')}
        ).rename('EVI').clamp(0, 1)
         .addBands(img.expression(
            '(nir - red) / (nir + red)',
            {'nir': img.select('B5'), 'red': img.select('B4')}
        ).rename('NDVI').clamp(0, 1))
         .copyProperties(img, ['system:time_start']))
    )

    # Update rolling pixel state with any new images
    print(f'\nUpdating pixel state for {year}...')
    state_path = update_pixel_state(collection, ma_forest, route_buffer,
                                    year, args.output_dir)

    # If no satellite data exists yet for this season, write a placeholder
    if not os.path.exists(state_path):
        print('No satellite data available yet for this season. Writing placeholder.')
        _render_placeholder_html(args.web_dir,
                                 message='No satellite imagery available yet for this season. '
                                         'Check back in a few days.')
        return

    # Run prediction (request raw feature grids for pixel-click popups)
    print(f'\nRunning prediction for {today_str}...')
    pred_grid, forest_mask, transform, crs, feature_grids = predict_from_pixel_state(
        state_path, today_str, args.output_dir, return_features=True
    )

    # Save prediction PNG
    rgba = _pred_grid_to_rgba(pred_grid, forest_mask, opacity=0.85)
    img  = Image.fromarray(rgba, mode='RGBA')
    pred_png_path = os.path.join(args.web_dir, 'current_pred.png')
    img.save(pred_png_path)
    print(f'  Saved {pred_png_path}')

    # Compute area per label (sq miles)
    pixel_area_sqmi = abs(transform.a * transform.e) / 2.59e6
    forest_labels   = pred_grid[forest_mask]
    areas = {}
    for label in LABEL_ORDER:
        areas[label] = float((forest_labels == label).sum() * pixel_area_sqmi)

    # Write metadata JSON
    bounds = _bounds_to_leaflet(transform, crs, *pred_grid.shape)
    meta = {'date': today_str, 'bounds': bounds, 'areas_sqmi': areas}
    meta_path = os.path.join(args.web_dir, 'current_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  Saved {meta_path}')

    # Write pixel feature JSON for click popups.
    # Each forest pixel stores its WGS84 center lat/lon (computed in Python
    # with pyproj) so the JS click handler can do a simple nearest-neighbour
    # search without any browser-side coordinate transform.
    print('\nWriting pixel_features.json...')
    from pyproj import Transformer as _Transformer
    import rasterio.transform as _rio_transform
    _fwd = _Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    forest_rows, forest_cols = np.where(forest_mask)
    # rasterio.transform.xy returns pixel centres by default
    px_xs, px_ys = _rio_transform.xy(transform, forest_rows.tolist(),
                                     forest_cols.tolist())
    px_lons, px_lats = _fwd.transform(px_xs, px_ys)

    pixels = []
    for i, (ri, ci) in enumerate(zip(forest_rows, forest_cols)):
        entry = {
            'lat': round(float(px_lats[i]), 6),
            'lon': round(float(px_lons[i]), 6),
            'label': str(pred_grid[ri, ci]),
        }
        for feat_col, grid in feature_grids.items():
            v = float(grid[ri, ci])
            entry[feat_col] = None if np.isnan(v) else round(v, 4)
        pixels.append(entry)
    pixel_json = {
        'pixels': pixels,
    }
    feat_path = os.path.join(args.web_dir, 'pixel_features.json')
    with open(feat_path, 'w') as f:
        json.dump(pixel_json, f, separators=(',', ':'))
    print(f'  Saved {feat_path} ({os.path.getsize(feat_path) / 1024:.0f} KB)')

    # Generate average transition DOY maps (one-time, skipped if already present)
    print('\nChecking average transition DOY maps...')
    _ensure_avg_pngs(args.output_dir, args.web_dir)

    # Render HTML
    print('\nRendering index.html...')
    _render_html(args.web_dir, meta)

    print('\nDone.')


if __name__ == '__main__':
    main()
