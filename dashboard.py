import base64
import datetime
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
import streamlit as st
from streamlit_folium import st_folium
from PIL import Image

from predict_for_date import predict_phenology
from constants import DATA_DIR, GREENDOWN_DIR, MODEL_DIR, LABEL_COLORS, LABEL_ORDER
from map_utils import _pred_grid_to_rgba, _get_wgs84_bounds


def _make_folium_map(pred_grid, forest_mask, transform, crs):
    """Build a folium Map with a CartoDB Positron basemap and the prediction overlay.

    Overlays the prediction grid as a semi-transparent PNG. Non-forest pixels are
    transparent. Adds an HTML legend for phenological state colors.

    Args:
        pred_grid: 2D label array of shape (h, w).
        forest_mask: Boolean mask of shape (h, w) marking forested pixels.
        transform: Affine transform of the prediction raster.
        crs: Coordinate reference system of the prediction raster.

    Returns:
        Configured folium.Map object.
    """
    bounds = _get_wgs84_bounds(transform, crs, *pred_grid.shape)
    center = [(bounds[0][0] + bounds[1][0]) / 2,
              (bounds[0][1] + bounds[1][1]) / 2]

    m = folium.Map(location=center, zoom_start=10, tiles='CartoDB positron')

    # Convert prediction grid to base64 PNG data URI
    rgba = _pred_grid_to_rgba(pred_grid, forest_mask, opacity=0.85)
    img  = Image.fromarray(rgba, mode='RGBA')
    buf  = io.BytesIO()
    img.save(buf, format='PNG')
    data_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

    folium.raster_layers.ImageOverlay(
        image=data_uri,
        bounds=bounds,
        opacity=1.0,   # opacity already baked into RGBA alpha channel
        interactive=False,
        cross_origin=False,
        name='Phenology Prediction',
    ).add_to(m)

    # Legend as a custom HTML control
    legend_html = '''
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:6px;
                border:2px solid #888; font-size:14px; line-height:2;
                color:#111;">
      <b style="color:#000;">Phenological State</b><br>
      {items}
    </div>
    '''.format(items=''.join(
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{LABEL_COLORS[l]};margin-right:8px;vertical-align:middle;'
        f'border-radius:2px;border:1px solid #555;"></span>'
        f'<span style="color:#111;">{l.capitalize()}</span><br>'
        for l in LABEL_ORDER if l != 'unknown'
    ))
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m


def _make_histogram(pred_grid, forest_mask, transform):
    """Plot a bar chart of area (sq miles) per phenological state for forest pixels.

    Args:
        pred_grid: 2D label array of shape (h, w).
        forest_mask: Boolean mask of shape (h, w) marking forested pixels.
        transform: Affine transform used to compute pixel area.

    Returns:
        matplotlib Figure with the bar chart.
    """
    # Pixel area in sq miles derived from raster transform (pixel width × height)
    pixel_area_sqmi = abs(transform.a * transform.e) / 2.59e6  # m² → sq miles

    forest_labels = pred_grid[forest_mask]
    labels, counts = np.unique(forest_labels, return_counts=True)
    ordered = [(l, counts[list(labels).index(l)] * pixel_area_sqmi)
               for l in LABEL_ORDER if l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([l.capitalize() for l, _ in ordered],
           [area for _, area in ordered],
           color=[LABEL_COLORS[l] for l, _ in ordered],
           edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Area (sq miles)')
    ax.set_title('Phenological States\n(forest pixels only)')
    plt.tight_layout()
    return fig


# ----------------------------
# Streamlit app
# ----------------------------
st.title('Appalachian Trail Phenology Prediction — 2025')
st.markdown('Select a date to predict the phenological state across the MA forest buffer.')

selected_date = st.date_input(
    'Date',
    value=datetime.date(2025, 9, 15),
    min_value=datetime.date(2025, 7, 1),
    max_value=datetime.date(2025, 12, 31),
)

# Only rerun prediction when date changes
if 'last_date' not in st.session_state or st.session_state.last_date != selected_date:
    with st.spinner('Running prediction...'):
        pred_grid, forest_mask, transform, crs = predict_phenology(
            selected_date.isoformat(),
            data_dir=DATA_DIR, greendown_dir=GREENDOWN_DIR, model_dir=MODEL_DIR
        )
    st.session_state.last_date   = selected_date
    st.session_state.pred_grid   = pred_grid
    st.session_state.forest_mask = forest_mask
    st.session_state.transform   = transform
    st.session_state.crs         = crs

pred_grid   = st.session_state.pred_grid
forest_mask = st.session_state.forest_mask
transform   = st.session_state.transform
crs         = st.session_state.crs

forest_labels = pred_grid[forest_mask]
n_forest = int(forest_mask.sum())
n_known  = int((forest_labels != 'unknown').sum())
st.caption(f'{n_known:,} of {n_forest:,} forest pixels have a prediction for {selected_date}.')

col1, col2 = st.columns([3, 1])
with col1:
    st_folium(_make_folium_map(pred_grid, forest_mask, transform, crs),
              width=700, height=500)
with col2:
    st.pyplot(_make_histogram(pred_grid, forest_mask, transform))
