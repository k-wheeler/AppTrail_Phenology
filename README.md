# AppTrail Phenology

Daily satellite-based fall foliage tracker for the Massachusetts Appalachian Trail.

**Live site:** https://k-wheeler.github.io/phenology/

[![Daily Phenology Update](https://github.com/k-wheeler/AppTrail_Phenology/actions/workflows/daily_update.yml/badge.svg)](https://github.com/k-wheeler/AppTrail_Phenology/actions/workflows/daily_update.yml)

---

## What it does

This project monitors ~15,000 30×30 m forest pixels along the Massachusetts AT using NASA Harmonized Landsat (HLS) imagery fetched daily from Google Earth Engine. For each pixel, a decreasing logistic curve is fitted to its multi-year EVI time series to estimate the day-of-year when fall foliage color change starts, peaks, and ends.

A decision tree classifier trained on 10 years of labeled pixel-observations uses 11 features — current EVI and NDVI, their recent changes, day length, days relative to each pixel's historical average mid-transition date, the most common (mode) predicted label over the past 7 days, accumulated cold degree-days (CDD) since July 1, and the most recent daily mean temperature — to assign one of four phenological states: **Before**, **Early**, **Late**, or **After** (color change complete). CDD is computed as the sum of max(0, 5 − T_mean°C) for each day since July 1, using gridMET daily Tmax/Tmin at ~4 km resolution. The rolling label history and accumulated CDD are both committed to GitHub daily so each Action run can pick up where the last left off.

Each morning the pipeline fetches new imagery, updates a rolling pixel state, reruns predictions across all forest pixels, and publishes results as a fully static Leaflet interactive map on GitHub Pages.

---

## Architecture

```
Offline training (Main.ipynb, run once per season)
    ├── Download HLS stacks, compute EVI/NDVI
    ├── Fit logistic curves per pixel per year
    ├── Assemble labeled feature table
    └── Train DecisionTreeClassifier
         └── commit: decision_tree_model.joblib, norm_stats.json,
                     greendown_{start,middle,end}_avg.tif

Daily GitHub Action (9 AM UTC)
    ├── update_pixel_state()    ← fetch new HLS images, update rolling 3-obs window
    │    └── commit: pixel_state_{year}.npz
    ├── predict_from_pixel_state()  ← z-score features, run decision tree
    └── generate_web_outputs.py     ← render PNG / JSON / HTML
         └── push to GitHub Pages repo
```

Large `.npy` stacks (~2.5 GB/year) live only on the local machine and are never pushed to GitHub.

---

## Repository layout

**Daily pipeline (GitHub Action)**

| File | Role |
|---|---|
| `generate_web_outputs.py` | GEE auth, pixel-state update, prediction, HTML/PNG/JSON rendering |
| `predict_for_date.py` | Loads pixel state + CDD state, builds 11-feature matrix, runs z-score + decision tree |
| `fit_greendown_curves.py` | Downloads HLS imagery, fits logistic curves, updates `pixel_state_{year}.npz` |
| `map_utils.py` | Raster-to-RGBA rendering, WGS84 bounds, Web Mercator warp |
| `health_check.py` | Post-run QC: verifies outputs, pixel counts, freshness; exits 1 on failure |

**Offline training (Jupyter / local only)**

| File | Role |
|---|---|
| `read_and_process_hls.py` | Download full HLS stacks, compute EVI/NDVI |
| `fit_greendown_curves.py` | Fit logistic curves to historical pixel time series |
| `build_data_table.py` | Assemble labeled feature table from historical transition estimates |
| `edit_data_table.py` | Gap-filling, global average middle DOY, class balancing |
| `decision_trees.py` | Train `DecisionTreeClassifier`, save `decision_tree_model.joblib` |
| `filter_ci_widths.py` | Filter pixels by confidence-interval width thresholds |
| `identify_locations.py` | Load AT route from GEE, clip to MA, compute forest mask |

**Utilities / inspection**

| File | Role |
|---|---|
| `constants.py` | Shared constants (NODATA, CI thresholds, label colors) |
| `inspect_tree.py` | Print full decision rules in raw feature units |
| `explain_prediction.py` | Trace one sample through the decision tree |
| `plot_feature_distributions.py` | Exploratory feature plots |
| `dashboard.py` | Local Streamlit map (uses full year stack; not used in Action) |

**Tests**

| File | Role |
|---|---|
| `qc_tests.py` | 47 pytest unit tests — run in CI before each deploy |

---

## Committed artifacts

The daily Action reads these from the repo. They are produced by offline training and must be re-committed whenever the model is retrained:

| File | Description |
|---|---|
| `decision_tree_model.joblib` | Trained sklearn model |
| `norm_stats.json` | Per-feature mean/std for z-score normalization |
| `greendown_{start,middle,end}_avg.tif` | Multi-year average transition-date rasters |
| `greendown_avg_meta.json` | Grid metadata (dimensions, CRS, nodata) |
| `pixel_state_{year}.npz` | Rolling 3-observation pixel state (updated daily by Action) |
| `cdd_state_{year}.npz` | Accumulated cold degree-days since Jul 1 (updated daily by Action) |

---

## Setup

**Prerequisites:** Python 3.11, `pip install -r requirements.txt`, a Google Earth Engine project with the `AT_Trail` asset registered.

**Run locally:**
```bash
export GEE_SERVICE_ACCOUNT_KEY="$(cat your-key.json)"
python generate_web_outputs.py --output-dir ./greendown_outputs --web-dir ./web_outputs
open web_outputs/index.html
```

**Run tests:**
```bash
pytest qc_tests.py -v
```

**Offline training:** Run `Main.ipynb` cells in order, then commit the updated model artifacts listed above.

---

## GitHub Action setup

Three repo secrets are required (Settings → Secrets → Actions):

| Secret | Value |
|---|---|
| `GEE_SERVICE_ACCOUNT_KEY` | Full contents of the GEE service account JSON key |
| `PAGES_REPO_TOKEN` | GitHub PAT with Contents: Read & Write on the Pages repo |
| `PAGES_REPO` | Pages repo path, e.g. `k-wheeler/k-wheeler.github.io` |

Trigger a manual run via **Actions → Daily Phenology Update → Run workflow**.

---

## Key constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_CI_WIDTH` | 15 days | Max CI width for pixels entering the training table |
| `CROSS_YEAR_MAX_CI_WIDTH` | 30 days | Max CI width for cross-year DOY averaging |
| `GAP_FILL_MAX_CI_WIDTH` | 14 days | Max CI width for gap-fill global mean |
| Season window | DOY 152–365 | Jun 1–Dec 31; off-season writes a placeholder page |

---

## Data sources

- **NASA HLS HLSL30 v002** — Harmonized Landsat 30 m surface reflectance (via Google Earth Engine)
- **gridMET (`IDAHO_EPSCOR/GRIDMET`)** — Daily Tmax/Tmin at ~4 km (University of Idaho / Climatology Lab); used for cold degree-day accumulation
- **TIGER/2018/States** — MA state boundary
- **`projects/turnkey-lacing-391919/assets/AT_Trail`** — AT route (custom GEE asset)
- **NLCD 2021** — Deciduous & mixed forest mask
