"""Daily integration health check for the phenology pipeline.

Run after generate_web_outputs.py to verify all outputs look sane.
Writes a Markdown summary to stdout and (inside GitHub Actions) appends
it to $GITHUB_STEP_SUMMARY so results appear in the Actions UI.
Exits 1 if any check fails, which triggers a GitHub failure email.

Usage:
    python health_check.py [--data-dir ./Data] [--web-dir ./web_outputs]
"""
import argparse
import datetime
import json
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np

EASTERN = ZoneInfo('America/New_York')


def _check(label, passed, detail):
    return {'label': label, 'passed': passed, 'detail': detail}


def _load_pixel_features(web_dir):
    """Return (pixels_list, count) or (None, None) on failure."""
    path = os.path.join(web_dir, 'pixel_features.json')
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            data = json.load(f)
        pixels = data.get('pixels', []) if isinstance(data, dict) else data
        return pixels, len(pixels)
    except Exception:
        return None, None


def check_expected_files(web_dir):
    expected = ['index.html', 'current_pred_dt.png', 'current_meta.json',
                'pixel_features.json', 'at_route.geojson']
    missing = [f for f in expected if not os.path.exists(os.path.join(web_dir, f))]
    if missing:
        return _check('Expected files present', False, f'Missing: {", ".join(missing)}')
    return _check('Expected files present', True, f'All {len(expected)} files found')


def check_index_size(web_dir):
    path = os.path.join(web_dir, 'index.html')
    if not os.path.exists(path):
        return _check('index.html not placeholder', False, 'File missing')
    size = os.path.getsize(path)
    if size < 10_000:
        return _check('index.html not placeholder', False,
                      f'Only {size} bytes — likely a placeholder')
    return _check('index.html not placeholder', True, f'{size // 1024} KB')


def check_pred_png(web_dir):
    path = os.path.join(web_dir, 'current_pred_dt.png')
    if not os.path.exists(path):
        return _check('current_pred_dt.png non-empty', False, 'File missing')
    size = os.path.getsize(path)
    if size < 1_000:
        return _check('current_pred_dt.png non-empty', False, f'Only {size} bytes')
    return _check('current_pred_dt.png non-empty', True, f'{size // 1024} KB')


def check_meta_json(web_dir, today_str):
    path = os.path.join(web_dir, 'current_meta.json')
    if not os.path.exists(path):
        return _check('current_meta.json fresh', False, 'File missing')
    try:
        with open(path) as f:
            meta = json.load(f)
    except Exception as e:
        return _check('current_meta.json fresh', False, f'Parse error: {e}')
    date_str = meta.get('date', '')
    if date_str != today_str:
        return _check('current_meta.json fresh', False,
                      f'date={date_str!r}, expected {today_str}')
    return _check('current_meta.json fresh', True, f'date: {date_str}')


def check_pixel_features_size(pixels, n_pixels):
    if pixels is None:
        return _check('pixel_features.json size', False, 'File missing or unreadable')
    if n_pixels < 10_000:
        return _check('pixel_features.json size', False,
                      f'Only {n_pixels} pixels (expected ≥10,000)')
    return _check('pixel_features.json size', True, f'{n_pixels:,} pixels')


def check_predictions_non_degenerate(pixels):
    if pixels is None:
        return _check('Predictions non-degenerate', False, 'pixel_features.json unavailable')
    labels = [p.get('label') for p in pixels]
    non_unknown = sum(1 for l in labels if l not in ('unknown', None))
    if non_unknown == 0:
        return _check('Predictions non-degenerate', False, 'All pixels are "unknown"')
    total = len(labels)
    counts = {}
    for l in labels:
        key = l or 'null'
        counts[key] = counts.get(key, 0) + 1
    summary = ' '.join(f'{k}:{v * 100 // total}%' for k, v in sorted(counts.items()))
    return _check('Predictions non-degenerate', True, summary)


def check_doy_minus_avg_middle(pixels):
    if pixels is None:
        return _check('doy_minus_avg_middle not all null', False,
                      'pixel_features.json unavailable')
    vals = [p.get('doy_minus_avg_middle') for p in pixels]
    non_null = sum(1 for v in vals if v is not None)
    total = len(vals)
    pct = non_null * 100 // total if total else 0
    if pct < 50:
        return _check('doy_minus_avg_middle not all null', False,
                      f'Only {non_null}/{total} ({pct}%) non-null — possible train/serve skew')
    return _check('doy_minus_avg_middle not all null', True, f'{non_null}/{total} non-null')


def check_pixel_state(data_dir, today, today_doy):
    year = today.year
    path = os.path.join(data_dir, f'pixel_state_{year}.npz')
    if not os.path.exists(path):
        return [
            _check('Pixel state exists', False, f'pixel_state_{year}.npz not found'),
            _check('Pixel state freshness', False, 'N/A — file missing'),
            _check('Pixel state valid pixels', False, 'N/A — file missing'),
            _check('Pixel state observations recent', False, 'N/A — file missing'),
        ]

    results = [_check('Pixel state exists', True, f'pixel_state_{year}.npz')]

    mtime_date = datetime.datetime.utcfromtimestamp(os.path.getmtime(path)).date()
    if mtime_date < today:
        results.append(_check('Pixel state freshness', False,
                              f'Last modified {mtime_date} (expected {today})'))
    else:
        results.append(_check('Pixel state freshness', True, 'Updated today'))

    try:
        state = np.load(path)
        evi0 = state['evi_0'].astype(float)
        doy0 = state['doy_0'].astype(float)
    except Exception as e:
        results.append(_check('Pixel state valid pixels', False, f'Load error: {e}'))
        results.append(_check('Pixel state observations recent', False, 'N/A — load error'))
        return results

    valid = int(np.isfinite(evi0).sum())
    if valid < 100:
        results.append(_check('Pixel state valid pixels', False,
                              f'Only {valid} pixels with finite EVI'))
    else:
        results.append(_check('Pixel state valid pixels', True,
                              f'{valid:,} pixels with observations'))

    finite_doys = doy0[np.isfinite(doy0)]
    if len(finite_doys) == 0:
        results.append(_check('Pixel state observations recent', False, 'No valid DOY values'))
    else:
        max_doy = int(finite_doys.max())
        lag = today_doy - max_doy
        if lag > 30:
            results.append(_check('Pixel state observations recent', False,
                                  f'Most recent obs DOY {max_doy}, today DOY {today_doy} ({lag} days stale)'))
        else:
            results.append(_check('Pixel state observations recent', True,
                                  f'Most recent obs DOY {max_doy} ({lag} days ago)'))

    return results


def build_markdown(checks, date_str):
    lines = [
        f'## Daily Health Check — {date_str}',
        '',
        '| Check | Status | Detail |',
        '|---|---|---|',
    ]
    for c in checks:
        icon = '✅ PASS' if c['passed'] else '❌ FAIL'
        lines.append(f'| {c["label"]} | {icon} | {c["detail"]} |')
    lines.append('')
    n_fail = sum(1 for c in checks if not c['passed'])
    if n_fail == 0:
        lines.append(f'**All {len(checks)} checks passed.**')
    else:
        lines.append(f'**{n_fail}/{len(checks)} checks FAILED.**')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Daily pipeline health check.')
    parser.add_argument('--data-dir', default='./Data')
    parser.add_argument('--web-dir',  default='./web_outputs')
    args = parser.parse_args()

    today = datetime.datetime.now(EASTERN).date()
    today_str = today.isoformat()
    today_doy = today.timetuple().tm_yday
    # generate_web_outputs.py only produces full outputs Jun 1–Dec 31 (DOY 152–365).
    in_season = 152 <= today_doy <= 365

    checks = []

    if not in_season:
        # Off-season: only a placeholder index.html is written.
        path = os.path.join(args.web_dir, 'index.html')
        exists = os.path.exists(path)
        checks.append(_check('Off-season placeholder present', exists,
                             'index.html found' if exists else 'index.html missing'))
    else:
        pixels, n_pixels = _load_pixel_features(args.web_dir)

        checks.append(check_expected_files(args.web_dir))
        checks.append(check_index_size(args.web_dir))
        checks.append(check_pred_png(args.web_dir))
        checks.append(check_meta_json(args.web_dir, today_str))
        checks.append(check_pixel_features_size(pixels, n_pixels))
        checks.append(check_predictions_non_degenerate(pixels))
        checks.append(check_doy_minus_avg_middle(pixels))
        checks.extend(check_pixel_state(args.data_dir, today, today_doy))

    md = build_markdown(checks, today_str)
    print(md)

    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a') as f:
            f.write(md + '\n')

    if any(not c['passed'] for c in checks):
        sys.exit(1)


if __name__ == '__main__':
    main()
