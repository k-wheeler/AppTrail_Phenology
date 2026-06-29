"""Trace a single set of feature values through the decision tree.

Pass raw feature values (the same units you'd read off the map popup) and this
prints the exact branch of decisions the tree followed to reach its prediction.

Inputs are z-scored with norm_stats.json before being fed to the model (the tree
was trained on normalized features), and any feature you omit is treated the same
way the live prediction treats a missing value: filled with the training mean
(i.e. a normalized 0). Thresholds are shown converted back to raw units.

Usage:
    python explain_prediction.py --EVI 0.43 --NDVI 0.70 --day_length_hrs 11.2 \
        --evi_delta -0.03 --doy_minus_avg_middle -5

    # or pass everything as one JSON blob:
    python explain_prediction.py --json '{"EVI":0.43,"NDVI":0.70,"day_length_hrs":11.2}'
"""
import argparse
import json
import os

import joblib
import numpy as np

from predict_for_date import FEATURE_COLS


def explain_prediction(feature_values, model_dir='./Model_Outputs', verbose=True):
    """Return (and optionally print) the decision path for one sample.

    Args:
        feature_values: Dict mapping FEATURE_COLS names to raw values. Missing
            keys (or NaN) are treated as the training mean, matching the live
            prediction's NaN->0 normalized substitution.
        model_dir: Directory containing decision_tree_model.joblib and
            norm_stats.json.
        verbose: If True, print the path to stdout.

    Returns:
        Dict with keys: 'prediction', 'steps' (list of per-node dicts), and
        'leaf' (leaf sample count and class distribution).
    """
    mdl = joblib.load(os.path.join(model_dir, 'decision_tree_model.joblib'))
    with open(os.path.join(model_dir, 'norm_stats.json')) as f:
        norm_stats = json.load(f)

    means = np.array([norm_stats[c]['mean'] for c in FEATURE_COLS])
    stds  = np.array([norm_stats[c]['std']  for c in FEATURE_COLS])

    # Build raw feature vector; missing/NaN -> NaN for now
    raw = np.full(len(FEATURE_COLS), np.nan)
    for j, col in enumerate(FEATURE_COLS):
        v = feature_values.get(col, None)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            raw[j] = float(v)

    # Normalize, then substitute 0 (the normalized mean) for missing values —
    # identical to predict_for_date's handling.
    x_norm = (raw - means) / stds
    x_norm = np.where(np.isnan(x_norm), 0.0, x_norm)
    # raw value actually compared by the tree (missing -> mean)
    raw_used = x_norm * stds + means

    X = x_norm.reshape(1, -1)
    t = mdl.tree_
    classes = mdl.classes_

    node_indicator = mdl.decision_path(X)
    path_nodes = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]]
    leaf_id = mdl.apply(X)[0]

    steps = []
    for node in path_nodes:
        if node == leaf_id:
            break
        f = t.feature[node]
        fname = FEATURE_COLS[f]
        thr_raw = t.threshold[node] * stds[f] + means[f]
        val = raw_used[f]
        went_left = X[0, f] <= t.threshold[node]
        steps.append({
            'feature': fname,
            'value': float(val),
            'threshold': float(thr_raw),
            'direction': 'left (<=)' if went_left else 'right (>)',
            'missing': bool(np.isnan(raw[f])),
        })

    value = t.value[leaf_id][0]
    probs = value / value.sum() if value.sum() else value
    prediction = classes[int(np.argmax(probs))]
    leaf = {
        'n': int(t.n_node_samples[leaf_id]),
        'purity': float(probs.max()),
        'distribution': {classes[i]: float(probs[i])
                         for i in range(len(classes)) if probs[i] > 0.005},
    }

    if verbose:
        provided = ', '.join(
            f'{c}={feature_values[c]}' for c in FEATURE_COLS
            if c in feature_values and feature_values[c] is not None)
        missing = [c for c in FEATURE_COLS
                   if c not in feature_values or feature_values[c] is None]
        print(f'Input (raw): {provided}')
        if missing:
            print(f'Missing (treated as training mean): {", ".join(missing)}')
        print('\nDecision path:')
        for i, s in enumerate(steps, 1):
            tag = '  [missing->mean]' if s['missing'] else ''
            op = '<=' if s['direction'].startswith('left') else '>'
            print(f'  {i:2}. {s["feature"]:22} = {s["value"]:9.4f}  '
                  f'{op} {s["threshold"]:9.4f}   -> go '
                  f'{s["direction"].split()[0].upper()}{tag}')
        dist = ', '.join(f'{k}={v * 100:.0f}%' for k, v in leaf['distribution'].items())
        print(f'\n=> PREDICT "{prediction}"  '
              f'(leaf n={leaf["n"]}, {leaf["purity"]:.0%} pure; {dist})')

    return {'prediction': prediction, 'steps': steps, 'leaf': leaf}


def main():
    parser = argparse.ArgumentParser(
        description='Trace feature values through the decision tree.')
    parser.add_argument('--model-dir', default='./Model_Outputs')
    parser.add_argument('--json', default=None,
                        help='JSON object of {feature: value}; overrides individual flags.')
    for col in FEATURE_COLS:
        parser.add_argument(f'--{col}', type=float, default=None)
    args = parser.parse_args()

    if args.json:
        feature_values = json.loads(args.json)
    else:
        feature_values = {col: getattr(args, col) for col in FEATURE_COLS
                          if getattr(args, col) is not None}

    if not feature_values:
        parser.error('Provide at least one feature value (e.g. --EVI 0.43) or --json.')

    explain_prediction(feature_values, model_dir=args.model_dir)


if __name__ == '__main__':
    main()
