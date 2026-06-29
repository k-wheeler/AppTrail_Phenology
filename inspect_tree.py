"""Inspect the fitted decision tree: print human-readable decision rules.

The tree is trained on z-scored features, so its raw thresholds are in
normalized units. This script converts each threshold back to real feature
units using norm_stats.json, so the rules read in actual EVI / NDVI /
day-length / day values. Follow the indented if/else branches from the top to
see exactly which conditions lead to each predicted phenological state.

Usage:
    python inspect_tree.py [--model-dir ./Model_Outputs] [--plot]
"""
import argparse
import json
import os

import joblib
import numpy as np

from predict_for_date import FEATURE_COLS


def _denormalize(threshold, feature_name, norm_stats):
    """Convert a z-scored threshold back to raw feature units."""
    s = norm_stats[feature_name]
    return threshold * s['std'] + s['mean']


def print_rules(mdl, feature_names, norm_stats):
    """Print the full tree as nested if/else rules with raw-unit thresholds."""
    t = mdl.tree_
    classes = mdl.classes_

    def recurse(node, depth):
        indent = '    ' * depth
        is_leaf = t.children_left[node] == t.children_right[node]
        if is_leaf:
            value = t.value[node][0]
            probs = value / value.sum() if value.sum() else value
            n = int(t.n_node_samples[node])          # true #training samples here
            pred = classes[int(np.argmax(probs))]
            purity = probs.max()
            dist = ', '.join(f'{classes[i]}={probs[i] * 100:.0f}%'
                             for i in range(len(classes)) if probs[i] > 0.005)
            print(f'{indent}=> PREDICT "{pred}"  '
                  f'(n={n}, {purity:.0%} pure; {dist})')
            return
        fname = feature_names[t.feature[node]]
        raw = _denormalize(t.threshold[node], fname, norm_stats)
        print(f'{indent}if {fname} <= {raw:.4f}:')
        recurse(t.children_left[node], depth + 1)
        print(f'{indent}else:  # {fname} > {raw:.4f}')
        recurse(t.children_right[node], depth + 1)

    recurse(0, 0)


def main():
    parser = argparse.ArgumentParser(description='Inspect the fitted decision tree.')
    parser.add_argument('--model-dir', default='./Model_Outputs')
    parser.add_argument('--plot', action='store_true',
                        help='Also save a full-depth tree plot PNG.')
    args = parser.parse_args()

    mdl = joblib.load(os.path.join(args.model_dir, 'decision_tree_model.joblib'))
    with open(os.path.join(args.model_dir, 'norm_stats.json')) as f:
        norm_stats = json.load(f)

    print(f'Tree depth: {mdl.get_depth()}   leaves: {mdl.get_n_leaves()}')
    print(f'Classes: {list(mdl.classes_)}\n')

    print('Feature importances (Gini):')
    order = np.argsort(mdl.feature_importances_)[::-1]
    for i in order:
        print(f'  {FEATURE_COLS[i]:24} {mdl.feature_importances_[i]:.4f}')

    print('\nDecision rules (thresholds in raw feature units):')
    print('-' * 70)
    print_rules(mdl, FEATURE_COLS, norm_stats)

    if args.plot:
        import matplotlib.pyplot as plt
        from sklearn import tree
        fig, ax = plt.subplots(figsize=(60, 30))
        tree.plot_tree(mdl, feature_names=FEATURE_COLS,
                       class_names=list(mdl.classes_), filled=True, ax=ax)
        out = os.path.join(args.model_dir, 'decision_tree_full.png')
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'\nSaved full-depth plot to {out}  '
              '(NOTE: thresholds in this image are z-scored, not raw units)')


if __name__ == '__main__':
    main()
