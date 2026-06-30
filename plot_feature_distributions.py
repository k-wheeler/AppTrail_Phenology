import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from constants import LABEL_COLORS


# Columns to skip (non-numeric or categorical)
SKIP_COLS = {'date', 'label'}

# Columns to exclude from the correlation matrix (in addition to date/label)
CORR_SKIP_COLS = SKIP_COLS | {'year'}

# Per-column display names for axis labels
COL_LABELS = {
    'year':           'Year',
    'doy':            'Day of Year',
    'EVI':            'EVI',
    'NDVI':           'NDVI',
    'evi_delta':      'EVI Δ1',
    'evi_delta2':     'EVI Δ2',
    'ndvi_delta':     'NDVI Δ1',
    'ndvi_delta2':    'NDVI Δ2',
    'day_length_hrs':             'Day Length (hrs)',
    'doy_minus_avg_start':  'DOY − Avg Start DOY',
    'doy_minus_avg_middle': 'DOY − Avg Middle DOY',
    'doy_minus_avg_end':    'DOY − Avg End DOY',
}


def plot_feature_distributions(feature_df):
    """Plot histograms and a correlation heatmap for all numeric features.

    For each numeric column, plots one histogram subplot per label category
    (before / early / late / after) with a dashed vertical line at the mean.
    Also plots a pairwise Pearson correlation heatmap across all columns except
    year and date.

    Args:
        feature_df: DataFrame with a 'label' column and numeric feature columns.
    """
    labels = ['before', 'early', 'late', 'after']

    numeric_cols = [c for c in feature_df.columns if c not in SKIP_COLS]

    for col in numeric_cols:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
        xlabel = COL_LABELS.get(col, col)

        for ax, label in zip(axes, labels):
            subset = feature_df.loc[feature_df['label'] == label, col].dropna()

            if col == 'year':
                year_vals   = sorted(subset.unique().astype(int))
                year_counts = [int((subset == y).sum()) for y in year_vals]
                ax.bar([str(y) for y in year_vals], year_counts,
                       color=LABEL_COLORS[label], edgecolor='white', linewidth=0.4)
                ax.set_title(f'{label.capitalize()} (n={len(subset):,})')
                ax.tick_params(axis='x', rotation=45)
            else:
                mean = subset.mean()
                ax.hist(subset, bins=40, color=LABEL_COLORS[label],
                        edgecolor='white', linewidth=0.4)
                ax.axvline(mean, color='black', linewidth=1.2, linestyle='--')
                ax.set_title(f'{label.capitalize()} (n={len(subset):,})\nmean = {mean:.3f}')

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')

        fig.suptitle(f'{xlabel} by greendown phenological stage', fontsize=13)
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Pairwise correlation heatmap
    # ----------------------------
    corr_cols  = [c for c in feature_df.columns if c not in CORR_SKIP_COLS]
    corr_df    = feature_df[corr_cols].dropna()
    corr_matrix = corr_df.corr()
    tick_labels = [COL_LABELS.get(c, c) for c in corr_cols]

    n = len(corr_cols)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.1), max(5, n * 1.0)))

    im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Pearson r')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(val) < 0.7 else 'white')

    ax.set_title('Pairwise Pearson Correlations', fontsize=13)
    plt.tight_layout()
    plt.show()
