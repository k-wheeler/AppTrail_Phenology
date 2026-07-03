"""LSTM-based phenology classifier for the AppTrail greendown tracker.

Trains on per-pixel-year observation sequences (many-to-many labeling) and
predicts the phenological state (before/early/late/after) at each time step.
A temperature feature (tmean_recent) is included here and must be enabled in
build_data_table.py (include_temperature=True) when building the RNN training
data.  (CDD was dropped: its current-year serving series only covers ~16 days,
so older observations in the sequence were fed a misleading zero-fill.)

Offline training only — the website continues to use the decision tree model
until rnn_predict_for_date.py (future) is wired into generate_web_outputs.py.

Usage in Main.ipynb (Cell 2):
    rnn_df = build_feature_table(OUTPUT_DIR, training_years,
                                  retain_pixel_id=True, include_temperature=True)
    sequences = build_rnn_sequences(rnn_df)
    norm_stats = compute_rnn_norm_stats(sequences)
    normalize_sequences(sequences, norm_stats)
    train_seqs, val_seqs = split_sequences(sequences)
    model = RNNPhenologyModel(input_size=len(RNN_FEATURE_COLS))
    model = train_rnn(model, train_seqs, val_seqs, epochs=60)
    evaluate_rnn(model, val_seqs)
    save_rnn_model(model, norm_stats, OUTPUT_DIR)
"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

RNN_FEATURE_COLS = [
    'EVI', 'NDVI', 'evi_delta', 'evi_delta2',
    'ndvi_delta', 'ndvi_delta2', 'day_length_hrs',
    'doy_minus_avg_middle', 'tmean_recent',
]

_LABEL_ORDER = ['before', 'early', 'late', 'after']
_LABEL_TO_INT = {lbl: i for i, lbl in enumerate(_LABEL_ORDER)}
_INT_TO_LABEL = {i: lbl for i, lbl in enumerate(_LABEL_ORDER)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RNNPhenologyModel(nn.Module):
    """2-layer LSTM → linear head for 4-class phenological state prediction.

    Processes packed variable-length sequences; predicts a label at every
    time step (many-to-many), trained with CrossEntropyLoss masked to
    non-padding positions.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.3, num_classes=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, packed_input):
        packed_out, _ = self.lstm(packed_input)
        out_padded, lengths = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.head(out_padded)   # (batch, max_len, num_classes)
        return logits, lengths


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_rnn_sequences(feature_df, use_soft_labels=False):
    """Reshape a flat per-observation DataFrame into per-pixel-year sequences.

    Args:
        feature_df: DataFrame returned by
            build_feature_table(..., retain_pixel_id=True, include_temperature=True).
            Required columns: pixel_id, year, doy, all RNN_FEATURE_COLS, label.
        use_soft_labels: If True and the CI-width and transition-date columns are
            present (retain_pixel_id=True path), compute probabilistic labels
            encoding boundary uncertainty via _compute_soft_label. Falls back to
            hard labels when required columns are absent.

    Returns:
        List of (feature_array, label_array) tuples, one per pixel-year.
        Hard labels: label_array is int64  ndarray of shape (seq_len,), values 0–3.
        Soft labels: label_array is float32 ndarray of shape (seq_len, 4).
    """
    df = feature_df.copy()

    # NaN delta fill: same 0-fill used in the DT pipeline
    delta_cols = ['evi_delta', 'evi_delta2', 'ndvi_delta', 'ndvi_delta2']
    df[delta_cols] = df[delta_cols].fillna(0.0)

    # NaN temperature fill: column mean (rare; only early-season before Aug 1)
    for col in ('cdd_accumulated', 'tmean_recent'):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    _soft_cols = ['doy', 'transition_start', 'transition_middle', 'transition_end',
                  'ci_width_start', 'ci_width_middle', 'ci_width_end']
    can_soft = use_soft_labels and all(c in df.columns for c in _soft_cols)
    if use_soft_labels and not can_soft:
        print('  Warning: soft label columns not found; falling back to hard labels.')

    label_int = df['label'].map(_LABEL_TO_INT)

    sequences = []
    for (year, pid), grp in df.groupby(['year', 'pixel_id'], sort=False):
        grp = grp.sort_values('doy')
        feats = grp[RNN_FEATURE_COLS].values.astype(np.float32)
        if can_soft:
            label_arr = np.stack([
                _compute_soft_label(
                    row['doy'],
                    row['transition_start'], row['transition_middle'], row['transition_end'],
                    row['ci_width_start'],   row['ci_width_middle'],   row['ci_width_end'],
                )
                for _, row in grp.iterrows()
            ])  # (seq_len, 4) float32
        else:
            label_arr = label_int.loc[grp.index].values.astype(np.int64)
        sequences.append((feats, label_arr))

    return sequences


def compute_rnn_norm_stats(sequences):
    """Compute per-feature mean and std across all time steps of all sequences.

    Returns:
        Dict: {feature_name: {'mean': float, 'std': float}, ...}
    """
    all_feats = np.concatenate([f for f, _ in sequences], axis=0)
    stats = {}
    for i, col in enumerate(RNN_FEATURE_COLS):
        vals = all_feats[:, i]
        mean = float(np.nanmean(vals))
        std  = float(np.nanstd(vals))
        stats[col] = {'mean': mean, 'std': max(std, 1e-8)}
    return stats


def normalize_sequences(sequences, norm_stats):
    """Z-score normalize all sequences in-place.

    Any NaN remaining after normalization is replaced with 0.0, which equals
    the feature mean in z-score space and is the correct neutral fill value.
    """
    means = np.array([norm_stats[c]['mean'] for c in RNN_FEATURE_COLS], dtype=np.float32)
    stds  = np.array([norm_stats[c]['std']  for c in RNN_FEATURE_COLS], dtype=np.float32)
    for i, (feats, labels) in enumerate(sequences):
        normalized = (feats - means) / stds
        normalized = np.nan_to_num(normalized, nan=0.0)
        sequences[i] = (normalized, labels)


def oversample_sequences(sequences, target_ratio=1.0, seed=42):
    """Duplicate minority-class sequences to reduce class imbalance.

    Sequences that contain observations of underrepresented classes are
    randomly duplicated until each class's total observation count reaches
    target_ratio × majority_class_count.  Apply to training sequences only —
    never to the validation set.

    Args:
        sequences:    List of (feature_array, label_array) tuples.
        target_ratio: 1.0 = fully balance all classes to majority count.
                      0.5 = halfway (less aggressive, lower overfitting risk
                      when the pool of minority sequences is small).
        seed:         RNG seed for reproducibility.

    Returns:
        Augmented list (originals + duplicates), shuffled.
    """
    class_counts = np.zeros(4, dtype=np.int64)
    for _, labels in sequences:
        for c in range(4):
            class_counts[c] += int((labels == c).sum())

    majority = int(class_counts.max())
    rng    = np.random.default_rng(seed)
    result = list(sequences)

    for cls in range(4):
        target = int(class_counts[cls]
                     + target_ratio * (majority - class_counts[cls]))
        if class_counts[cls] >= target:
            continue
        containing = [s for s in sequences if (s[1] == cls).any()]
        if not containing:
            continue
        needed = target - int(class_counts[cls])
        added  = 0
        while added < needed:
            s = containing[int(rng.integers(len(containing)))]
            result.append(s)
            added += int((s[1] == cls).sum())

    indices = list(range(len(result)))
    rng.shuffle(indices)
    return [result[i] for i in indices]


def split_sequences(sequences, val_frac=0.2, seed=42):
    """Split by pixel-year (not by row) to keep each sequence intact.

    Returns:
        (train_sequences, val_sequences)
    """
    rng = random.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_frac))
    val_set   = set(indices[:n_val])
    train_seqs = [sequences[i] for i in range(len(sequences)) if i not in val_set]
    val_seqs   = [sequences[i] for i in range(len(sequences)) if i in val_set]
    return train_seqs, val_seqs


# ---------------------------------------------------------------------------
# DataLoader utilities
# ---------------------------------------------------------------------------

class _SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feats, labels = self.sequences[idx]
        label_tensor = (torch.tensor(labels, dtype=torch.float32)
                        if labels.ndim == 2
                        else torch.tensor(labels, dtype=torch.long))
        return torch.tensor(feats, dtype=torch.float32), label_tensor


def collate_rnn_batch(batch):
    """Pad a batch of variable-length sequences and pack for the LSTM.

    Returns:
        packed:        PackedSequence ready for model.forward()
        padded_labels: (batch, max_len) long tensor; padding positions = -1
        lengths:       (batch,) long tensor of true sequence lengths
    """
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    feat_tensors  = [x[0] for x in batch]
    label_tensors = [x[1] for x in batch]
    lengths = torch.tensor([len(f) for f in feat_tensors], dtype=torch.long)
    padded_feats = pad_sequence(feat_tensors, batch_first=True, padding_value=0.0)
    if label_tensors[0].dim() == 2:
        # Soft labels (seq_len, 4): pad with zero rows; masked by zero-sum check
        padded_labels = pad_sequence(label_tensors, batch_first=True, padding_value=0.0)
    else:
        # Hard labels (seq_len,): pad with -1; masked by ignore_index=-1
        padded_labels = pad_sequence(label_tensors, batch_first=True, padding_value=-1)
    packed = pack_padded_sequence(padded_feats, lengths.cpu(),
                                  batch_first=True, enforce_sorted=True)
    return packed, padded_labels, lengths


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_rnn(model, train_seqs, val_seqs, epochs=60, lr=1e-3,
              batch_size=64, device='cpu', early_stopping=False,
              patience=10, L1_regular=False, l1_lambda=1e-4):
    """Train the RNN model and return it with per-epoch loss/accuracy history.

    Args:
        model:          RNNPhenologyModel instance (weights initialised).
        train_seqs:     Normalized training sequences.
        val_seqs:       Normalized validation sequences.
        epochs:         Maximum number of full passes over the training set.
        lr:             Adam learning rate.
        batch_size:     Sequences per batch.
        device:         'cpu' or 'cuda'.
        early_stopping: If True, stop training when val_loss has not improved
                        for `patience` consecutive epochs and restore the best
                        weights seen so far.
        patience:       Number of epochs without val_loss improvement before
                        stopping (only used when early_stopping=True).
        L1_regular:     If True, add L1 regularization to the loss.
        l1_lambda:      Regularization strength (only used when L1_regular=True).

    Returns:
        Tuple of (trained model, history, training_time_sec).
    """
    import copy

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    train_loader = DataLoader(
        _SequenceDataset(train_seqs), batch_size=batch_size,
        shuffle=True, collate_fn=collate_rnn_batch,
    )

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    t_start = time.time()

    best_val_loss   = float('inf')
    best_weights    = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for packed, padded_labels, _ in train_loader:
            packed        = _move_packed(packed, device)
            padded_labels = padded_labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(packed)
            if padded_labels.dim() == 3:  # soft labels (batch, max_len, 4)
                log_probs = F.log_softmax(logits, dim=-1)
                mask_s = padded_labels.sum(dim=-1) > 0
                loss = -(padded_labels * log_probs).sum(dim=-1)[mask_s].mean()
            else:
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    padded_labels.reshape(-1),
                )
            if L1_regular:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        epoch_loss = total_loss / max(n_batches, 1)
        val_loss   = _eval_loss(model, val_seqs, device, batch_size)
        val_acc    = _eval_accuracy(model, val_seqs, device, batch_size)
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{epochs}  '
                  f'train_loss={epoch_loss:.4f}  '
                  f'val_loss={val_loss:.4f}  '
                  f'val_acc={val_acc:.4f}')

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'  Early stopping at epoch {epoch} '
                          f'(no val_loss improvement for {patience} epochs)')
                    model.load_state_dict(best_weights)
                    break

    training_time_sec = time.time() - t_start
    print(f'  Training time: {training_time_sec / 60:.1f} min')
    return model, history, training_time_sec


def plot_rnn_training_curves(history, filename_ext='', model_dir=None):
    """Plot training and validation loss on the same graph vs epoch.

    Args:
        history:      Dict returned by train_rnn with keys 'train_loss', 'val_loss',
                      and 'val_acc'.
        filename_ext: Optional suffix used to label the plot title and output filename.
        model_dir:    If provided, saves the plot as rnn_training_curves{filename_ext}.png
                      in that directory. Otherwise just displays it.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, history['train_loss'], color='steelblue', label='Train loss')
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], color='darkorange', label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'RNN Training Curves{filename_ext}')
    ax.legend()

    plt.tight_layout()

    if model_dir is not None:
        import os
        out = os.path.join(model_dir, f'rnn_training_curves{filename_ext}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')

    plt.show()


def evaluate_rnn(model, sequences, device='cpu', batch_size=64):
    """Run inference and print a per-class classification report.

    Returns:
        Overall accuracy (float).
    """
    model.eval()
    all_preds, all_true = [], []
    loader = DataLoader(
        _SequenceDataset(sequences), batch_size=batch_size,
        shuffle=False, collate_fn=collate_rnn_batch,
    )
    with torch.no_grad():
        for packed, padded_labels, _ in loader:
            packed = _move_packed(packed, device)
            logits, _ = model(packed)
            preds = logits.argmax(dim=-1).cpu()
            if padded_labels.dim() == 3:  # soft labels
                true_cls = padded_labels.argmax(dim=-1).cpu()
                mask = padded_labels.sum(dim=-1) > 0
            else:
                true_cls = padded_labels
                mask = padded_labels != -1
            all_preds.extend(preds[mask].tolist())
            all_true.extend(true_cls[mask].tolist())

    print(classification_report(all_true, all_preds,
                                 target_names=_LABEL_ORDER, zero_division=0))
    acc = sum(p == t for p, t in zip(all_preds, all_true)) / max(len(all_true), 1)
    print(f'Overall accuracy: {acc:.4f}')
    return acc


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_rnn_model(model, norm_stats, model_dir, history=None, training_time_sec=None, filename_ext=""):
    """Save model weights, norm stats, architecture config, and optional training history.

    Writes to model_dir:
        rnn_model{ext}.pt            — PyTorch state dict
        rnn_norm_stats{ext}.json     — per-feature mean/std
        rnn_model_config{ext}.json   — architecture params for load_rnn_model
        rnn_history{ext}.json        — train_loss, val_loss, val_acc per epoch, and
                                       training_time_sec (if history or time provided)
    """
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f'rnn_model{filename_ext}.pt'))
    with open(os.path.join(model_dir, f'rnn_norm_stats{filename_ext}.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)
    config = {
        'input_size':  model.lstm.input_size,
        'hidden_size': model.hidden_size,
        'num_layers':  model.num_layers,
        'num_classes': model.head.out_features,
    }
    with open(os.path.join(model_dir, f'rnn_model_config{filename_ext}.json'), 'w') as f:
        json.dump(config, f, indent=2)
    saved = f'rnn_model{filename_ext}.pt + rnn_norm_stats{filename_ext}.json + rnn_model_config{filename_ext}.json'
    if history is not None or training_time_sec is not None:
        history_payload = dict(history) if history is not None else {}
        if training_time_sec is not None:
            history_payload['training_time_sec'] = training_time_sec
        with open(os.path.join(model_dir, f'rnn_history{filename_ext}.json'), 'w') as f:
            json.dump(history_payload, f, indent=2)
        saved += f' + rnn_history{filename_ext}.json'
    print(f'  Saved {saved} → {model_dir}')


def load_rnn_model(model_dir, filename_ext=""):
    """Load RNN model, norm stats, and training history (if saved) from model_dir.

    Returns:
        (model, norm_stats, history): RNNPhenologyModel in eval mode, norm stats dict,
        and history dict with 'train_loss'/'val_acc' lists (or None if not found).
    """
    with open(os.path.join(model_dir, f'rnn_model_config{filename_ext}.json')) as f:
        config = json.load(f)
    model = RNNPhenologyModel(**config)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, f'rnn_model{filename_ext}.pt'), map_location='cpu'))
    model.eval()
    with open(os.path.join(model_dir, f'rnn_norm_stats{filename_ext}.json')) as f:
        norm_stats = json.load(f)
    history_path = os.path.join(model_dir, f'rnn_history{filename_ext}.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = None
    return model, norm_stats, history


# ---------------------------------------------------------------------------
# Serving: predict from rolling pixel state
# ---------------------------------------------------------------------------

def predict_rnn_from_pixel_state(state_path, date_str, data_dir=None,
                                  greendown_dir=None, model_dir=None,filename_ext=""):
    """Predict phenological state using the trained LSTM and rolling pixel state.

    Assembles per-pixel observation sequences (oldest→newest) from the stored
    observation window, computes the same 9 features used during training
    (including recent daily mean temperature), normalizes, and runs the LSTM.
    The prediction at the final (most recent) time step is used.

    Args:
        state_path:    Path to pixel_state_{year}.npz.
        date_str:      ISO date string, e.g. '2026-09-15'.
        data_dir:      Directory with reference GeoTIFFs and cdd_state_{year}.npz.
        greendown_dir: Directory with transition GeoTIFFs and avg assets.
        model_dir:     Directory with rnn_model.pt, rnn_norm_stats.json,
                       and rnn_model_config.json.

    Returns:
        (pred_grid, forest_mask, transform, crs) where pred_grid is an (h, w)
        object array of label strings, or None if model files are absent.
    """
    import datetime
    import rasterio
    from torch.nn.utils.rnn import pack_sequence
    from predict_for_date import (
        _day_length_vec, _per_pixel_avg_middle,
        _build_cross_year_transition_lookup, _load_global_avg_middle,
        _load_lat_lon_arrays,
    )
    from gridmet_utils import load_cdd_state, cdd_state_tmean_at_doys
    from constants import DATA_DIR, GREENDOWN_DIR, MODEL_DIR
    if data_dir is None:
        data_dir = DATA_DIR
    if greendown_dir is None:
        greendown_dir = GREENDOWN_DIR
    if model_dir is None:
        model_dir = MODEL_DIR

    if not os.path.exists(os.path.join(model_dir, f'rnn_model{filename_ext}.pt')):
        print('  RNN model not found — skipping RNN prediction.')
        return None

    model, norm_stats = load_rnn_model(model_dir)
    norm_mean = np.array([norm_stats[c]['mean'] for c in RNN_FEATURE_COLS], dtype=np.float32)
    norm_std  = np.array([norm_stats[c]['std']  for c in RNN_FEATURE_COLS], dtype=np.float32)

    date = datetime.date.fromisoformat(date_str)
    year = date.year

    # Load pixel state
    raw = dict(np.load(state_path))
    h, w = raw['evi_0'].shape
    if 'evi_w' in raw:
        evi_w  = raw['evi_w'].astype(np.float32)
        ndvi_w = raw['ndvi_w'].astype(np.float32)
        doy_w  = raw['doy_w'].astype(np.float32)
    else:
        N0 = 3
        evi_w  = np.full((h, w, N0), np.nan, dtype=np.float32)
        ndvi_w = np.full((h, w, N0), np.nan, dtype=np.float32)
        doy_w  = np.full((h, w, N0), np.nan, dtype=np.float32)
        for k in range(N0):
            evi_w[:, :, k]  = raw.get(f'evi_{k}', np.full((h, w), np.nan))
            ndvi_w[:, :, k] = raw.get(f'ndvi_{k}', np.full((h, w), np.nan))
            doy_w[:, :, k]  = raw.get(f'doy_{k}', np.full((h, w), np.nan))

    ref_path = os.path.join(data_dir, f'hls_indices_ref_{year}.tif')
    if not os.path.exists(ref_path):
        ref_path = os.path.join(data_dir, 'hls_indices_ref_current.tif')
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs       = src.crs

    lat_array, lon_array = _load_lat_lon_arrays(data_dir, year)
    cross_year_lookup    = _build_cross_year_transition_lookup(greendown_dir)
    global_avg_middle    = _load_global_avg_middle(greendown_dir)
    avg_middle           = _per_pixel_avg_middle(cross_year_lookup, greendown_dir, h, w,
                                                  exclude_year=year)

    cdd_state_path = os.path.join(data_dir, f'cdd_state_{year}.npz')
    cdd_state = load_cdd_state(cdd_state_path) if os.path.exists(cdd_state_path) else None

    forest_mask = np.isfinite(evi_w[:, :, 0]) & (evi_w[:, :, 0] > 0)
    rows_idx, cols_idx = np.where(forest_mask)
    n_px = len(rows_idx)

    if n_px == 0:
        return np.full((h, w), 'unknown', dtype=object), forest_mask, transform, crs

    # Pass 1: collect valid observations per pixel (oldest→newest)
    pixel_infos   = []
    all_doys_flat = []
    all_lats_flat = []
    all_lons_flat = []

    for i in range(n_px):
        ri, ci   = rows_idx[i], cols_idx[i]
        evi_i    = evi_w[ri, ci, :]
        ndvi_i   = ndvi_w[ri, ci, :]
        doy_i    = doy_w[ri, ci, :]
        valid    = (np.isfinite(evi_i) & (evi_i > 0) &
                    np.isfinite(ndvi_i) & np.isfinite(doy_i))
        if not valid.any():
            pixel_infos.append(None)
            continue
        order    = np.argsort(doy_i[valid])
        evi_seq  = evi_i[valid][order]
        ndvi_seq = ndvi_i[valid][order]
        doy_seq  = doy_i[valid][order]
        lat_i    = float(lat_array[ri, ci])
        lon_i    = float(lon_array[ri, ci])
        avg_m    = float(avg_middle[ri, ci])
        if not np.isfinite(avg_m):
            avg_m = float(global_avg_middle) if not np.isnan(global_avg_middle) else float(doy_seq[-1])
        pixel_infos.append({'evi': evi_seq, 'ndvi': ndvi_seq, 'doy': doy_seq,
                            'lat': lat_i, 'lon': lon_i, 'avg_m': avg_m})
        all_doys_flat.extend(doy_seq.tolist())
        all_lats_flat.extend([lat_i] * len(doy_seq))
        all_lons_flat.extend([lon_i] * len(doy_seq))

    # Pass 2: vectorized T_mean lookup for all valid observations at once.
    # Sample at doy-1 to match training (build_data_table samples temperature at
    # the previous day, mirroring gridMET's ~1-day reporting lag).
    n_obs   = len(all_doys_flat)
    all_tmn = np.zeros(n_obs, dtype=np.float32)
    if cdd_state is not None and n_obs > 0:
        da = np.array(all_doys_flat) - 1
        la = np.array(all_lats_flat)
        lo = np.array(all_lons_flat)
        all_tmn = np.nan_to_num(
            cdd_state_tmean_at_doys(cdd_state, da, la, lo), nan=0.0).astype(np.float32)

    # Pass 3: build normalized feature tensors, run LSTM in batches
    n_feats           = len(RNN_FEATURE_COLS)
    tensors           = []
    valid_pixel_idx   = []
    ptr = 0

    for i, info in enumerate(pixel_infos):
        if info is None:
            continue
        evi_seq  = info['evi']
        ndvi_seq = info['ndvi']
        doy_seq  = info['doy']
        T        = len(doy_seq)
        feats    = np.zeros((T, n_feats), dtype=np.float32)
        feats[:, 0] = evi_seq
        feats[:, 1] = ndvi_seq
        if T > 1:
            feats[1:, 2] = evi_seq[1:]  - evi_seq[:-1]
            feats[1:, 4] = ndvi_seq[1:] - ndvi_seq[:-1]
        if T > 2:
            feats[2:, 3] = evi_seq[2:]  - evi_seq[:-2]
            feats[2:, 5] = ndvi_seq[2:] - ndvi_seq[:-2]
        feats[:, 6] = _day_length_vec(doy_seq, info['lat'])
        feats[:, 7] = doy_seq - info['avg_m']
        feats[:, 8] = all_tmn[ptr:ptr + T]
        ptr += T
        feats = np.nan_to_num((feats - norm_mean) / norm_std, nan=0.0)
        tensors.append(torch.tensor(feats, dtype=torch.float32))
        valid_pixel_idx.append(i)

    BATCH    = 512
    all_pred = []
    for b0 in range(0, len(tensors), BATCH):
        batch = tensors[b0:b0 + BATCH]
        with torch.no_grad():
            packed        = pack_sequence(batch, enforce_sorted=False)
            logits, lens  = model(packed)
            for j in range(len(batch)):
                last_t = int(lens[j]) - 1
                all_pred.append(_INT_TO_LABEL[int(logits[j, last_t].argmax())])

    pred_grid = np.full((h, w), 'unknown', dtype=object)
    for pred_i, pix_i in enumerate(valid_pixel_idx):
        pred_grid[rows_idx[pix_i], cols_idx[pix_i]] = all_pred[pred_i]

    return pred_grid, forest_mask, transform, crs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_soft_label(doy, start, middle, end, ci_start, ci_middle, ci_end):
    """Compute a probabilistic label distribution from transition dates and CI widths.

    Treats each transition boundary as normally distributed with σ = CI_width / (2×1.96),
    then derives class probabilities via the standard normal CDF.

    Returns:
        float32 array of shape (4,) summing to 1: [P(before), P(early), P(late), P(after)].
    """
    from scipy.stats import norm as _norm
    _sigma = lambda ci: max(float(ci) / (2 * 1.96), 1e-6)
    p_past_start  = _norm.cdf(doy, loc=start,  scale=_sigma(ci_start))
    p_past_middle = _norm.cdf(doy, loc=middle, scale=_sigma(ci_middle))
    p_past_end    = _norm.cdf(doy, loc=end,    scale=_sigma(ci_end))
    probs = np.array([
        1.0 - p_past_start,
        p_past_start  - p_past_middle,
        p_past_middle - p_past_end,
        p_past_end,
    ], dtype=np.float32)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    return probs / total if total > 0 else np.full(4, 0.25, dtype=np.float32)


def _move_packed(packed, device):
    """Move a PackedSequence to `device`."""
    from torch.nn.utils.rnn import PackedSequence
    return PackedSequence(
        packed.data.to(device),
        packed.batch_sizes,
        packed.sorted_indices,
        packed.unsorted_indices,
    )


def _eval_accuracy(model, sequences, device, batch_size):
    """Quick scalar accuracy for logging during training."""
    model.eval()
    correct = total = 0
    loader = DataLoader(
        _SequenceDataset(sequences), batch_size=batch_size,
        shuffle=False, collate_fn=collate_rnn_batch,
    )
    with torch.no_grad():
        for packed, padded_labels, _ in loader:
            packed = _move_packed(packed, device)
            logits, _ = model(packed)
            preds = logits.argmax(dim=-1).cpu()
            if padded_labels.dim() == 3:  # soft labels
                true_cls = padded_labels.argmax(dim=-1).cpu()
                mask = padded_labels.sum(dim=-1) > 0
            else:
                true_cls = padded_labels
                mask = padded_labels != -1
            correct += (preds[mask] == true_cls[mask]).sum().item()
            total   += mask.sum().item()
    model.train()
    return correct / max(total, 1)


def _eval_loss(model, sequences, device, batch_size):
    """Compute mean cross-entropy loss over sequences for logging during training."""
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    loader = DataLoader(
        _SequenceDataset(sequences), batch_size=batch_size,
        shuffle=False, collate_fn=collate_rnn_batch,
    )
    with torch.no_grad():
        for packed, padded_labels, _ in loader:
            packed        = _move_packed(packed, device)
            padded_labels = padded_labels.to(device)
            logits, _     = model(packed)
            if padded_labels.dim() == 3:  # soft labels
                log_probs = F.log_softmax(logits, dim=-1)
                mask_s = padded_labels.sum(dim=-1) > 0
                loss = -(padded_labels * log_probs).sum(dim=-1)[mask_s].mean()
            else:
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    padded_labels.reshape(-1),
                )
            total_loss += loss.item()
            n_batches  += 1
    model.train()
    return total_loss / max(n_batches, 1)
