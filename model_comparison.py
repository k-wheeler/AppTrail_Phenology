import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from rnn_model import _SequenceDataset, collate_rnn_batch, _move_packed

LABEL_ORDER = ['before', 'early', 'late', 'after']
device = "cpu"

def _dt_report(model, x_tr, y_tr, x_te, y_te):
    tr = classification_report(y_tr, model.predict(x_tr), target_names=LABEL_ORDER,
                               output_dict=True, zero_division=0)
    te = classification_report(y_te, model.predict(x_te), target_names=LABEL_ORDER,
                               output_dict=True, zero_division=0)
    return tr, te

def _rnn_report(model, sequences):
    loader = DataLoader(_SequenceDataset(sequences), batch_size=64,
                        shuffle=False, collate_fn=collate_rnn_batch)
    preds_all, true_all = [], []
    model.eval()
    with torch.no_grad():
        for packed, padded_labels, _ in loader:
            packed = _move_packed(packed, device)
            logits, _ = model(packed)
            preds = logits.argmax(dim=-1).cpu()
            mask  = padded_labels != -1
            preds_all.extend(preds[mask].tolist())
            true_all.extend(padded_labels[mask].tolist())
    return classification_report(true_all, preds_all, target_names=LABEL_ORDER,
                                 output_dict=True, zero_division=0)

    
def _fmt_time(s):
    if np.isnan(s):
        return 'N/A'
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{sec:02d}'