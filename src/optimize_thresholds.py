"""
=============================================================================
optimize_thresholds.py — Per-factor threshold optimizer for V3 model
=============================================================================

Sweeps thresholds 0.05→0.95 (step 0.01) on the VALIDATION set.
Writes the best thresholds back to models/uzum_nlp_v3/thresholds.json.

Then re-evaluates both old and new thresholds on the TEST set to prove
the improvement.

Usage:
    python src/optimize_thresholds.py
=============================================================================
"""

import os, sys, json
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import XLMRobertaTokenizerFast
from src.inference_api import UZUMInferenceAPI, FACTORS

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_DIR   = PROJECT / 'models' / 'uzum_nlp_v3'
VAL_CSV     = PROJECT / 'data' / 'processed' / 'uzum_val.csv'
TEST_CSV    = PROJECT / 'data' / 'processed' / 'uzum_test.csv'
BATCH_SIZE  = 32
THRESHOLD_RANGE = np.arange(0.05, 0.96, 0.01)

# ── Load model ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  UZUM V3 THRESHOLD OPTIMIZER")
print("="*70)
print(f"  Model : {MODEL_DIR}")
print(f"  Val   : {VAL_CSV}")
print(f"  Test  : {TEST_CSV}")
print()

api = UZUMInferenceAPI(str(MODEL_DIR))
model     = api.model
tokenizer = api.tokenizer
device    = api.device
max_len   = api.max_length

model.eval()

# ── Inference helper ────────────────────────────────────────────────────────
def get_raw_probs(texts: list) -> np.ndarray:
    """Return sigmoid probabilities shape (N, 8)."""
    all_probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            if api.is_v3:
                factor_logits, _ = model(enc['input_ids'], enc['attention_mask'])
            else:
                factor_logits = model(enc['input_ids'], enc['attention_mask'])
        probs = torch.sigmoid(factor_logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)          # (N, 8)


def evaluate_with_thresholds(texts, labels, thresholds: dict, label=""):
    """Return per-factor and macro/micro F1 for a given threshold dict."""
    probs = get_raw_probs(texts)
    preds = np.zeros_like(probs, dtype=int)
    for j, factor in enumerate(FACTORS):
        preds[:, j] = (probs[:, j] >= thresholds[factor]).astype(int)

    results = {}
    for j, factor in enumerate(FACTORS):
        f1 = f1_score(labels[:, j], preds[:, j], zero_division=0)
        p  = precision_score(labels[:, j], preds[:, j], zero_division=0)
        r  = recall_score(labels[:, j], preds[:, j], zero_division=0)
        results[factor] = dict(f1=f1, precision=p, recall=r,
                               threshold=thresholds[factor])

    y_true_flat = labels.ravel()
    y_pred_flat = preds.ravel()
    macro  = f1_score(labels, preds, average='macro',  zero_division=0)
    micro  = f1_score(labels, preds, average='micro',  zero_division=0)
    weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    return results, macro, micro, weighted


# ── Load datasets ───────────────────────────────────────────────────────────
def load_split(csv_path):
    df = pd.read_csv(csv_path)
    texts  = df['Input_Text'].fillna('').tolist()
    labels = df[FACTORS].values.astype(int)
    return texts, labels

print("Loading validation set ...")
val_texts, val_labels = load_split(VAL_CSV)
print(f"  {len(val_texts)} reviews, {val_labels.shape[1]} factors\n")

print("Loading test set ...")
test_texts, test_labels = load_split(TEST_CSV)
print(f"  {len(test_texts)} reviews\n")


# ── Evaluate with ORIGINAL thresholds first ─────────────────────────────────
original_thresholds = api.thresholds.copy()
print("="*70)
print("  BASELINE (original thresholds from training)")
print("="*70)
base_results, base_macro, base_micro, base_weighted = evaluate_with_thresholds(
    test_texts, test_labels, original_thresholds, label="TEST/original"
)
print(f"  {'Factor':<28} {'F1':>6} {'P':>6} {'R':>6} {'t':>5}")
print("  " + "-"*55)
for factor in FACTORS:
    r = base_results[factor]
    print(f"  {factor:<28} {r['f1']:.4f} {r['precision']:.4f} {r['recall']:.4f} {r['threshold']:.2f}")
print(f"\n  Macro F1   : {base_macro:.4f}")
print(f"  Micro F1   : {base_micro:.4f}")
print(f"  Weighted F1: {base_weighted:.4f}")


# ── SWEEP thresholds on VAL set ─────────────────────────────────────────────
print("\n" + "="*70)
print("  SWEEPING THRESHOLDS ON VALIDATION SET ...")
print("  (0.05 → 0.95, step 0.01)")
print("="*70)

print("\nRunning single inference pass on val set ...")
val_probs = get_raw_probs(val_texts)     # (N, 8) — done ONCE

best_thresholds = {}
sweep_results   = {}

for j, factor in enumerate(FACTORS):
    y_true = val_labels[:, j]
    best_f1 = -1.0
    best_t  = 0.5

    curve = []
    for t in THRESHOLD_RANGE:
        y_pred = (val_probs[:, j] >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        p  = precision_score(y_true, y_pred, zero_division=0)
        r  = recall_score(y_true, y_pred, zero_division=0)
        curve.append((t, f1, p, r))
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    best_thresholds[factor] = round(float(best_t), 2)
    sweep_results[factor]   = dict(best_f1_val=best_f1, best_t=best_t, curve=curve)

    old_t   = original_thresholds[factor]
    old_f1  = f1_score(val_labels[:, j],
                       (val_probs[:, j] >= old_t).astype(int), zero_division=0)
    delta   = best_f1 - old_f1
    arrow   = "+" if delta >= 0 else ""
    print(f"  {factor:<28}: old_t={old_t:.2f} (val F1={old_f1:.4f})  "
          f"  best_t={best_t:.2f} (val F1={best_f1:.4f})  [{arrow}{delta:+.4f}]")

print(f"\n  New thresholds: {best_thresholds}")


# ── Evaluate NEW thresholds on TEST set ──────────────────────────────────────
print("\n" + "="*70)
print("  NEW THRESHOLDS on TEST SET")
print("="*70)
new_results, new_macro, new_micro, new_weighted = evaluate_with_thresholds(
    test_texts, test_labels, best_thresholds, label="TEST/optimized"
)

print(f"  {'Factor':<28} {'F1':>6} {'P':>6} {'R':>6} {'t':>5}  "
      f"{'vs old':>8}")
print("  " + "-"*70)
for factor in FACTORS:
    r_new = new_results[factor]
    r_old = base_results[factor]
    delta = r_new['f1'] - r_old['f1']
    arrow = "+" if delta >= 0 else ""
    print(f"  {factor:<28} {r_new['f1']:.4f} {r_new['precision']:.4f} "
          f"{r_new['recall']:.4f} {r_new['threshold']:.2f}  "
          f"[{arrow}{delta:+.4f}]")

print()
print(f"  Macro F1    :  {base_macro:.4f} --> {new_macro:.4f}  "
      f"[{'+'if new_macro>=base_macro else ''}{new_macro-base_macro:+.4f}]")
print(f"  Micro F1    :  {base_micro:.4f} --> {new_micro:.4f}  "
      f"[{'+'if new_micro>=base_micro else ''}{new_micro-base_micro:+.4f}]")
print(f"  Weighted F1 :  {base_weighted:.4f} --> {new_weighted:.4f}  "
      f"[{'+'if new_weighted>=base_weighted else ''}{new_weighted-base_weighted:+.4f}]")


# ── Save if improved ─────────────────────────────────────────────────────────
THRESHOLDS_PATH = MODEL_DIR / 'thresholds.json'

if new_macro >= base_macro:
    with open(THRESHOLDS_PATH, 'w') as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"\n  [SAVED] New thresholds written to {THRESHOLDS_PATH}")

    # Also patch config.json
    config_path = MODEL_DIR / 'config.json'
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    cfg['thresholds'] = best_thresholds
    cfg['threshold_source'] = 'optimize_thresholds.py (val-set sweep)'
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  [SAVED] config.json updated with new thresholds")
else:
    print(f"\n  [SKIPPED] New thresholds did NOT improve macro F1 — keeping originals.")
    print(f"            (orig={base_macro:.4f} new={new_macro:.4f})")

print("\n" + "="*70)
print("  DONE")
print("="*70 + "\n")
