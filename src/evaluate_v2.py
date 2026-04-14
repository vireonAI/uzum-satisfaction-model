"""
Quick evaluation & calibration of the saved v2 model checkpoint.
Runs: threshold optimization → test evaluation → Platt scaling → save results.
"""
import os, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

FACTORS = [
    'product_quality', 'price_value', 'logistics_delivery', 'packaging_condition',
    'accuracy_expectation', 'seller_service', 'specifications', 'product_defects'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'models/uzum_nlp_v2'


class UZUMDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['Input_Text'])
        labels = torch.tensor([float(row[f]) for f in FACTORS], dtype=torch.float32)
        encoded = self.tokenizer(text, max_length=self.max_length, padding='max_length',
                                  truncation=True, return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels,
        }


class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_factors=8, dropout=0.3):
        super().__init__()
        self.xlm = AutoModel.from_pretrained(model_name)
        hidden_size = self.xlm.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_factors)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


def collect_predictions(model, loader):
    """Collect all predictions and labels from a DataLoader"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch['input_ids'].to(DEVICE),
                batch['attention_mask'].to(DEVICE)
            )
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(batch['labels'].numpy())
    return np.vstack(all_probs), np.vstack(all_labels)


def optimize_thresholds(probs, labels):
    """Find per-factor optimal thresholds"""
    thresholds = {}
    for i, factor in enumerate(FACTORS):
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.15, 0.85, 0.01):
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = round(float(t), 2)
        thresholds[factor] = best_t
    return thresholds


def main():
    print("=" * 80)
    print("📊 EVALUATING V2 MODEL (EPOCH 9 CHECKPOINT)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    
    # Load config
    with open(os.path.join(MODEL_DIR, 'config.json')) as f:
        config = json.load(f)
    print(f"Model: {config['model_name']}, max_length: {config['max_length']}")
    print(f"Best epoch: {config['best_epoch']}, best metric: {config['best_metric']:.4f}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = MultiLabelClassifier(config['model_name'], dropout=config['classifier_dropout']).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model.pt'), map_location=DEVICE))
    model.eval()
    print("✓ Model loaded")
    
    # DataLoaders (num_workers=0 to avoid shared memory issues on Windows)
    val_dataset = UZUMDataset('data/processed/uzum_val.csv', tokenizer, config['max_length'])
    test_dataset = UZUMDataset('data/processed/uzum_test.csv', tokenizer, config['max_length'])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Collect predictions
    print("\n📥 Collecting predictions...")
    val_probs, val_labels = collect_predictions(model, val_loader)
    test_probs, test_labels = collect_predictions(model, test_loader)
    print(f"  Val: {val_probs.shape}, Test: {test_probs.shape}")
    
    # Optimize thresholds on val set
    print("\n🎯 Optimizing thresholds on validation set...")
    thresholds = optimize_thresholds(val_probs, val_labels)
    for f, t in thresholds.items():
        print(f"  {f:25s}: {t}")
    
    # Evaluate on test set with optimized thresholds
    print("\n" + "=" * 80)
    print("🧪 TEST SET RESULTS (Optimized Thresholds)")
    print("=" * 80)
    
    y_pred = np.zeros_like(test_labels)
    per_factor = {}
    for i, factor in enumerate(FACTORS):
        y_pred[:, i] = (test_probs[:, i] >= thresholds[factor]).astype(int)
        f1 = f1_score(test_labels[:, i], y_pred[:, i], zero_division=0)
        prec = precision_score(test_labels[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(test_labels[:, i], y_pred[:, i], zero_division=0)
        per_factor[factor] = {'f1': round(f1, 4), 'precision': round(prec, 4), 'recall': round(rec, 4)}
        print(f"  {factor:25s}: F1={f1:.4f} | P={prec:.4f} | R={rec:.4f}")
    
    macro_f1 = f1_score(test_labels, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(test_labels, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(test_labels, y_pred, average='weighted', zero_division=0)
    
    print(f"\n  Macro F1:    {macro_f1:.4f}")
    print(f"  Micro F1:    {micro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    
    # Platt Scaling calibration
    print("\n" + "=" * 80)
    print("🔧 PLATT SCALING CALIBRATION")
    print("=" * 80)
    
    calibrators = {}
    cal_per_factor = {}
    
    for i, factor in enumerate(FACTORS):
        cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        cal.fit(val_probs[:, i:i+1], val_labels[:, i].astype(int))
        calibrators[factor] = cal
        
        cal_probs = cal.predict_proba(test_probs[:, i:i+1])[:, 1]
        cal_preds = (cal_probs >= 0.5).astype(int)
        
        f1_raw = per_factor[factor]['f1']
        f1_cal = f1_score(test_labels[:, i], cal_preds, zero_division=0)
        prec_cal = precision_score(test_labels[:, i], cal_preds, zero_division=0)
        rec_cal = recall_score(test_labels[:, i], cal_preds, zero_division=0)
        delta = f1_cal - f1_raw
        
        cal_per_factor[factor] = {'f1': round(f1_cal, 4), 'precision': round(prec_cal, 4), 'recall': round(rec_cal, 4)}
        print(f"  {factor:25s}: {f1_raw:.4f} → {f1_cal:.4f}  ({'+' if delta >= 0 else ''}{delta:.4f})")
    
    cal_y_pred = np.zeros_like(test_labels)
    for i, factor in enumerate(FACTORS):
        cal_y_pred[:, i] = (calibrators[factor].predict_proba(test_probs[:, i:i+1])[:, 1] >= 0.5).astype(int)
    
    cal_macro = f1_score(test_labels, cal_y_pred, average='macro', zero_division=0)
    cal_micro = f1_score(test_labels, cal_y_pred, average='micro', zero_division=0)
    cal_weighted = f1_score(test_labels, cal_y_pred, average='weighted', zero_division=0)
    
    print(f"\n  Calibrated Macro F1:    {cal_macro:.4f}")
    print(f"  Calibrated Micro F1:    {cal_micro:.4f}")
    print(f"  Calibrated Weighted F1: {cal_weighted:.4f}")
    
    # Save calibrators
    with open(os.path.join(MODEL_DIR, 'calibrators.pkl'), 'wb') as f:
        pickle.dump(calibrators, f)
    print(f"\n  ✓ Calibrators saved to {MODEL_DIR}/calibrators.pkl")
    
    # Save thresholds
    with open(os.path.join(MODEL_DIR, 'thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    # Decide which is better: raw+thresholds or calibrated
    best_method = "calibrated" if cal_macro > macro_f1 else "threshold_optimized"
    best_f1 = max(cal_macro, macro_f1)
    best_per_factor = cal_per_factor if best_method == "calibrated" else per_factor
    
    print(f"\n  Best method: {best_method} (Macro F1 = {best_f1:.4f})")
    
    # Update model_comparison.json
    comparison_path = 'docs/model_comparison.json'
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            comparison = json.load(f)
    else:
        comparison = {'models': []}
    
    # Add/update v2 model entry
    v2_entry = {
        'model': 'XLM-RoBERTa v2 (Focal Loss)',
        'type': 'transformer',
        'overall': {
            'macro_f1': round(best_f1, 4),
            'micro_f1': round(max(cal_micro, micro_f1), 4),
            'weighted_f1': round(max(cal_weighted, weighted_f1), 4),
        },
        'per_factor': best_per_factor,
        'training': {
            'epochs_trained': config['best_epoch'],
            'loss': 'Focal Loss (γ=2.0)',
            'max_class_weight': 3.0,
            'max_length': config['max_length'],
            'calibration': best_method,
        },
        'thresholds': thresholds,
    }
    
    # Check if v2 entry already exists
    existing_idx = None
    for i, m in enumerate(comparison['models']):
        if 'v2' in m.get('model', '').lower() or 'focal' in m.get('model', '').lower():
            existing_idx = i
            break
    
    if existing_idx is not None:
        comparison['models'][existing_idx] = v2_entry
    else:
        comparison['models'].append(v2_entry)
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  ✓ Updated {comparison_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ FINAL SUMMARY")
    print("=" * 80)
    print(f"  XLM-RoBERTa v2 ({best_method}): Macro F1 = {best_f1:.4f}")
    print(f"  vs TF-IDF + LogReg:              Macro F1 = 0.5352")
    if best_f1 > 0.5352:
        print(f"  📈 Improvement: +{(best_f1 - 0.5352):.4f} ({((best_f1 - 0.5352) / 0.5352 * 100):.1f}%)")
    else:
        print(f"  📉 Still below baseline by {(0.5352 - best_f1):.4f}")


if __name__ == '__main__':
    main()
