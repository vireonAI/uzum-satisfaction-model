"""
=============================================================================
🔬 BASELINE MODEL COMPARISON
=============================================================================

Compares XLM-RoBERTa against simpler baseline models:
  1. TF-IDF + Logistic Regression (One-vs-Rest)
  2. TF-IDF + Random Forest (One-vs-Rest)
  3. XLM-RoBERTa (our fine-tuned model via inference_api.py)

Usage:
    python src/baseline_comparison.py

Output:
    docs/model_comparison.json
=============================================================================
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "model_comparison.json"

FACTORS = [
    'product_quality', 'price_value', 'logistics_delivery',
    'packaging_condition', 'accuracy_expectation', 'seller_service',
    'specifications', 'product_defects'
]


def load_data():
    """Load train and test splits"""
    logger.info("📂 Loading data splits...")
    
    train_df = pd.read_csv(DATA_DIR / "uzum_train.csv")
    test_df = pd.read_csv(DATA_DIR / "uzum_test.csv")
    
    logger.info(f"   Train: {len(train_df):,} samples")
    logger.info(f"   Test:  {len(test_df):,} samples")
    
    return train_df, test_df


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict:
    """Calculate per-factor and overall metrics"""
    results = {
        'model': model_name,
        'per_factor': {},
        'overall': {}
    }
    
    for i, factor in enumerate(FACTORS):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]
        
        f1 = f1_score(true_col, pred_col, zero_division=0)
        precision = precision_score(true_col, pred_col, zero_division=0)
        recall = recall_score(true_col, pred_col, zero_division=0)
        
        results['per_factor'][factor] = {
            'f1': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'support': int(true_col.sum())
        }
    
    # Overall metrics
    results['overall'] = {
        'macro_f1': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'micro_f1': round(f1_score(y_true, y_pred, average='micro', zero_division=0), 4),
        'weighted_f1': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
    }
    
    return results


def train_tfidf_logreg(train_df, test_df):
    """Baseline 1: TF-IDF + Logistic Regression"""
    logger.info("🔧 Training TF-IDF + Logistic Regression...")
    
    start = time.time()
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train = tfidf.fit_transform(train_df['Input_Text'].fillna(''))
    X_test = tfidf.transform(test_df['Input_Text'].fillna(''))
    
    y_train = train_df[FACTORS].values
    y_test = test_df[FACTORS].values
    
    # Train One-vs-Rest
    model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', class_weight='balanced'),
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start
    
    # Predict
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start) / len(test_df) * 1000  # ms per sample
    
    results = evaluate_model(y_test, y_pred, "TF-IDF + LogReg")
    results['train_time_sec'] = round(train_time, 1)
    results['inference_ms'] = round(inference_time, 2)
    results['params'] = f"~{X_train.shape[1] * len(FACTORS):,}"
    
    logger.info(f"   ✅ Done in {train_time:.1f}s | Macro F1: {results['overall']['macro_f1']}")
    return results, y_test


def train_tfidf_rf(train_df, test_df):
    """Baseline 2: TF-IDF + Random Forest"""
    logger.info("🌲 Training TF-IDF + Random Forest...")
    
    start = time.time()
    
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train = tfidf.fit_transform(train_df['Input_Text'].fillna(''))
    X_test = tfidf.transform(test_df['Input_Text'].fillna(''))
    
    y_train = train_df[FACTORS].values
    y_test = test_df[FACTORS].values
    
    model = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=200, max_depth=30, class_weight='balanced', n_jobs=-1, random_state=42),
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start) / len(test_df) * 1000
    
    results = evaluate_model(y_test, y_pred, "TF-IDF + RandomForest")
    results['train_time_sec'] = round(train_time, 1)
    results['inference_ms'] = round(inference_time, 2)
    results['params'] = f"~{200 * len(FACTORS):,} trees"
    
    logger.info(f"   ✅ Done in {train_time:.1f}s | Macro F1: {results['overall']['macro_f1']}")
    return results


def evaluate_xlm_roberta(test_df):
    """Evaluate our fine-tuned XLM-RoBERTa on the test set"""
    logger.info("🤖 Evaluating XLM-RoBERTa on test set...")
    
    # Add src to path for imports
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.inference_api import UZUMInferenceAPI
    
    model_dir = str(PROJECT_ROOT / "models" / "uzum_nlp_v3")
    
    start = time.time()
    api = UZUMInferenceAPI(model_dir=model_dir)
    load_time = time.time() - start
    logger.info(f"   Model loaded in {load_time:.1f}s")
    
    import torch
    y_test = test_df[FACTORS].values
    texts  = test_df['Input_Text'].fillna('').tolist()
    total  = len(texts)
    BATCH  = 64
    all_probs = []

    start = time.time()
    for i in range(0, total, BATCH):
        batch = texts[i:i+BATCH]
        enc = api.tokenizer(
            batch, padding=True, truncation=True,
            max_length=api.max_length, return_tensors='pt'
        ).to(api.device)
        with torch.no_grad():
            if api.is_v3:
                logits, _ = api.model(enc['input_ids'], enc['attention_mask'])
            else:
                logits = api.model(enc['input_ids'], enc['attention_mask'])
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        if (i // BATCH + 1) % 10 == 0:
            logger.info(f"   Processed {min(i+BATCH, total)}/{total}...")

    import numpy as _np
    all_probs = _np.vstack(all_probs)
    y_pred = _np.zeros_like(y_test)
    for j, factor in enumerate(FACTORS):
        t = api.thresholds.get(factor, 0.5)
        y_pred[:, j] = (all_probs[:, j] >= t).astype(int)

    inference_time = (time.time() - start) / total * 1000
    
    results = evaluate_model(y_test, y_pred, "XLM-RoBERTa V3 (Multi-Task)")
    results['train_time_sec'] = "~60 min (RTX 3060)"
    results['inference_ms'] = round(inference_time, 2)
    results['params'] = "125M"
    
    logger.info(f"   ✅ Done | Macro F1: {results['overall']['macro_f1']}")
    return results


def print_comparison_table(all_results: List[Dict]):
    """Pretty-print the comparison"""
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Model':<30} {'Macro F1':>10} {'Micro F1':>10} {'Weighted F1':>12} {'Speed (ms)':>12}")
    print("-" * 74)
    
    for r in all_results:
        name = r['model']
        o = r['overall']
        speed = r['inference_ms']
        print(f"{name:<30} {o['macro_f1']:>10.4f} {o['micro_f1']:>10.4f} {o['weighted_f1']:>12.4f} {speed:>12}")
    
    # Per-factor breakdown
    print(f"\n{'Factor':<25}", end="")
    for r in all_results:
        print(f" {r['model'][:15]:>15}", end="")
    print()
    print("-" * (25 + 15 * len(all_results)))
    
    for factor in FACTORS:
        print(f"{factor:<25}", end="")
        for r in all_results:
            f1 = r['per_factor'][factor]['f1']
            print(f" {f1:>15.4f}", end="")
        print()


def main():
    train_df, test_df = load_data()
    
    all_results = []
    
    # 1. TF-IDF + LogReg
    logreg_results, y_test = train_tfidf_logreg(train_df, test_df)
    all_results.append(logreg_results)
    
    # 2. TF-IDF + Random Forest
    rf_results = train_tfidf_rf(train_df, test_df)
    all_results.append(rf_results)
    
    # 3. XLM-RoBERTa
    xlm_results = evaluate_xlm_roberta(test_df)
    all_results.append(xlm_results)
    
    # Print comparison
    print_comparison_table(all_results)
    
    # Save to JSON
    output = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_set_size': len(test_df),
        'factors': FACTORS,
        'dataset_stats': {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'class_distribution': {
                factor: {
                    'positive_rate': round(test_df[factor].mean(), 4),
                    'count': int(test_df[factor].sum())
                } for factor in FACTORS
            }
        },
        'models': all_results
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 Results saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
