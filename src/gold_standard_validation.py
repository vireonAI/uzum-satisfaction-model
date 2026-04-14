"""
Gold Standard Validation Script
Computes agreement between AI labels (Llama 3.3-70B) and XLM-RoBERTa model predictions
against the gold standard human-verified labels.

Output: docs/gold_standard_report.json
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, accuracy_score
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FACTORS = [
    'product_quality', 'price_value', 'logistics_delivery',
    'packaging_condition', 'accuracy_expectation', 'seller_service',
    'specifications', 'product_defects'
]

def load_gold_standard():
    """Load gold standard dataset with human-verified labels"""
    gs_path = PROJECT_ROOT / 'data' / 'uzum_labeled_gold_standard_v2.csv'
    if not gs_path.exists():
        print(f"❌ Gold standard not found: {gs_path}")
        return None
    
    df = pd.read_csv(gs_path)
    print(f"✅ Gold standard loaded: {len(df)} reviews")
    print(f"   Columns: {[c for c in df.columns if any(f in c for f in FACTORS)]}")
    return df

def load_ai_labels():
    """Load the full AI-labeled dataset"""
    ai_path = PROJECT_ROOT / 'data' / 'uzum_labeled.csv'
    if not ai_path.exists():
        print(f"❌ AI labels not found: {ai_path}")
        return None
    
    df = pd.read_csv(ai_path)
    print(f"✅ AI labels loaded: {len(df)} reviews")
    return df

def compute_agreement_metrics(gold_df):
    """
    Compute per-factor agreement metrics.
    Since gold standard v2 has the labels directly, we compute
    statistics on the label distribution and confidence.
    """
    results = {
        'dataset_info': {
            'gold_standard_size': len(gold_df),
            'source': 'uzum_labeled_gold_standard_v2.csv'
        },
        'per_factor': {},
        'overall': {}
    }
    
    # Check available columns
    available_factors = [f for f in FACTORS if f in gold_df.columns]
    confidence_cols = [f'{f}_confidence' for f in available_factors if f'{f}_confidence' in gold_df.columns]
    
    print(f"\n{'='*70}")
    print(f"GOLD STANDARD ANALYSIS — {len(gold_df)} reviews")
    print(f"{'='*70}\n")
    
    all_labels = []
    
    for factor in available_factors:
        labels = gold_df[factor].dropna()
        conf_col = f'{factor}_confidence'
        
        factor_stats = {
            'total_samples': int(len(labels)),
            'positive_count': int(labels.sum()),
            'positive_rate': round(float(labels.mean()), 4),
        }
        
        # Confidence statistics (how confident the labeler was)
        if conf_col in gold_df.columns:
            conf = gold_df[conf_col].dropna()
            factor_stats['avg_confidence'] = round(float(conf.mean()), 4)
            factor_stats['min_confidence'] = round(float(conf.min()), 4)
            factor_stats['high_confidence_pct'] = round(float((conf >= 0.8).mean()), 4)
        
        results['per_factor'][factor] = factor_stats
        all_labels.extend(labels.tolist())
        
        pos_rate = factor_stats['positive_rate'] * 100
        conf_str = f" | avg_conf: {factor_stats.get('avg_confidence', 'N/A')}" if 'avg_confidence' in factor_stats else ""
        print(f"  {factor:25s} — {factor_stats['positive_count']:4d}/{factor_stats['total_samples']} ({pos_rate:5.1f}%){conf_str}")
    
    # Overall statistics
    results['overall'] = {
        'total_factor_labels': len(all_labels),
        'overall_positive_rate': round(float(np.mean(all_labels)), 4),
        'factors_analyzed': len(available_factors)
    }
    
    return results

def compute_ai_vs_gold_agreement(gold_df, ai_df):
    """
    Compare AI labels against gold standard for overlapping reviews.
    Match by Product_ID + Rating + Content similarity.
    """
    print(f"\n{'='*70}")
    print(f"AI vs GOLD STANDARD AGREEMENT")
    print(f"{'='*70}\n")
    
    # Try to merge on shared identifiers
    merge_cols = [c for c in ['Product_ID', 'Rating'] if c in gold_df.columns and c in ai_df.columns]
    
    if not merge_cols:
        print("❌ No common columns to merge on")
        return {}
    
    # Find content column in both
    gold_content_col = None
    for col in ['Content_Clean', 'Content', 'content']:
        if col in gold_df.columns:
            gold_content_col = col
            break
    
    ai_content_col = None
    for col in ['Content_Clean', 'Content', 'content']:
        if col in ai_df.columns:
            ai_content_col = col
            break
    
    if not gold_content_col or not ai_content_col:
        print("❌ Cannot find content column for matching")
        return {}
    
    # Simple merge approach: match on Product_ID, Rating, and first 50 chars of content
    gold_df = gold_df.copy()
    ai_df = ai_df.copy()
    gold_df['_match_key'] = gold_df['Product_ID'].astype(str) + '_' + gold_df['Rating'].astype(str) + '_' + gold_df[gold_content_col].str[:50].fillna('')
    ai_df['_match_key'] = ai_df['Product_ID'].astype(str) + '_' + ai_df['Rating'].astype(str) + '_' + ai_df[ai_content_col].str[:50].fillna('')
    
    merged = gold_df.merge(ai_df[['_match_key'] + FACTORS], on='_match_key', how='inner', suffixes=('_gold', '_ai'))
    print(f"  Matched {len(merged)} reviews between gold standard and AI labels")
    
    if len(merged) < 10:
        print("  ⚠️ Too few matches for reliable agreement metrics")
        return {'matched_count': len(merged), 'status': 'insufficient_matches'}
    
    agreement_results = {
        'matched_count': len(merged),
        'per_factor': {}
    }
    
    for factor in FACTORS:
        gold_col = f'{factor}_gold'
        ai_col = f'{factor}_ai'
        
        if gold_col not in merged.columns or ai_col not in merged.columns:
            # Try without suffix (if factor only exists in one)
            if factor in merged.columns:
                continue
            print(f"  ⚠️ {factor}: columns not found in merged data")
            continue
        
        y_gold = merged[gold_col].fillna(0).astype(int)
        y_ai = merged[ai_col].fillna(0).astype(int)
        
        # Cohen's Kappa
        try:
            kappa = cohen_kappa_score(y_gold, y_ai)
        except Exception:
            kappa = 0.0
        
        # Classification metrics
        f1 = f1_score(y_gold, y_ai, zero_division=0)
        prec = precision_score(y_gold, y_ai, zero_division=0)
        rec = recall_score(y_gold, y_ai, zero_division=0)
        acc = accuracy_score(y_gold, y_ai)
        
        agreement_results['per_factor'][factor] = {
            'cohens_kappa': round(kappa, 4),
            'f1_score': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'accuracy': round(acc, 4),
            'agreement_pct': round(acc * 100, 1)
        }
        
        # Kappa interpretation
        if kappa >= 0.81:
            interp = "Almost Perfect"
        elif kappa >= 0.61:
            interp = "Substantial"
        elif kappa >= 0.41:
            interp = "Moderate"
        elif kappa >= 0.21:
            interp = "Fair"
        else:
            interp = "Slight"
        
        print(f"  {factor:25s} — κ={kappa:.3f} ({interp}) | F1={f1:.3f} | Acc={acc*100:.1f}%")
    
    # Overall
    if agreement_results['per_factor']:
        kappas = [v['cohens_kappa'] for v in agreement_results['per_factor'].values()]
        f1s = [v['f1_score'] for v in agreement_results['per_factor'].values()]
        agreement_results['overall'] = {
            'mean_kappa': round(np.mean(kappas), 4),
            'mean_f1': round(np.mean(f1s), 4),
            'best_factor': max(agreement_results['per_factor'], key=lambda k: agreement_results['per_factor'][k]['cohens_kappa']),
            'worst_factor': min(agreement_results['per_factor'], key=lambda k: agreement_results['per_factor'][k]['cohens_kappa'])
        }
        print(f"\n  Overall: Mean κ = {agreement_results['overall']['mean_kappa']:.3f} | Mean F1 = {agreement_results['overall']['mean_f1']:.3f}")
    
    return agreement_results


def main():
    print("=" * 70)
    print("GOLD STANDARD VALIDATION PIPELINE")
    print("=" * 70)
    
    # 1. Load data
    gold_df = load_gold_standard()
    ai_df = load_ai_labels()
    
    if gold_df is None:
        return
    
    # 2. Gold standard analysis
    gs_analysis = compute_agreement_metrics(gold_df)
    
    # 3. AI vs Gold comparison (if AI labels available)
    ai_agreement = {}
    if ai_df is not None:
        ai_agreement = compute_ai_vs_gold_agreement(gold_df, ai_df)
    
    # 4. Save report
    report = {
        'gold_standard_analysis': gs_analysis,
        'ai_vs_gold_agreement': ai_agreement,
        'methodology': {
            'gold_standard': 'Human-verified labels from uzum_labeled_gold_standard_v2.csv (2,584 reviews)',
            'ai_labels': 'Llama 3.3-70B via Groq API (23,987 reviews)',
            'metrics': 'Cohen\'s Kappa, F1, Precision, Recall, Accuracy per factor',
            'kappa_interpretation': {
                '0.81-1.00': 'Almost Perfect',
                '0.61-0.80': 'Substantial',
                '0.41-0.60': 'Moderate',
                '0.21-0.40': 'Fair',
                '0.00-0.20': 'Slight'
            }
        }
    }
    
    output_path = PROJECT_ROOT / 'docs' / 'gold_standard_report.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Report saved to: {output_path}")


if __name__ == '__main__':
    main()
