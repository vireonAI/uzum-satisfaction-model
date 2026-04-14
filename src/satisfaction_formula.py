"""
CUSTOMER SATISFACTION IMPACT MODELING
Determining the Mathematical Formula for Customer Satisfaction

Objective: Identify factor weights that drive Rating (1-5)
Dataset: uzum_labeled.csv (23,987 records)
Method: Random Forest + Permutation Importance + Correlation Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("CUSTOMER SATISFACTION IMPACT FORMULA ANALYSIS")
print("Determining the Physics of Satisfaction")
print("="*80)

# ============================================================================
# STEP 1: PREPROCESSING
# ============================================================================

print("\n[1/6] PREPROCESSING DATA...")
print("-" * 80)

# Load data
df = pd.read_csv('data/uzum_labeled.csv')
print(f"✓ Loaded: {df.shape[0]:,} records, {df.shape[1]} columns")

# Define 8 factors (0/1 binary)
FACTORS = [
    'product_quality',
    'price_value',
    'logistics_delivery',
    'packaging_condition',
    'accuracy_expectation',
    'seller_service',
    'specifications',
    'product_defects'
]

# Extract only required columns
df_clean = df[['Category', 'Rating'] + FACTORS].copy()

# Fill NaN with 0
df_clean[FACTORS] = df_clean[FACTORS].fillna(0).astype(int)
df_clean = df_clean.fillna(0)

# Verify no NaN
print(f"✓ Missing values handled: {df_clean.isnull().sum().sum()} NaN remaining")
print(f"✓ Rating distribution: {df_clean['Rating'].value_counts().sort_index().to_dict()}")
print(f"✓ Factor statistics:")
for factor in FACTORS:
    pct_positive = (df_clean[factor] == 1).sum() / len(df_clean) * 100
    print(f"  {factor:25s} {pct_positive:5.1f}% positive")

# ============================================================================
# STEP 2: GLOBAL IMPACT MODELING (ALL DATA)
# ============================================================================

print("\n[2/6] BUILDING RANDOM FOREST REGRESSOR (GLOBAL)...")
print("-" * 80)

X = df_clean[FACTORS]
y = df_clean['Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate
train_pred = rf_model.predict(X_train)
test_pred = rf_model.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print(f"✓ Model trained on {len(X_train):,} samples")
print(f"✓ Training R² Score: {train_r2:.4f}")
print(f"✓ Testing R² Score: {test_r2:.4f}")
print(f"✓ Testing MAE: {test_mae:.4f} stars")
print(f"✓ Testing RMSE: {test_rmse:.4f} stars")

# ============================================================================
# STEP 3: FEATURE IMPORTANCE EXTRACTION
# ============================================================================

print("\n[3/6] EXTRACTING FEATURE IMPORTANCE...")
print("-" * 80)

# Gini Importance (MDI - Mean Decrease in Impurity)
gini_importance = pd.DataFrame({
    'Feature': FACTORS,
    'Gini_Importance': rf_model.feature_importances_
})
gini_importance['Gini_Importance_Pct'] = gini_importance['Gini_Importance'] / gini_importance['Gini_Importance'].sum() * 100
gini_importance = gini_importance.sort_values('Gini_Importance_Pct', ascending=False)

print("✓ Gini Importance (Mean Decrease in Impurity):")
for idx, row in gini_importance.iterrows():
    print(f"  {row['Feature']:25s} {row['Gini_Importance_Pct']:6.2f}%")

# Permutation Importance (more reliable)
print("\n✓ Computing Permutation Importance (this may take a minute)...")
perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)

perm_df = pd.DataFrame({
    'Feature': FACTORS,
    'Perm_Importance': perm_importance.importances_mean,
    'Perm_Importance_Std': perm_importance.importances_std
})
perm_df['Perm_Importance_Pct'] = perm_df['Perm_Importance'] / perm_df['Perm_Importance'].sum() * 100
perm_df = perm_df.sort_values('Perm_Importance_Pct', ascending=False)

print("✓ Permutation Importance (on test set):")
for idx, row in perm_df.iterrows():
    print(f"  {row['Feature']:25s} {row['Perm_Importance_Pct']:6.2f}% (±{row['Perm_Importance_Std']:5.2f}%)")

# ============================================================================
# STEP 4: CORRELATION & DIRECTIONAL ANALYSIS
# ============================================================================

print("\n[4/6] CORRELATION & DIRECTIONAL ANALYSIS...")
print("-" * 80)

correlation_data = []
for factor in FACTORS:
    # Pearson correlation
    pearson_corr, pearson_pval = pearsonr(df_clean[factor], df_clean['Rating'])
    
    # Spearman correlation
    spearman_corr, spearman_pval = spearmanr(df_clean[factor], df_clean['Rating'])
    
    # Direction
    direction = "POSITIVE ↑" if pearson_corr > 0 else "NEGATIVE ↓"
    
    correlation_data.append({
        'Feature': factor,
        'Pearson_Corr': pearson_corr,
        'Spearman_Corr': spearman_corr,
        'Direction': direction,
        'Abs_Corr': abs(pearson_corr)
    })

corr_df = pd.DataFrame(correlation_data).sort_values('Abs_Corr', ascending=False)

print("✓ Correlation with Rating (Pearson & Spearman):")
for idx, row in corr_df.iterrows():
    print(f"  {row['Feature']:25s} Pearson: {row['Pearson_Corr']:+.4f}  Spearman: {row['Spearman_Corr']:+.4f}  {row['Direction']}")

# ============================================================================
# STEP 5: FORMULA COEFFICIENTS
# ============================================================================

print("\n[5/6] DERIVING SATISFACTION FORMULA...")
print("-" * 80)

# Merge importance data
formula_df = gini_importance.copy()
formula_df = formula_df.merge(perm_df[['Feature', 'Perm_Importance_Pct']], on='Feature')
formula_df = formula_df.merge(corr_df[['Feature', 'Pearson_Corr']], on='Feature')

# Calculate weights (combination of importance and correlation)
# Weight = (Importance % + |Correlation| * 100) / 2
formula_df['Gini_Weight'] = formula_df['Gini_Importance_Pct'] / 100
formula_df['Perm_Weight'] = formula_df['Perm_Importance_Pct'] / 100
formula_df['Corr_Weight'] = np.abs(formula_df['Pearson_Corr']) / corr_df['Abs_Corr'].sum()

# Use Permutation importance (more reliable) as base weight
formula_df['Final_Weight'] = formula_df['Perm_Weight']
formula_df['Direction'] = formula_df['Pearson_Corr'].apply(lambda x: '+' if x > 0 else '-')

# Normalize to sum to 1.0
formula_df['Final_Weight'] = formula_df['Final_Weight'] / formula_df['Final_Weight'].sum()

# Calculate baseline (mean Rating)
baseline = df_clean['Rating'].mean()

print(f"✓ Baseline Satisfaction (Mean Rating): {baseline:.3f} stars")
print(f"\n✓ Feature Weights (from Permutation Importance):")
for idx, row in formula_df.sort_values('Final_Weight', ascending=False).iterrows():
    print(f"  W_{row['Feature']:20s} = {row['Final_Weight']:6.4f}  ({row['Direction']})")

# ============================================================================
# STEP 6: CATEGORY-SPECIFIC ANALYSIS
# ============================================================================

print("\n[6/6] CATEGORY-SPECIFIC ANALYSIS...")
print("-" * 80)

categories = df_clean['Category'].unique()
print(f"✓ Categories found: {', '.join(categories)}")

category_analysis = {}

for category in categories:
    print(f"\n  Analyzing '{category}'...")
    
    df_cat = df_clean[df_clean['Category'] == category]
    X_cat = df_cat[FACTORS]
    y_cat = df_cat['Rating']
    
    # Train category-specific RF
    rf_cat = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_cat.fit(X_cat, y_cat)
    
    # Feature importance
    cat_importance = pd.DataFrame({
        'Feature': FACTORS,
        'Importance_Pct': rf_cat.feature_importances_ / rf_cat.feature_importances_.sum() * 100
    }).sort_values('Importance_Pct', ascending=False)
    
    # Store
    category_analysis[category] = {
        'data': df_cat,
        'model': rf_cat,
        'importance': cat_importance,
        'baseline': y_cat.mean(),
        'count': len(df_cat)
    }
    
    print(f"    Samples: {len(df_cat):,}")
    print(f"    Baseline: {y_cat.mean():.3f} stars")
    print(f"    Top 3 factors:")
    for idx, (_, row) in enumerate(cat_importance.head(3).iterrows()):
        print(f"      {idx+1}. {row['Feature']:25s} {row['Importance_Pct']:6.2f}%")

# ============================================================================
# VISUALIZATION 1: GLOBAL IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Satisfaction Impact Analysis', fontsize=16, fontweight='bold')

# Plot 1: Gini Importance
ax = axes[0, 0]
gini_sorted = gini_importance.sort_values('Gini_Importance_Pct')
colors = plt.cm.RdYlGn(gini_sorted['Gini_Importance_Pct'] / gini_sorted['Gini_Importance_Pct'].max())
ax.barh(gini_sorted['Feature'], gini_sorted['Gini_Importance_Pct'], color=colors)
ax.set_xlabel('Importance (%)', fontweight='bold')
ax.set_title('Gini Importance (MDI)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(gini_sorted['Gini_Importance_Pct']):
    ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)

# Plot 2: Permutation Importance
ax = axes[0, 1]
perm_sorted = perm_df.sort_values('Perm_Importance_Pct')
colors = plt.cm.RdYlGn(perm_sorted['Perm_Importance_Pct'] / perm_sorted['Perm_Importance_Pct'].max())
ax.barh(perm_sorted['Feature'], perm_sorted['Perm_Importance_Pct'], color=colors)
ax.set_xlabel('Importance (%)', fontweight='bold')
ax.set_title('Permutation Importance (Test Set)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(perm_sorted['Perm_Importance_Pct']):
    ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)

# Plot 3: Correlation with Rating
ax = axes[1, 0]
corr_sorted = corr_df.sort_values('Pearson_Corr')
colors = ['#d7191c' if x < 0 else '#2b83ba' for x in corr_sorted['Pearson_Corr']]
ax.barh(corr_sorted['Feature'], corr_sorted['Pearson_Corr'], color=colors)
ax.set_xlabel('Pearson Correlation', fontweight='bold')
ax.set_title('Impact Direction (Positive/Negative)', fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(corr_sorted['Pearson_Corr']):
    label = f'{v:.4f}'
    x_pos = v + (0.005 if v > 0 else -0.005)
    ha = 'left' if v > 0 else 'right'
    ax.text(x_pos, i, label, va='center', ha=ha, fontweight='bold', fontsize=9)

# Plot 4: Model Performance
ax = axes[1, 1]
metrics = ['R² Score', 'MAE (stars)', 'RMSE (stars)']
train_vals = [train_r2, test_mae, test_rmse]
test_vals = [test_r2, test_mae, test_rmse]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, [train_r2, test_mae, test_rmse], width, label='Global Model', color='#2b83ba')
ax.set_ylabel('Score / Error', fontweight='bold')
ax.set_title('Model Performance Metrics', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.grid(axis='y', alpha=0.3)
for i, (m, v) in enumerate(zip(metrics, train_vals)):
    ax.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('logs/satisfaction_impact_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: logs/satisfaction_impact_analysis.png")

# ============================================================================
# VISUALIZATION 2: CATEGORY COMPARISON
# ============================================================================

if len(categories) > 1:
    fig, axes = plt.subplots(1, len(categories), figsize=(6*len(categories), 5))
    if len(categories) == 1:
        axes = [axes]
    
    fig.suptitle('Satisfaction Formula by Category', fontsize=16, fontweight='bold')
    
    for ax, category in enumerate(categories):
        cat_data = category_analysis[category]
        importance = cat_data['importance'].sort_values('Importance_Pct')
        colors = plt.cm.RdYlGn(importance['Importance_Pct'] / importance['Importance_Pct'].max())
        
        axes[ax].barh(importance['Feature'], importance['Importance_Pct'], color=colors)
        axes[ax].set_xlabel('Importance (%)', fontweight='bold')
        axes[ax].set_title(f"{category}\n(n={cat_data['count']:,}, μ={cat_data['baseline']:.2f}★)", fontweight='bold')
        axes[ax].grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(importance['Importance_Pct']):
            axes[ax].text(v + 0.3, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('logs/satisfaction_by_category.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: logs/satisfaction_by_category.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY & FORMULA")
print("="*80)

# Generate formula string
formula_parts = []
formula_parts.append(f"{baseline:.4f}")
for _, row in formula_df.sort_values('Final_Weight', ascending=False).iterrows():
    coef = row['Final_Weight'] * (1 if row['Direction'] == '+' else -1)
    formula_parts.append(f"{coef:+.4f}*{row['Feature']}")

formula_str = " + ".join([formula_parts[0]] + [p.replace("+", "\n  + ").replace("-", "\n  - ") for p in formula_parts[1:]])

print("\n📐 THE PHYSICS OF SATISFACTION - MATHEMATICAL FORMULA:")
print("-" * 80)
print("\nRating = " + formula_parts[0] + "\n" + "\n".join(formula_parts[1:]))

# Save formula to file
with open('logs/satisfaction_formula.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("CUSTOMER SATISFACTION FORMULA\n")
    f.write("="*80 + "\n\n")
    f.write("Rating = " + formula_parts[0] + "\n" + "\n".join(formula_parts[1:]) + "\n\n")
    f.write("\nWhere each factor is 0 or 1:\n")
    for factor in FACTORS:
        f.write(f"  {factor}: 0 = absence, 1 = presence\n")

print("\n✓ Formula saved to: logs/satisfaction_formula.txt")

# Save detailed results
formula_df.to_csv('logs/satisfaction_weights.csv', index=False)
print("✓ Weights saved to: logs/satisfaction_weights.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
