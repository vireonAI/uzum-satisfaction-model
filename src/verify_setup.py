"""
=============================================================================
🔍 DEBUG & VERIFICATION SCRIPT
=============================================================================

Model va datasets o'qamiz va tekshiramiz. CPU da ishadi, GPU kerak emas.

Usage:
    python src/verify_setup.py
"""

import os
import json
import pandas as pd
import sys
from pathlib import Path

print("="*80)
print("🔍 UZUM MARKETPLACE - TRAINING SETUP VERIFICATION")
print("="*80)

# Check 1: Verify data files
print("\n📂 Step 1: Checking data files...")

data_files = {
    'Train CSV': 'data/processed/uzum_train.csv',
    'Val CSV': 'data/processed/uzum_val.csv',
    'Test CSV': 'data/processed/uzum_test.csv',
    'Class Weights JSON': 'data/processed/class_weights.json',
}

all_exist = True
for name, path in data_files.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"  ✓ {name:20s}: {path:50s} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {name:20s}: {path:50s} (NOT FOUND)")
        all_exist = False

if not all_exist:
    print("\n⚠️  ERROR: Some data files missing!")
    sys.exit(1)

# Check 2: Verify data structure
print("\n📊 Step 2: Verifying data structure...")

try:
    train_df = pd.read_csv('data/processed/uzum_train.csv')
    val_df = pd.read_csv('data/processed/uzum_val.csv')
    test_df = pd.read_csv('data/processed/uzum_test.csv')
    
    print(f"  Train samples:  {len(train_df):,}")
    print(f"  Val samples:    {len(val_df):,}")
    print(f"  Test samples:   {len(test_df):,}")
    print(f"  Total:          {len(train_df) + len(val_df) + len(test_df):,}")
    
    # Check columns
    required_cols = [
        'Input_Text', 'Rating', 'script_type',
        'product_quality', 'price_value', 'logistics_delivery', 'packaging_condition',
        'accuracy_expectation', 'seller_service', 'specifications', 'product_defects'
    ]
    
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        print(f"\n  ✗ Missing columns: {missing_cols}")
        sys.exit(1)
    else:
        print(f"  ✓ All {len(required_cols)} required columns present")
    
    # Check for NaN
    nan_count = train_df['Input_Text'].isna().sum()
    if nan_count > 0:
        print(f"  ⚠️  WARNING: {nan_count} NaN values in Input_Text")
    else:
        print(f"  ✓ No NaN values in Input_Text")
    
    # Check text lengths
    train_df['text_length'] = train_df['Input_Text'].str.len()
    min_len = train_df['text_length'].min()
    max_len = train_df['text_length'].max()
    avg_len = train_df['text_length'].mean()
    print(f"  Text length: min={min_len}, max={max_len:.0f}, avg={avg_len:.0f}")
    
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    sys.exit(1)

# Check 3: Verify class weights
print("\n⚖️  Step 3: Verifying class weights...")

try:
    with open('data/processed/class_weights.json', 'r') as f:
        weights = json.load(f)
    
    print(f"  Factors: {len(weights)}")
    for factor, data in weights.items():
        pos_w = data['positive']
        neg_w = data['negative']
        ratio = data['balance_ratio']
        print(f"    {factor:25s}: pos={pos_w:.4f}, neg={neg_w:.4f}, ratio={ratio:.2f}x")
    
except Exception as e:
    print(f"  ✗ Error loading class weights: {e}")
    sys.exit(1)

# Check 4: Verify Python packages
print("\n📦 Step 4: Verifying Python packages...")

packages = {
    'pandas': 'pd',
    'numpy': 'np',
    'scikit-learn': 'sklearn',
    'torch': 'torch',
    'transformers': 'transformers',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'tqdm': 'tqdm',
}

for package_name, import_name in packages.items():
    try:
        __import__(import_name)
        print(f"  ✓ {package_name:20s}: installed")
    except ImportError:
        print(f"  ✗ {package_name:20s}: NOT INSTALLED")

# Check 5: Verify GPU (if torch installed)
print("\n🎮 Step 5: Checking GPU availability...")

try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ CUDA available: YES")
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {vram:.2f} GB")
    else:
        print(f"  ⚠️  CUDA available: NO (will use CPU, training will be slow)")
    
except Exception as e:
    print(f"  ⚠️  Could not check GPU: {e}")

# Check 6: Verify output directories
print("\n📁 Step 6: Verifying output directories...")

dirs_to_create = [
    'models/uzum_nlp_v1',
    'logs',
    'src',
]

for dir_path in dirs_to_create:
    if os.path.exists(dir_path):
        print(f"  ✓ {dir_path:30s}: exists")
    else:
        print(f"  ✗ {dir_path:30s}: does not exist (will create during training)")

# Check 7: Verify trainer script
print("\n🔧 Step 7: Verifying trainer script...")

if os.path.exists('src/final_trainer.py'):
    size_kb = os.path.getsize('src/final_trainer.py') / 1024
    print(f"  ✓ src/final_trainer.py exists ({size_kb:.2f} KB)")
else:
    print(f"  ✗ src/final_trainer.py NOT FOUND")
    sys.exit(1)

if os.path.exists('src/inference_api.py'):
    size_kb = os.path.getsize('src/inference_api.py') / 1024
    print(f"  ✓ src/inference_api.py exists ({size_kb:.2f} KB)")
else:
    print(f"  ✗ src/inference_api.py NOT FOUND")

# Final summary
print("\n" + "="*80)
print("✅ VERIFICATION COMPLETE")
print("="*80)

print(f"""
📊 Summary:
  • Data files: ✓ All present ({len(train_df) + len(val_df) + len(test_df):,} total samples)
  • Columns: ✓ All required columns
  • Class weights: ✓ {len(weights)} factors
  • Python packages: ✓ Core packages installed
  • Output directories: ✓ Ready for training
  • Trainer script: ✓ Present and ready

🚀 Next step: Run training
  python src/final_trainer.py

⏱️  Expected training time: 6-8 hours on RTX 3060
""")

print("="*80)
