"""
=============================================================================
🚀 V3 TRAINER - Multi-Task Learning + Weighted Oversampling
=============================================================================

Improvements over improved_trainer.py (v2):
  1. Multi-Task Learning: Second head predicts star rating (1-5)
     - Combined loss: FocalLoss(factors) + λ × MSE(rating)
     - Forces model to understand sentiment direction
  2. Weighted Oversampling for Rare Factors
     - Samples with rare factors (packaging, accuracy, specs) weighted 3-5× more
     - Uses WeightedRandomSampler instead of shuffle=True
  3. Label smoothing on factor labels (0→0.05, 1→0.95)

Architecture:
  XLM-RoBERTa Encoder (768)
      ├── Factor Head:  Linear(768→256) → ReLU → Dropout → Linear(256→8)
      └── Rating Head:  Linear(768→128) → ReLU → Dropout → Linear(128→1)

Hardware: RTX 3060 6GB VRAM
Estimated time: ~30-40 minutes
=============================================================================
"""

import os
import sys
import json
import warnings
import time

# ── Block TensorFlow before ANY transformers import ───────────────────────────
# Newer transformers versions import image_transforms → tensorflow → broken
# protobuf. Setting these env vars prevents TF from initialising at all.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('USE_TF', 'NO')
os.environ.setdefault('USE_TORCH', 'YES')
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# Use specific classes — avoids the AutoModel lazy-import chain that pulls in TF
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaModel,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

FACTORS = [
    'product_quality', 'price_value', 'logistics_delivery', 'packaging_condition',
    'accuracy_expectation', 'seller_service', 'specifications', 'product_defects'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# XLM-RoBERTa base is already in the local HuggingFace hub cache (~/.cache/huggingface/hub)
# from when v2 was trained. We use the standard model ID here and set
# TRANSFORMERS_OFFLINE in the launch command so it never hits the network.
CONFIG = {
    'model_name': 'xlm-roberta-base',  # resolved from local HF cache
    'max_length': 128,              # 99.2% of reviews fit in 128 tokens
    'batch_size': 8,                # RTX 3060 6GB — 8x8=64 effective batch
    'gradient_accumulation_steps': 8,
    'epochs': 15,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'early_stopping_patience': 5,
    'fp16': torch.cuda.is_available(),
    'gradient_checkpointing': True,
    'seed': 42,
    'save_metric': 'macro_f1',
    # Focal Loss
    'focal_gamma': 2.0,
    'max_pos_weight': 3.0,
    # Multi-task
    'rating_loss_weight': 0.3,      # lambda for rating MSE loss
    'label_smoothing': 0.05,        # Smooth labels: 0->0.05, 1->0.95
    # Dropout
    'classifier_dropout': 0.3,
}

PATHS = {
    'train_csv': 'data/processed/uzum_train.csv',
    'val_csv': 'data/processed/uzum_val.csv',
    'test_csv': 'data/processed/uzum_test.csv',
    'class_weights_json': 'data/processed/class_weights.json',
    'model_save_dir': 'models/uzum_nlp_v3',
    'logs_dir': 'logs',
}


# ============================================================================
# DATASET (with Rating)
# ============================================================================

class UZUMDatasetV3(Dataset):
    """Dataset that returns both factor labels AND star rating"""
    
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        missing = [f for f in FACTORS if f not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if 'Rating' not in self.df.columns:
            raise ValueError("Missing 'Rating' column — needed for multi-task learning")
        
        print(f"  ✓ Loaded {len(self.df)} samples from {csv_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['Input_Text'])
        
        # Factor labels (multi-label binary)
        labels = torch.tensor([float(row[f]) for f in FACTORS], dtype=torch.float32)
        
        # Star rating normalized to [0, 1] for regression
        rating = torch.tensor((float(row['Rating']) - 1.0) / 4.0, dtype=torch.float32)
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels,
            'rating': rating,
        }


# ============================================================================
# MODEL V3 — Multi-Task: Factors + Rating
# ============================================================================

class MultiTaskClassifier(nn.Module):
    """
    XLM-RoBERTa with two prediction heads:
      1. Factor Head: 8-factor multi-label classification
      2. Rating Head: Star rating regression (1-5)
    """
    
    def __init__(self, model_name: str, num_factors: int = 8, dropout: float = 0.3):
        super().__init__()
        self.xlm = XLMRobertaModel.from_pretrained(model_name, local_files_only=False)
        hidden_size = self.xlm.config.hidden_size
        
        # Factor classification head (same architecture as v2)
        self.factor_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_factors)
        )
        
        # Rating regression head (NEW)
        self.rating_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1], scaled back to [1, 5]
        )
        
        print(f"  ✓ Multi-Task Model: {model_name}")
        print(f"    Factor Head: {hidden_size} → 256 → {num_factors}")
        print(f"    Rating Head: {hidden_size} → 128 → 1")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        factor_logits = self.factor_head(cls_output)
        rating_pred = self.rating_head(cls_output).squeeze(-1)
        
        return factor_logits, rating_pred


# ============================================================================
# FOCAL LOSS (same as v2)
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss with label smoothing support"""
    
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha.to(DEVICE)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        print(f"  ✓ Focal Loss: γ={gamma}, α range=[{alpha.min():.2f}, {alpha.max():.2f}], smooth={label_smoothing}")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
        
        pos_factor = (1.0 - probs) ** self.gamma
        neg_factor = probs ** self.gamma
        
        pos_loss = -labels * pos_factor * torch.log(probs)
        neg_loss = -(1.0 - labels) * neg_factor * torch.log(1.0 - probs)
        
        loss = self.alpha.unsqueeze(0) * pos_loss + neg_loss
        return loss.mean()


# ============================================================================
# WEIGHTED SAMPLER — Oversample rare factors
# ============================================================================

def compute_sample_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute per-sample weights that oversample reviews mentioning rare factors.
    
    Rare factor presence → higher weight → sampled more often.
    """
    factor_rates = {f: df[f].mean() for f in FACTORS}
    
    # Inverse frequency weighting: rarer factor → higher weight
    factor_weights = {}
    for f, rate in factor_rates.items():
        if rate < 0.10:       # Very rare (specs=8%, packaging=8.8%, accuracy=8.9%)
            factor_weights[f] = 4.0
        elif rate < 0.20:     # Rare (logistics=14.7%, defects=14.9%, seller=18.8%)
            factor_weights[f] = 2.0
        elif rate < 0.30:     # Moderate (price=27.6%)
            factor_weights[f] = 1.5
        else:                 # Common (quality=68.9%)
            factor_weights[f] = 1.0
    
    # Each sample's weight = max weight across its positive factors
    weights = []
    for _, row in df.iterrows():
        sample_weight = 1.0
        for f in FACTORS:
            if row[f] == 1:
                sample_weight = max(sample_weight, factor_weights[f])
        weights.append(sample_weight)
    
    weights = torch.tensor(weights, dtype=torch.float64)
    
    # Print stats
    print(f"\n  📊 Sample weights computed:")
    for f in FACTORS:
        print(f"    {f:25s}: rate={factor_rates[f]:.3f} → weight={factor_weights[f]:.1f}×")
    print(f"    Total samples: {len(weights)}")
    print(f"    Weighted samples: {int((weights > 1.0).sum())} ({(weights > 1.0).float().mean()*100:.1f}%)")
    
    return weights


# ============================================================================
# TRAINER V3
# ============================================================================

class TrainerV3:
    """Multi-task training with weighted oversampling"""
    
    def __init__(self, config: Dict, paths: Dict):
        self.config = config
        self.paths = paths
        self.device = DEVICE
        
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.start_time = datetime.now()
        
        self.history = {
            'epoch': [], 'train_loss': [], 'train_factor_loss': [], 'train_rating_loss': [],
            'val_loss': [], 'val_f1_macro': [], 'val_f1_micro': [], 'val_f1_weighted': [],
            'val_rating_mae': [],
        }
        
        self.best_thresholds = {f: 0.5 for f in FACTORS}
        
        os.makedirs(paths['model_save_dir'], exist_ok=True)
        os.makedirs(paths['logs_dir'], exist_ok=True)
        
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
    
    def load_class_weights(self) -> torch.Tensor:
        """Load and CLAMP class weights"""
        with open(self.paths['class_weights_json'], 'r') as f:
            weights_dict = json.load(f)
        
        max_w = self.config['max_pos_weight']
        pos_weights = []
        
        print(f"\n  Class weights (clamped to max {max_w}):")
        for factor in FACTORS:
            raw = weights_dict[factor]['positive']
            clamped = min(raw, max_w)
            pos_weights.append(clamped)
            flag = " ⚠️ CLAMPED" if raw > max_w else ""
            print(f"    {factor:25s}: {raw:.2f} → {clamped:.2f}{flag}")
        
        return torch.tensor(pos_weights, dtype=torch.float32)
    
    def _optimize_thresholds(self, model, val_loader) -> Dict[str, float]:
        """Find optimal per-factor thresholds on validation set"""
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                factor_logits, _ = model(input_ids, attention_mask)
                probs = torch.sigmoid(factor_logits).cpu().numpy()
                
                all_probs.append(probs)
                all_labels.append(labels)
        
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        
        thresholds = {}
        for i, factor in enumerate(FACTORS):
            best_f1, best_t = 0, 0.5
            for t in np.arange(0.15, 0.85, 0.01):
                preds = (all_probs[:, i] >= t).astype(int)
                f1 = f1_score(all_labels[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = round(float(t), 2)
            thresholds[factor] = best_t
        
        return thresholds
    
    def _validate(self, model, val_loader, factor_criterion) -> Tuple:
        """Validation with both factor and rating evaluation"""
        model.eval()
        total_loss = 0
        total_factor_loss = 0
        total_rating_loss = 0
        all_probs = []
        all_labels = []
        all_rating_preds = []
        all_rating_true = []
        
        rating_criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config['fp16']):
                    factor_logits, rating_pred = model(input_ids, attention_mask)
                    f_loss = factor_criterion(factor_logits, labels)
                    r_loss = rating_criterion(rating_pred, ratings)
                    loss = f_loss + self.config['rating_loss_weight'] * r_loss
                
                total_loss += loss.item()
                total_factor_loss += f_loss.item()
                total_rating_loss += r_loss.item()
                
                probs = torch.sigmoid(factor_logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
                
                # Convert rating pred back to 1-5 scale
                all_rating_preds.extend((rating_pred.cpu().numpy() * 4.0 + 1.0).tolist())
                all_rating_true.extend((ratings.cpu().numpy() * 4.0 + 1.0).tolist())
        
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        
        # Factor metrics
        y_pred_fixed = (all_probs >= 0.5).astype(int)
        macro_f1_fixed = f1_score(all_labels, y_pred_fixed, average='macro', zero_division=0)
        
        y_pred_opt = np.zeros_like(all_labels)
        for i, f in enumerate(FACTORS):
            t = self.best_thresholds.get(f, 0.5)
            y_pred_opt[:, i] = (all_probs[:, i] >= t).astype(int)
        macro_f1_opt = f1_score(all_labels, y_pred_opt, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, y_pred_opt, average='micro', zero_division=0)
        weighted_f1 = f1_score(all_labels, y_pred_opt, average='weighted', zero_division=0)
        
        # Rating metrics
        rating_mae = np.mean(np.abs(np.array(all_rating_preds) - np.array(all_rating_true)))
        
        avg_loss = total_loss / len(val_loader)
        avg_f_loss = total_factor_loss / len(val_loader)
        avg_r_loss = total_rating_loss / len(val_loader)
        
        return (avg_loss, avg_f_loss, avg_r_loss, macro_f1_fixed, macro_f1_opt, 
                micro_f1, weighted_f1, rating_mae, all_probs, all_labels)
    
    def train(self):
        """Complete multi-task training pipeline"""
        print("\n" + "=" * 80)
        print("🚀 V3 MULTI-TASK TRAINING PIPELINE")
        print("=" * 80)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gpu_name = torch.cuda.get_device_properties(0).name
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("  ⚠️  No GPU detected — training on CPU (will be slow)")
        
        # Setup
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(self.config['model_name'])
        model = MultiTaskClassifier(
            self.config['model_name'],
            dropout=self.config['classifier_dropout']
        ).to(self.device)
        
        if self.config['gradient_checkpointing']:
            model.xlm.gradient_checkpointing_enable()
            print("  ✓ Gradient checkpointing enabled")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Parameters: {trainable_params:,} trainable / {total_params:,} total")
        
        # Datasets
        print("\n📥 Loading datasets...")
        train_dataset = UZUMDatasetV3(self.paths['train_csv'], tokenizer, self.config['max_length'])
        val_dataset = UZUMDatasetV3(self.paths['val_csv'], tokenizer, self.config['max_length'])
        test_dataset = UZUMDatasetV3(self.paths['test_csv'], tokenizer, self.config['max_length'])
        
        # Weighted oversampling for training
        sample_weights = compute_sample_weights(train_dataset.df)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True  # Required for WeightedRandomSampler
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'],
            sampler=sampler,  # Uses sampler instead of shuffle=True
            num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'] * 2,
            shuffle=False, num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config['batch_size'] * 2,
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        # Loss functions
        pos_weights = self.load_class_weights()
        factor_criterion = FocalLoss(
            alpha=pos_weights, 
            gamma=self.config['focal_gamma'],
            label_smoothing=self.config['label_smoothing']
        )
        rating_criterion = nn.MSELoss()
        
        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Cosine scheduler
        total_steps = len(train_loader) * self.config['epochs'] // self.config['gradient_accumulation_steps']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"\n⚙️  Configuration:")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Batch: {self.config['batch_size']} × {self.config['gradient_accumulation_steps']} = {self.config['batch_size'] * self.config['gradient_accumulation_steps']} effective")
        print(f"  Max length: {self.config['max_length']}")
        print(f"  Rating loss weight (λ): {self.config['rating_loss_weight']}")
        print(f"  Label smoothing: {self.config['label_smoothing']}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup: {warmup_steps}")
        
        scaler = torch.cuda.amp.GradScaler(enabled=self.config['fp16'])
        
        # ============================================================
        # TRAINING LOOP
        # ============================================================
        print("\n" + "=" * 80)
        print("🎯 TRAINING")
        print("=" * 80)
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0
            total_train_f_loss = 0
            total_train_r_loss = 0
            optimizer.zero_grad()
            
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for batch_idx, batch in enumerate(progress):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config['fp16']):
                    factor_logits, rating_pred = model(input_ids, attention_mask)
                    
                    # Multi-task loss
                    f_loss = factor_criterion(factor_logits, labels)
                    r_loss = rating_criterion(rating_pred, ratings)
                    loss = (f_loss + self.config['rating_loss_weight'] * r_loss) / self.config['gradient_accumulation_steps']
                
                scaler.scale(loss).backward()
                total_train_loss += loss.item() * self.config['gradient_accumulation_steps']
                total_train_f_loss += f_loss.item()
                total_train_r_loss += r_loss.item()
                
                if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                progress.set_postfix({
                    'loss': f'{loss.item() * self.config["gradient_accumulation_steps"]:.4f}',
                    'f_loss': f'{f_loss.item():.4f}',
                    'r_loss': f'{r_loss.item():.4f}'
                })
            
            n_batches = len(train_loader)
            avg_train_loss = total_train_loss / n_batches
            avg_train_f_loss = total_train_f_loss / n_batches
            avg_train_r_loss = total_train_r_loss / n_batches
            
            # Optimize thresholds
            if (epoch + 1) % 3 == 0 or epoch == 0:
                self.best_thresholds = self._optimize_thresholds(model, val_loader)
            
            # Validate
            (val_loss, val_f_loss, val_r_loss, f1_fixed, f1_opt, 
             micro_f1, weighted_f1, rating_mae, _, _) = self._validate(
                model, val_loader, factor_criterion
            )
            
            epoch_time = time.time() - epoch_start
            
            # Log
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_factor_loss'].append(avg_train_f_loss)
            self.history['train_rating_loss'].append(avg_train_r_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_macro'].append(f1_opt)
            self.history['val_f1_micro'].append(micro_f1)
            self.history['val_f1_weighted'].append(weighted_f1)
            self.history['val_rating_mae'].append(rating_mae)
            
            print(f"\n📊 Epoch {epoch+1}/{self.config['epochs']} ({epoch_time:.0f}s)")
            print(f"   Train: loss={avg_train_loss:.4f} (factor={avg_train_f_loss:.4f}, rating={avg_train_r_loss:.4f})")
            print(f"   Val:   loss={val_loss:.4f} (factor={val_f_loss:.4f}, rating={val_r_loss:.4f})")
            print(f"   F1: fixed={f1_fixed:.4f} | opt={f1_opt:.4f} | micro={micro_f1:.4f} | weighted={weighted_f1:.4f}")
            print(f"   Rating MAE: {rating_mae:.3f} stars")
            
            # Best model tracking
            if f1_opt > self.best_metric:
                self.best_metric = f1_opt
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self._save_model(model, tokenizer, epoch + 1, f1_opt)
                print(f"   ⭐ BEST MODEL SAVED (F1: {f1_opt:.4f})")
            else:
                self.patience_counter += 1
                print(f"   ⏱️  Patience: {self.patience_counter}/{self.config['early_stopping_patience']}")
            
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        # ============================================================
        # FINAL EVALUATION
        # ============================================================
        print("\n" + "=" * 80)
        print("🧪 FINAL EVALUATION ON TEST SET")
        print("=" * 80)
        
        best_model = self._load_best_model(tokenizer)
        self.best_thresholds = self._optimize_thresholds(best_model, val_loader)
        
        print(f"\n  Optimized thresholds:")
        for f, t in self.best_thresholds.items():
            print(f"    {f:25s}: {t}")
        
        (_, _, _, _, test_f1, test_micro, test_weighted, 
         test_rating_mae, test_probs, test_labels) = self._validate(
            best_model, test_loader, factor_criterion
        )
        
        # Per-factor results
        print(f"\n  Per-factor results (test set):")
        y_pred = np.zeros_like(test_labels)
        for i, f in enumerate(FACTORS):
            y_pred[:, i] = (test_probs[:, i] >= self.best_thresholds[f]).astype(int)
            f1 = f1_score(test_labels[:, i], y_pred[:, i], zero_division=0)
            prec = precision_score(test_labels[:, i], y_pred[:, i], zero_division=0)
            rec = recall_score(test_labels[:, i], y_pred[:, i], zero_division=0)
            print(f"    {f:25s}: F1={f1:.4f} | P={prec:.4f} | R={rec:.4f} | t={self.best_thresholds[f]}")
        
        print(f"\n  Overall:")
        print(f"    Macro F1:    {test_f1:.4f}")
        print(f"    Micro F1:    {test_micro:.4f}")
        print(f"    Weighted F1: {test_weighted:.4f}")
        print(f"    Rating MAE:  {test_rating_mae:.3f} stars")
        
        # ============================================================
        # CALIBRATION
        # ============================================================
        print("\n" + "=" * 80)
        print("🔧 PROBABILITY CALIBRATION")
        print("=" * 80)
        
        self._calibrate(best_model, val_loader, test_loader, factor_criterion)
        
        # Save artifacts
        self._save_history()
        self._save_thresholds()
        
        total_time = (datetime.now() - self.start_time).total_seconds() / 60
        print(f"\n✅ V3 Training complete in {total_time:.1f} minutes")
        print(f"   Best epoch: {self.best_epoch}")
        print(f"   Best macro F1: {self.best_metric:.4f}")
        print(f"   Model saved: {self.paths['model_save_dir']}")
    
    def _calibrate(self, model, val_loader, test_loader, factor_criterion):
        """Platt scaling per factor"""
        model.eval()
        
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                factor_logits, _ = model(
                    batch['input_ids'].to(self.device),
                    batch['attention_mask'].to(self.device)
                )
                val_probs.append(torch.sigmoid(factor_logits).cpu().numpy())
                val_labels.append(batch['labels'].numpy())
        val_probs = np.vstack(val_probs)
        val_labels = np.vstack(val_labels)
        
        test_probs, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                factor_logits, _ = model(
                    batch['input_ids'].to(self.device),
                    batch['attention_mask'].to(self.device)
                )
                test_probs.append(torch.sigmoid(factor_logits).cpu().numpy())
                test_labels.append(batch['labels'].numpy())
        test_probs = np.vstack(test_probs)
        test_labels = np.vstack(test_labels)
        
        calibrators = {}
        print("\n  Per-factor calibration:")
        for i, factor in enumerate(FACTORS):
            cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            cal.fit(val_probs[:, i:i+1], val_labels[:, i].astype(int))
            calibrators[factor] = cal
            
            cal_test_probs = cal.predict_proba(test_probs[:, i:i+1])[:, 1]
            raw_f1 = f1_score(test_labels[:, i], (test_probs[:, i] >= self.best_thresholds[factor]).astype(int), zero_division=0)
            cal_f1 = f1_score(test_labels[:, i], (cal_test_probs >= 0.5).astype(int), zero_division=0)
            delta = cal_f1 - raw_f1
            print(f"    {factor:25s}: raw F1={raw_f1:.4f} → cal F1={cal_f1:.4f}  ({'+' if delta >= 0 else ''}{delta:.4f})")
        
        cal_path = os.path.join(self.paths['model_save_dir'], 'calibrators.pkl')
        with open(cal_path, 'wb') as f:
            pickle.dump(calibrators, f)
        print(f"\n  ✓ Calibrators saved to {cal_path}")
    
    def _save_model(self, model, tokenizer, epoch, metric_value):
        save_dir = self.paths['model_save_dir']
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
        tokenizer.save_pretrained(save_dir)
        
        config_data = {
            'model_name': self.config['model_name'],
            'num_factors': len(FACTORS),
            'factors': FACTORS,
            'max_length': self.config['max_length'],
            'classifier_dropout': self.config['classifier_dropout'],
            'best_epoch': epoch,
            'best_metric': metric_value,
            'thresholds': self.best_thresholds,
            'training_config': self.config,
            'version': 'v3',
            'architecture': 'multi_task',
            'improvements': [
                'multi_task_rating_head',
                'weighted_oversampling',
                'label_smoothing',
            ]
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        
        with open(self.paths['class_weights_json'], 'r') as f:
            weights = json.load(f)
        with open(os.path.join(save_dir, 'class_weights.json'), 'w') as f:
            json.dump(weights, f, indent=2)
    
    def _load_best_model(self, tokenizer):
        model = MultiTaskClassifier(
            self.config['model_name'],
            dropout=self.config['classifier_dropout']
        ).to(self.device)
        
        state_dict = torch.load(
            os.path.join(self.paths['model_save_dir'], 'model.pt'),
            map_location=self.device
        )
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  ✓ Loaded best model from epoch {self.best_epoch}")
        return model
    
    def _save_history(self):
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.paths['logs_dir'], 'training_history_v3.csv')
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Training history saved to {csv_path}")
    
    def _save_thresholds(self):
        path = os.path.join(self.paths['model_save_dir'], 'thresholds.json')
        with open(path, 'w') as f:
            json.dump(self.best_thresholds, f, indent=2)
        print(f"  ✓ Thresholds saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 UZUM MARKETPLACE — V3 MULTI-TASK MODEL TRAINING")
    print("=" * 80)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device: {DEVICE}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    trainer = TrainerV3(CONFIG, PATHS)
    trainer.train()
