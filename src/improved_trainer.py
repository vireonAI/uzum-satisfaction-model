"""
=============================================================================
🚀 IMPROVED TRAINER - XLM-RoBERTa Fine-Tuning with Focal Loss
=============================================================================

Fixes applied vs final_trainer.py:
  1. Focal Loss (γ=2.0) instead of Weighted BCE
  2. Class weights clamped to max 3.0 (was up to 11.3)
  3. max_length=128 (was 512 — reviews avg 12 tokens)
  4. batch_size=32 (was 8) → stabler gradients
  5. 15 epochs, patience=5 (was 7/2 → caused early stop at epoch 3)
  6. Cosine annealing scheduler
  7. Per-factor threshold optimization on val set
  8. Post-training Platt scaling calibration

Hardware: RTX 3060 6GB VRAM (sufficient for max_length=128, batch_size=32)
Estimated time: ~20-30 minutes
=============================================================================
"""

import os
import sys
import json
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
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

CONFIG = {
    'model_name': 'xlm-roberta-base',
    'max_length': 128,          # Was 512 — reviews avg 12 tokens, 99th pctile < 100
    'batch_size': 16,           # Was 8 — fits in 6GB VRAM
    'gradient_accumulation_steps': 4,  # Effective batch = 64
    'epochs': 15,               # Was 7 — give model time to learn
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,        # 10% warmup (was fixed 500 steps)
    'early_stopping_patience': 5,  # Was 2 — don't stop too early!  
    'fp16': torch.cuda.is_available(),
    'gradient_checkpointing': True,
    'seed': 42,
    'save_metric': 'macro_f1',
    # Focal Loss params
    'focal_gamma': 2.0,        # Focus on hard examples
    'max_pos_weight': 3.0,     # Clamp weights (was up to 11.3!)
    # Dropout
    'classifier_dropout': 0.3,
}

PATHS = {
    'train_csv': 'data/processed/uzum_train.csv',
    'val_csv': 'data/processed/uzum_val.csv',
    'test_csv': 'data/processed/uzum_test.csv',
    'class_weights_json': 'data/processed/class_weights.json',
    'model_save_dir': 'models/uzum_nlp_v2',  # Save as v2 to preserve v1
    'logs_dir': 'logs',
}


# ============================================================================
# DATASET
# ============================================================================

class UZUMDataset(Dataset):
    """Multi-lingual UZUM marketplace reviews dataset"""
    
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.factors = FACTORS
        
        missing = [f for f in FACTORS if f not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        print(f"  ✓ Loaded {len(self.df)} samples from {csv_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['Input_Text'])
        labels = torch.tensor([float(row[f]) for f in FACTORS], dtype=torch.float32)
        
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
        }


# ============================================================================
# MODEL ARCHITECTURE (same structure as original for weight compatibility)
# ============================================================================

class MultiLabelClassifier(nn.Module):
    """XLM-RoBERTa with Multi-Label Classification Head"""
    
    def __init__(self, model_name: str, num_factors: int = 8, dropout: float = 0.3):
        super().__init__()
        self.xlm = AutoModel.from_pretrained(model_name)
        hidden_size = self.xlm.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_factors)
        )
        
        print(f"  ✓ Model: {model_name} (hidden={hidden_size}, factors={num_factors})")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# ============================================================================
# FOCAL LOSS — key fix for class imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    FL(p) = -α * (1-p)^γ * log(p)    for positive samples
    FL(p) = -(1-α) * p^γ * log(1-p)  for negative samples
    
    γ (gamma) = focusing parameter — higher = more focus on hard examples
    α (alpha) = per-factor weight from class distribution
    """
    
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha.to(DEVICE)  # Shape: (num_factors,)
        self.gamma = gamma
        print(f"  ✓ Focal Loss: γ={gamma}, α range=[{alpha.min():.2f}, {alpha.max():.2f}]")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, num_factors) — raw, before sigmoid
        labels: (batch_size, num_factors) — binary 0/1
        """
        probs = torch.sigmoid(logits)
        
        # Clamp for numerical stability
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
        
        # Focal modulating factor
        # For positive: (1 - p)^γ — easy positives (high p) get low weight
        # For negative: p^γ — easy negatives (low p) get low weight
        pos_factor = (1.0 - probs) ** self.gamma
        neg_factor = probs ** self.gamma
        
        # BCE terms
        pos_loss = -labels * pos_factor * torch.log(probs)
        neg_loss = -(1.0 - labels) * neg_factor * torch.log(1.0 - probs)
        
        # Apply alpha weighting (per-factor)
        # alpha weight for positives, (1-alpha_normalized) for negatives
        # Since alpha here is pos_weight ratio, apply directly to positive term
        loss = self.alpha.unsqueeze(0) * pos_loss + neg_loss
        
        return loss.mean()


# ============================================================================
# TRAINER
# ============================================================================

class ImprovedTrainer:
    """Training pipeline with Focal Loss, calibration, and proper evaluation"""
    
    def __init__(self, config: Dict, paths: Dict):
        self.config = config
        self.paths = paths
        self.device = DEVICE
        
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.start_time = datetime.now()
        
        self.history = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'val_f1_macro': [], 'val_f1_micro': [], 'val_f1_weighted': [],
        }
        
        # Best thresholds (will be optimized on val set)
        self.best_thresholds = {f: 0.5 for f in FACTORS}
        
        # Create output directories
        os.makedirs(paths['model_save_dir'], exist_ok=True)
        os.makedirs(paths['logs_dir'], exist_ok=True)
        
        # Set seed
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
                
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                
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
    
    def _validate(self, model, val_loader, criterion) -> Tuple:
        """Validation with threshold-optimized metrics"""
        model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config['fp16']):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        
        # Use fixed 0.5 threshold for tracking (consistent metric)
        y_pred_fixed = (all_probs >= 0.5).astype(int)
        macro_f1_fixed = f1_score(all_labels, y_pred_fixed, average='macro', zero_division=0)
        
        # Also compute with optimized thresholds for reporting
        y_pred_opt = np.zeros_like(all_labels)
        for i, f in enumerate(FACTORS):
            t = self.best_thresholds.get(f, 0.5)
            y_pred_opt[:, i] = (all_probs[:, i] >= t).astype(int)
        macro_f1_opt = f1_score(all_labels, y_pred_opt, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, y_pred_opt, average='micro', zero_division=0)
        weighted_f1 = f1_score(all_labels, y_pred_opt, average='weighted', zero_division=0)
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, macro_f1_fixed, macro_f1_opt, micro_f1, weighted_f1, all_probs, all_labels
    
    def train(self):
        """Complete training pipeline"""
        print("\n" + "=" * 80)
        print("🚀 IMPROVED TRAINING PIPELINE")
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
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        model = MultiLabelClassifier(
            self.config['model_name'],
            dropout=self.config['classifier_dropout']
        ).to(self.device)
        
        if self.config['gradient_checkpointing']:
            model.xlm.gradient_checkpointing_enable()
            print("  ✓ Gradient checkpointing enabled")
        
        # Count params
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Parameters: {trainable_params:,} trainable / {total_params:,} total")
        
        # Datasets
        print("\n📥 Loading datasets...")
        train_dataset = UZUMDataset(self.paths['train_csv'], tokenizer, self.config['max_length'])
        val_dataset = UZUMDataset(self.paths['val_csv'], tokenizer, self.config['max_length'])
        test_dataset = UZUMDataset(self.paths['test_csv'], tokenizer, self.config['max_length'])
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'],
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'] * 2,
            shuffle=False, num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config['batch_size'] * 2,
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        # Loss — Focal Loss with clamped weights
        pos_weights = self.load_class_weights()
        criterion = FocalLoss(alpha=pos_weights, gamma=self.config['focal_gamma'])
        
        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Cosine scheduler with warmup
        total_steps = len(train_loader) * self.config['epochs'] // self.config['gradient_accumulation_steps']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"\n⚙️  Configuration:")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Batch size: {self.config['batch_size']} × {self.config['gradient_accumulation_steps']} accum = {self.config['batch_size'] * self.config['gradient_accumulation_steps']} effective")
        print(f"  Max length: {self.config['max_length']}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Patience: {self.config['early_stopping_patience']}")
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler(enabled=self.config['fp16'])
        
        # ============================================================
        # TRAINING LOOP
        # ============================================================
        print("\n" + "=" * 80)
        print("🎯 TRAINING")
        print("=" * 80)
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # --- Train ---
            model.train()
            total_train_loss = 0
            optimizer.zero_grad()
            
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for batch_idx, batch in enumerate(progress):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config['fp16']):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss = loss / self.config['gradient_accumulation_steps']
                
                scaler.scale(loss).backward()
                total_train_loss += loss.item() * self.config['gradient_accumulation_steps']
                
                if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                progress.set_postfix({'loss': f'{loss.item() * self.config["gradient_accumulation_steps"]:.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # --- Optimize thresholds every 3 epochs ---
            if (epoch + 1) % 3 == 0 or epoch == 0:
                self.best_thresholds = self._optimize_thresholds(model, val_loader)
            
            # --- Validate ---
            val_loss, f1_fixed, f1_opt, micro_f1, weighted_f1, _, _ = self._validate(
                model, val_loader, criterion
            )
            
            epoch_time = time.time() - epoch_start
            
            # Log
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_macro'].append(f1_opt)
            self.history['val_f1_micro'].append(micro_f1)
            self.history['val_f1_weighted'].append(weighted_f1)
            
            print(f"\n📊 Epoch {epoch+1}/{self.config['epochs']} ({epoch_time:.0f}s)")
            print(f"   Train Loss:     {avg_train_loss:.4f}")
            print(f"   Val Loss:       {val_loss:.4f}")
            print(f"   Val F1 (fixed): {f1_fixed:.4f}  |  Val F1 (opt): {f1_opt:.4f}")
            print(f"   Micro F1:       {micro_f1:.4f}  |  Weighted F1: {weighted_f1:.4f}")
            
            # Check for best model (use optimized F1)
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
        
        # Load best model
        best_model = self._load_best_model(tokenizer)
        
        # Optimize thresholds on val set with best model
        self.best_thresholds = self._optimize_thresholds(best_model, val_loader)
        print(f"\n  Optimized thresholds:")
        for f, t in self.best_thresholds.items():
            print(f"    {f:25s}: {t}")
        
        # Evaluate on test set
        _, _, test_f1, test_micro, test_weighted, test_probs, test_labels = self._validate(
            best_model, test_loader, criterion
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
        
        # ============================================================
        # CALIBRATION (Platt Scaling)
        # ============================================================
        print("\n" + "=" * 80)
        print("🔧 PROBABILITY CALIBRATION (Platt Scaling)")
        print("=" * 80)
        
        self._calibrate(best_model, val_loader, test_loader)
        
        # Save everything
        self._save_history()
        self._save_thresholds()
        
        total_time = (datetime.now() - self.start_time).total_seconds() / 60
        print(f"\n✅ Training complete in {total_time:.1f} minutes")
        print(f"   Best epoch: {self.best_epoch}")
        print(f"   Best macro F1: {self.best_metric:.4f}")
        print(f"   Model saved: {self.paths['model_save_dir']}")
    
    def _calibrate(self, model, val_loader, test_loader):
        """Apply Platt scaling (isotonic regression) per factor"""
        model.eval()
        
        # Collect val probabilities
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch['input_ids'].to(self.device),
                    batch['attention_mask'].to(self.device)
                )
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(batch['labels'].numpy())
        val_probs = np.vstack(val_probs)
        val_labels = np.vstack(val_labels)
        
        # Collect test probabilities
        test_probs, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(
                    batch['input_ids'].to(self.device),
                    batch['attention_mask'].to(self.device)
                )
                test_probs.append(torch.sigmoid(logits).cpu().numpy())
                test_labels.append(batch['labels'].numpy())
        test_probs = np.vstack(test_probs)
        test_labels = np.vstack(test_labels)
        
        # Fit per-factor Platt scaling on val, evaluate on test
        calibrators = {}
        print("\n  Per-factor calibration:")
        
        for i, factor in enumerate(FACTORS):
            # Fit Platt scaling (logistic regression on val probs → val labels)
            cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            cal.fit(val_probs[:, i:i+1], val_labels[:, i].astype(int))
            calibrators[factor] = cal
            
            # Apply to test
            cal_test_probs = cal.predict_proba(test_probs[:, i:i+1])[:, 1]
            
            # Compare before/after calibration
            raw_f1 = f1_score(test_labels[:, i], (test_probs[:, i] >= self.best_thresholds[factor]).astype(int), zero_division=0)
            cal_f1 = f1_score(test_labels[:, i], (cal_test_probs >= 0.5).astype(int), zero_division=0)
            delta = cal_f1 - raw_f1
            
            print(f"    {factor:25s}: raw F1={raw_f1:.4f} → calibrated F1={cal_f1:.4f}  ({'+' if delta >= 0 else ''}{delta:.4f})")
        
        # Save calibrators
        cal_path = os.path.join(self.paths['model_save_dir'], 'calibrators.pkl')
        with open(cal_path, 'wb') as f:
            pickle.dump(calibrators, f)
        print(f"\n  ✓ Calibrators saved to {cal_path}")
    
    def _save_model(self, model, tokenizer, epoch, metric_value):
        """Save model, tokenizer, and config"""
        save_dir = self.paths['model_save_dir']
        
        # Save model weights
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
        
        # Save tokenizer
        tokenizer.save_pretrained(save_dir)
        
        # Save config
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
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Save class weights
        with open(self.paths['class_weights_json'], 'r') as f:
            weights = json.load(f)
        with open(os.path.join(save_dir, 'class_weights.json'), 'w') as f:
            json.dump(weights, f, indent=2)
    
    def _load_best_model(self, tokenizer):
        """Load best model checkpoint"""
        model = MultiLabelClassifier(
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
        """Save training history"""
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.paths['logs_dir'], 'training_history_v2.csv')
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Training history saved to {csv_path}")
    
    def _save_thresholds(self):
        """Save optimized thresholds"""
        path = os.path.join(self.paths['model_save_dir'], 'thresholds.json')
        with open(path, 'w') as f:
            json.dump(self.best_thresholds, f, indent=2)
        print(f"  ✓ Thresholds saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 UZUM MARKETPLACE — IMPROVED MODEL TRAINING")
    print("=" * 80)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device: {DEVICE}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    trainer = ImprovedTrainer(CONFIG, PATHS)
    trainer.train()
