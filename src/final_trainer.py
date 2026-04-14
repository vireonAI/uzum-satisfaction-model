"""
=============================================================================
🚀 UZUM MARKETPLACE - XLM-RoBERTa FINE-TUNING (MULTILINGUAL 8-FACTOR)
=============================================================================

Model: xlm-roberta-base
Task: Multi-Label Classification (8 independent binary factors)
Language: Russian + Uzbek (Multilingual)
Hardware: NVIDIA RTX 3060 (6GB VRAM) with Mixed Precision (fp16)

Key Features:
- BCEWithLogitsLoss with per-factor class weights
- Early stopping with macro_f1 metric
- Memory optimization: fp16, gradient checkpointing
- Stratified train/val/test with proper metrics
- Best model saving strategy

=============================================================================
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, hamming_loss, classification_report,
    confusion_matrix, accuracy_score
)

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup
)

# PyTorch
from torch.utils.data import Dataset, DataLoader
from torch import nn

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

FACTORS = [
    'product_quality', 'price_value', 'logistics_delivery', 'packaging_condition',
    'accuracy_expectation', 'seller_service', 'specifications', 'product_defects'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

CONFIG = {
    'model_name': 'xlm-roberta-base',
    'max_length': 512,
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
    'epochs': 7,
    'learning_rate': 2e-5,
    'warmup_steps': 500,
    'early_stopping_patience': 2,
    'fp16': True,
    'gradient_checkpointing': True,
    'seed': 42,
    'save_metric': 'macro_f1',  # Save best model by macro_f1
}

PATHS = {
    'train_csv': 'data/processed/uzum_train.csv',
    'val_csv': 'data/processed/uzum_val.csv',
    'test_csv': 'data/processed/uzum_test.csv',
    'class_weights_json': 'data/processed/class_weights.json',
    'model_save_dir': 'models/uzum_nlp_v1',
    'logs_dir': 'logs',
}

# ============================================================================
# DATASET CLASS
# ============================================================================

class UZUMDataset(Dataset):
    """Multi-lingual UZUM marketplace reviews dataset"""
    
    def __init__(self, csv_path: str, tokenizer, max_length: int = 512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate required columns
        required_cols = ['Input_Text'] + FACTORS
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        print(f"✓ Loaded {len(self.df)} samples from {csv_path}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['Input_Text'])
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels (multi-label: 0 or 1 for each factor)
        labels = torch.tensor(
            [float(row[factor]) for factor in FACTORS],
            dtype=torch.float32
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels,
            'text': text,
            'script_type': row.get('script_type', 'unknown'),
        }

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MultiLabelClassifier(nn.Module):
    """XLM-RoBERTa with Multi-Label Classification Head"""
    
    def __init__(self, model_name: str, num_factors: int = 8, dropout: float = 0.3):
        super().__init__()
        self.xlm = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Classification head: Dense -> Dropout -> Output
        hidden_size = self.xlm.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_factors)
        )
        
        print(f"✓ Model initialized: {model_name}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Output factors: {num_factors}")
    
    def forward(self, input_ids, attention_mask):
        # XLM-RoBERTa encoding
        outputs = self.xlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)
        
        # Classification
        logits = self.classifier(cls_output)
        return logits

# ============================================================================
# WEIGHTED BCE LOSS
# ============================================================================

class WeightedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with per-factor positive weights"""
    
    def __init__(self, pos_weights: torch.Tensor):
        """
        Args:
            pos_weights: Shape (num_factors,) - positive class weights
        """
        super().__init__()
        self.pos_weights = pos_weights.to(DEVICE)
    
    def forward(self, logits, labels):
        """
        logits: (batch_size, num_factors)
        labels: (batch_size, num_factors)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Weighted BCE loss
        loss = torch.zeros(labels.shape[1], device=DEVICE)
        
        for factor_idx in range(labels.shape[1]):
            pos_weight = self.pos_weights[factor_idx]
            factor_labels = labels[:, factor_idx]
            factor_logits = logits[:, factor_idx]
            
            # BCEWithLogitsLoss with positive weight
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(0))
            loss[factor_idx] = bce_loss(factor_logits.unsqueeze(1), factor_labels.unsqueeze(1))
        
        return loss.mean()

# ============================================================================
# TRAINER CLASS
# ============================================================================

class Trainer:
    """Complete training pipeline with early stopping and model saving"""
    
    def __init__(self, config: Dict, paths: Dict):
        self.config = config
        self.paths = paths
        self.device = DEVICE
        
        # Create output directories
        Path(paths['model_save_dir']).mkdir(parents=True, exist_ok=True)
        Path(paths['logs_dir']).mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_f1_macro': [],
            'val_hamming_loss': [],
        }
        
        self.best_metric = -np.inf if config['save_metric'] == 'macro_f1' else np.inf
        self.best_epoch = 0
        self.patience_counter = 0
        self.start_time = datetime.now()
        
        print("✓ Trainer initialized")
    
    def load_class_weights(self) -> torch.Tensor:
        """Load class weights from JSON"""
        with open(self.paths['class_weights_json'], 'r') as f:
            weights_dict = json.load(f)
        
        pos_weights = []
        for factor in FACTORS:
            weight = weights_dict[factor]['positive']
            pos_weights.append(weight)
        
        pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        print("✓ Class weights loaded:")
        for i, factor in enumerate(FACTORS):
            print(f"  {factor:25s}: {pos_weights[i]:.4f}")
        
        return pos_weights
    
    def train(self):
        """Complete training pipeline"""
        print("\n" + "="*80)
        print("🚀 STARTING TRAINING PIPELINE")
        print("="*80)
        
        # Memory optimization for 6GB VRAM
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Setup
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        model = MultiLabelClassifier(self.config['model_name']).to(self.device)
        
        if self.config['gradient_checkpointing']:
            model.xlm.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        
        # Load datasets
        print("\n📥 Loading datasets...")
        train_dataset = UZUMDataset(self.paths['train_csv'], tokenizer, self.config['max_length'])
        val_dataset = UZUMDataset(self.paths['val_csv'], tokenizer, self.config['max_length'])
        test_dataset = UZUMDataset(self.paths['test_csv'], tokenizer, self.config['max_length'])
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Loss & Optimizer
        pos_weights = self.load_class_weights()
        criterion = WeightedBCEWithLogitsLoss(pos_weights)
        
        # Optimizer with weight decay
        optimizer = AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        print(f"\n⚙️  Training Configuration:")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {self.config['warmup_steps']}")
        print(f"  Device: {self.device}")
        print(f"  Mixed Precision (fp16): {self.config['fp16']}")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Training loop
        print("\n" + "="*80)
        print("🎯 TRAINING STARTED")
        print("="*80)
        
        for epoch in range(self.config['epochs']):
            epoch_start = datetime.now()
            
            # Train phase
            model.train()
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer, scheduler)
            
            # Validation phase
            model.eval()
            val_loss, val_f1_macro, val_hamming, _, _ = self._validate(
                model, val_loader, criterion
            )
            
            # Log metrics
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_macro'].append(val_f1_macro)
            self.history['val_hamming_loss'].append(val_hamming)
            
            epoch_time = (datetime.now() - epoch_start).total_seconds() / 60
            
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} | Time: {epoch_time:.2f}m")
            print(f"  Train Loss:       {train_loss:.4f}")
            print(f"  Val Loss:         {val_loss:.4f}")
            print(f"  Val F1 (Macro):   {val_f1_macro:.4f}")
            print(f"  Val Hamming Loss: {val_hamming:.4f}")
            
            # Model saving (best by macro_f1)
            if val_f1_macro > self.best_metric:
                self.best_metric = val_f1_macro
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self._save_model(model, tokenizer, epoch + 1, val_f1_macro)
                print(f"  ⭐ BEST MODEL SAVED (F1: {val_f1_macro:.4f})")
            else:
                self.patience_counter += 1
                print(f"  ⏱️  Patience: {self.patience_counter}/{self.config['early_stopping_patience']}")
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\n⚠️  Early stopping triggered (patience={self.config['early_stopping_patience']})")
                break
        
        # Save training history
        self._save_history()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED")
        print("="*80)
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Metric (Macro F1): {self.best_metric:.4f}")
        total_time = (datetime.now() - self.start_time).total_seconds() / 3600
        print(f"Total Time: {total_time:.2f} hours")
        
        # Final evaluation on test set
        print("\n" + "="*80)
        print("🧪 FINAL EVALUATION ON TEST SET")
        print("="*80)
        
        model_best = self._load_best_model(tokenizer)
        model_best.eval()
        test_loss, test_f1, test_hamming, y_pred_test, y_true_test = self._validate(
            model_best, test_loader, criterion
        )
        
        print(f"\nTest Metrics:")
        print(f"  Test Loss:         {test_loss:.4f}")
        print(f"  Test F1 (Macro):   {test_f1:.4f}")
        print(f"  Test Hamming Loss: {test_hamming:.4f}")
        
        # Per-factor test metrics
        print(f"\nPer-Factor Test F1 Scores:")
        for i, factor in enumerate(FACTORS):
            f1 = f1_score(y_true_test[:, i], y_pred_test[:, i], zero_division=0)
            print(f"  {factor:25s}: {f1:.4f}")
        
        # Script-type comparison
        self._script_type_analysis(test_dataset.df.iloc[test_dataset.df.index], y_true_test, y_pred_test)
        
        # Generate reports
        self._generate_evaluation_report(y_true_test, y_pred_test)
        
        return model_best, tokenizer
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, scheduler):
        """Train for one epoch with gradient accumulation and mixed precision"""
        total_loss = 0
        optimizer.zero_grad()
        
        # GradScaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=self.config['fp16'])
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config['fp16']):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            total_loss += loss.item()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _validate(self, model, val_loader, criterion):
        """Validation loop"""
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Get predictions (threshold 0.5)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
        
        # Aggregate
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        val_loss = total_loss / len(val_loader)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_hamming = hamming_loss(all_labels, all_preds)
        
        return val_loss, val_f1_macro, val_hamming, all_preds, all_labels
    
    def _save_model(self, model, tokenizer, epoch: int, metric_value: float):
        """Save model, tokenizer, config, and weights"""
        save_path = Path(self.paths['model_save_dir'])
        
        # Save model weights
        torch.save(model.state_dict(), save_path / 'pytorch_model.bin')
        
        # Save tokenizer
        tokenizer.save_pretrained(save_path)
        
        # Save config
        config_to_save = {
            'model_name': self.config['model_name'],
            'num_factors': len(FACTORS),
            'factors': FACTORS,
            'max_length': self.config['max_length'],
            'epoch': epoch,
            'metric_value': metric_value,
            'save_metric': self.config['save_metric'],
            'timestamp': str(datetime.now()),
        }
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        # Copy class weights
        import shutil
        shutil.copy(self.paths['class_weights_json'], save_path / 'class_weights.json')
    
    def _load_best_model(self, tokenizer):
        """Load best model from checkpoint"""
        model = MultiLabelClassifier(self.config['model_name']).to(self.device)
        checkpoint = torch.load(Path(self.paths['model_save_dir']) / 'pytorch_model.bin', map_location=self.device)
        model.load_state_dict(checkpoint)
        return model
    
    def _save_history(self):
        """Save training history to CSV"""
        history_df = pd.DataFrame(self.history)
        history_path = Path(self.paths['logs_dir']) / 'training_history.csv'
        history_df.to_csv(history_path, index=False)
        print(f"✓ Training history saved to {history_path}")
    
    def _script_type_analysis(self, test_df, y_true, y_pred):
        """Compare Russian vs Uzbek performance"""
        print(f"\n📊 Script-Type Performance Analysis:")
        
        if 'script_type' in test_df.columns:
            scripts = test_df['script_type'].unique()
            for script in scripts:
                mask = test_df['script_type'] == script
                if mask.sum() > 0:
                    script_f1 = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
                    count = mask.sum()
                    print(f"  {script:15s}: F1={script_f1:.4f} (n={count})")
    
    def _generate_evaluation_report(self, y_true, y_pred):
        """Generate classification report and confusion matrices"""
        print(f"\n📈 Generating evaluation reports...")
        
        report_path = Path(self.paths['logs_dir']) / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLASSIFICATION REPORT - ALL FACTORS\n")
            f.write("="*80 + "\n\n")
            
            for i, factor in enumerate(FACTORS):
                f.write(f"\n{'='*80}\n")
                f.write(f"FACTOR: {factor.upper()}\n")
                f.write(f"{'='*80}\n")
                report = classification_report(
                    y_true[:, i],
                    y_pred[:, i],
                    target_names=['Negative', 'Positive'],
                    zero_division=0
                )
                f.write(report)
        
        print(f"✓ Classification report saved to {report_path}")
        
        # Confusion matrices visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Confusion Matrices - All 8 Factors', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, factor in enumerate(FACTORS):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_title(factor.replace('_', ' ').title())
            axes[i].set_ylabel('True')
            axes[i].set_xlabel('Predicted')
        
        fig.tight_layout()
        cm_path = Path(self.paths['logs_dir']) / 'confusion_matrices.png'
        fig.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrices saved to {cm_path}")
        
        # Training history plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], marker='o', label='Train Loss')
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], marker='s', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[0, 1].plot(self.history['epoch'], self.history['val_f1_macro'], marker='o', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Macro F1 Score')
        axes[0, 1].set_title('Validation F1 Score (Macro)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hamming Loss
        axes[1, 0].plot(self.history['epoch'], self.history['val_hamming_loss'], marker='o', color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Hamming Loss')
        axes[1, 0].set_title('Validation Hamming Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary
        axes[1, 1].axis('off')
        summary_text = f"""
        Training Summary
        {'─'*40}
        Best Epoch:        {self.best_epoch}
        Best Macro F1:     {self.best_metric:.4f}
        Final Val Loss:    {self.history['val_loss'][-1]:.4f}
        Final Hamming:     {self.history['val_hamming_loss'][-1]:.4f}
        Total Epochs:      {len(self.history['epoch'])}
        Device:            {self.device}
        Mixed Precision:   {self.config['fp16']}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        plot_path = Path(self.paths['logs_dir']) / 'training_history.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training history plot saved to {plot_path}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🎬 UZUM MARKETPLACE - XLM-RoBERTa FINE-TUNING")
    print("="*80)
    
    # Verify GPU
    print(f"\n💻 Device Info:")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set seed for reproducibility
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Create trainer and train
    trainer = Trainer(CONFIG, PATHS)
    model, tokenizer = trainer.train()
    
    print("\n" + "="*80)
    print("✨ ALL DONE! Check logs/ and models/uzum_nlp_v1/ for results")
    print("="*80)

if __name__ == '__main__':
    main()
