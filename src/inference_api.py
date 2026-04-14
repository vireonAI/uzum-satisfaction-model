"""
=============================================================================
🔮 UZUM MARKETPLACE - INFERENCE API
=============================================================================

Trained XLM-RoBERTa model bilan yangi sharhlarni test qilish uchun API.

Usage:
    from inference_api import UZUMInferenceAPI
    
    api = UZUMInferenceAPI('models/uzum_nlp_v1')
    
    review = "Mahsulot javob berdi, yetkazish tez va to'g'ri. Tavsif etgan narsaga mos."
    predictions = api.predict(review)
    api.print_results(predictions)

=============================================================================
"""

import os
import sys

# ── Block TensorFlow before ANY transformers import ───────────────────────────
# Newer transformers pulls in TF via image_transforms → broken protobuf.
os.environ.setdefault('USE_TF', 'NO')
os.environ.setdefault('USE_TORCH', 'YES')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# ──────────────────────────────────────────────────────────────────────────────

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
import logging

# Use specific classes — avoids AutoModel lazy-import chain that can pull in TF
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
from torch import nn

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

FACTORS = [
    'product_quality', 'price_value', 'logistics_delivery', 'packaging_condition',
    'accuracy_expectation', 'seller_service', 'specifications', 'product_defects'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL ARCHITECTURES (SAME AS TRAINER)
# ============================================================================

class MultiLabelClassifier(nn.Module):
    """XLM-RoBERTa with Multi-Label Classification Head (v1/v2)"""
    
    def __init__(self, model_name: str, num_factors: int = 8, dropout: float = 0.3):
        super().__init__()
        # Try loading from local cache first (avoids HuggingFace timeout)
        try:
            self.xlm = XLMRobertaModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.xlm = XLMRobertaModel.from_pretrained(model_name)
        
        hidden_size = self.xlm.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_factors)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


class MultiTaskClassifier(nn.Module):
    """XLM-RoBERTa with Factor Head + Rating Head (v3)"""
    
    def __init__(self, model_name: str, num_factors: int = 8, dropout: float = 0.3):
        super().__init__()
        try:
            self.xlm = XLMRobertaModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.xlm = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.xlm.config.hidden_size
        
        self.factor_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_factors)
        )
        self.rating_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        factor_logits = self.factor_head(cls_output)
        rating_pred = self.rating_head(cls_output).squeeze(-1)
        return factor_logits, rating_pred

# ============================================================================
# INFERENCE API
# ============================================================================

class UZUMInferenceAPI:
    """Inference API for XLM-RoBERTa trained model"""
    
    def __init__(self, model_dir: str, device: torch.device = DEVICE):
        """
        Initialize API with trained model
        
        Args:
            model_dir: Path to model directory (supports v1 and v2 formats)
            device: torch device (cuda or cpu)
        """
        self.model_dir = Path(model_dir)
        self.device = device
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        print("[inference_api] Loading model components...")
        
        # Load config
        config_path = self.model_dir / 'config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Detect model version
        self.is_v2 = (self.model_dir / 'model.pt').exists()
        self.is_v3 = self.config.get('version') == 'v3' or self.config.get('architecture') == 'multi_task'

        # Max length from config
        self.max_length = self.config.get('max_length', 128)  # default 128, not 512

        # Load tokenizer — use fast tokenizer directly to avoid TF import chain
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(str(self.model_dir))

        # Load model — v3 uses MultiTaskClassifier, v1/v2 use MultiLabelClassifier
        # model_name in config refers to 'xlm-roberta-base' hub ID — resolved from local HF cache
        model_name = self.config.get('model_name', 'xlm-roberta-base')
        dropout = self.config.get('classifier_dropout', 0.3)
        
        if self.is_v3:
            self.model = MultiTaskClassifier(
                model_name,
                num_factors=self.config['num_factors'],
                dropout=dropout
            ).to(self.device)
        else:
            self.model = MultiLabelClassifier(
                model_name,
                num_factors=self.config['num_factors'],
                dropout=dropout
            ).to(self.device)

        # Load weights
        weights_file = 'model.pt' if self.is_v2 else 'pytorch_model.bin'
        checkpoint = torch.load(
            self.model_dir / weights_file,
            map_location=self.device,
            weights_only=True  # safer, avoids pickle issues
        )
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Per-factor thresholds
        thresholds_path = self.model_dir / 'thresholds.json'
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                self.thresholds = json.load(f)
            print(f"  Thresholds loaded ({len(self.thresholds)} factors)")
        else:
            self.thresholds = {f: 0.5 for f in FACTORS}

        # Calibrators — disabled for V3: Platt scaling hurt F1 on 6/8 factors
        # (accuracy_expectation dropped from 0.38 → 0.00 after calibration).
        # For V2 we keep them as they were trained that way.
        self.calibrators = None
        if not self.is_v3:
            cal_path = self.model_dir / 'calibrators.pkl'
            if cal_path.exists():
                import pickle
                with open(cal_path, 'rb') as f:
                    self.calibrators = pickle.load(f)
                print(f"  Calibrators loaded (v2)")

        # Class weights (reference only)
        weights_path = self.model_dir / 'class_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.class_weights = json.load(f)
        else:
            self.class_weights = {f: {'positive': 1.0} for f in FACTORS}
        
        version_str = 'v3 (Multi-Task)' if self.is_v3 else ('v2 (Focal Loss)' if self.is_v2 else 'v1')
        print(f"[inference_api] Model loaded [{version_str}]")
        print(f"  Model     : {self.config.get('model_name', 'Unknown')}")
        print(f"  Max length: {self.max_length}")
        print(f"  Factors   : {self.config.get('num_factors', 0)}")
        print(f"  Device    : {self.device}")
        print(f"  Best epoch: {self.config.get('best_epoch', self.config.get('epoch', 'N/A'))}")
    
    def _detect_script_type(self, text: str) -> str:
        """Detect if text is Russian (Cyrillic), Uzbek (Latin), or Mixed"""
        cyrillic_count = sum(1 for c in text if ord(c) >= 0x0400 and ord(c) <= 0x04FF)
        latin_count = sum(1 for c in text if c.isalpha() and ord(c) < 0x0400)
        
        total_alpha = cyrillic_count + latin_count
        if total_alpha == 0:
            return 'unknown'
        
        cyrillic_ratio = cyrillic_count / total_alpha
        
        if cyrillic_ratio > 0.7:
            return 'cyrillic'
        elif cyrillic_ratio < 0.3:
            return 'latin'
        else:
            return 'mixed'
    
    def predict(self, text: str, confidence_threshold: float = None, temperature: float = 1.0) -> Dict:
        """
        Predict labels for a review text
        
        Args:
            text: Review text (Russian or Uzbek)
            confidence_threshold: Override threshold (if None, uses per-factor thresholds)
            temperature: Sigmoid temperature (default 1.0 = no scaling)
        
        Returns:
            Dictionary with predictions, confidences, and metadata
        """
        # Tokenize with max_length from config
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            # v3 returns tuple (factor_logits, rating_pred), v1/v2 return just logits
            if isinstance(output, tuple):
                logits = output[0]
                # rating_pred = output[1] # Not used in current output format
            else:
                logits = output
            logits_np = logits.cpu().numpy()[0]
            logger.debug(f"Logits: Min={logits_np.min():.4f}, Max={logits_np.max():.4f}, Mean={logits_np.mean():.4f}")
            
            # Apply temperature scaling (1.0 = no change)
            if temperature != 1.0:
                scaled_logits = logits / temperature
            else:
                scaled_logits = logits
            probs = torch.sigmoid(scaled_logits).cpu().numpy()[0]
        
        # Parse predictions with per-factor thresholds
        predictions = {}
        for i, factor in enumerate(FACTORS):
            prob = float(probs[i])
            # Use per-factor threshold if available, otherwise fall back to confidence_threshold or 0.5
            threshold = confidence_threshold if confidence_threshold is not None else self.thresholds.get(factor, 0.5)
            predicted_label = 1 if prob >= threshold else 0
            
            predictions[factor] = {
                'prediction': predicted_label,
                'confidence': prob,
                'raw_logit': float(logits_np[i]),
                'threshold': threshold,
                'class_weight': self.class_weights[factor]['positive'],
            }
        
        # Metadata
        script_type = self._detect_script_type(text)
        
        return {
            'text': text,
            'text_length': len(text),
            'script_type': script_type,
            'model_version': 'v2' if self.is_v2 else 'v1',
            'predictions': predictions,
            'timestamp': str(datetime.now().isoformat()),
        }
    
    def predict_batch(self, texts: List[str], confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Predict labels for multiple texts
        
        Args:
            texts: List of review texts
            confidence_threshold: Threshold for positive classification
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict(text, confidence_threshold)
            results.append(result)
        
        return results
    
    def print_results(self, prediction: Dict, show_confidence: bool = True):
        """Pretty-print prediction results"""
        print("\n" + "="*80)
        print("📊 PREDICTION RESULTS")
        print("="*80)
        
        print(f"\nText: {prediction['text'][:100]}...")
        print(f"Length: {prediction['text_length']} characters")
        print(f"Script: {prediction['script_type'].upper()}")
        print(f"Threshold: {prediction['confidence_threshold']}")
        
        print(f"\n{'Factor':<30} {'Prediction':<15} {'Confidence':<15}")
        print("─" * 60)
        
        for factor, pred_info in prediction['predictions'].items():
            pred_label = "✅ Positive" if pred_info['prediction'] == 1 else "❌ Negative"
            conf = pred_info['confidence']
            
            if show_confidence:
                print(f"{factor:<30} {pred_label:<15} {conf:.4f}")
            else:
                print(f"{factor:<30} {pred_label:<15}")
        
        # Summary
        positive_count = sum(
            1 for p in prediction['predictions'].values()
            if p['prediction'] == 1
        )
        print(f"\n📈 Summary: {positive_count}/{len(FACTORS)} positive predictions")
        print("="*80)
    
    def export_predictions(self, prediction: Dict, output_path: str):
        """Export predictions to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prediction, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Predictions exported to {output_path}")

# ============================================================================
# INTERACTIVE DEMO
# ============================================================================

def interactive_demo():
    """Interactive prediction demo"""
    print("\n" + "="*80)
    print("🔮 UZUM MARKETPLACE - INTERACTIVE PREDICTION DEMO")
    print("="*80)
    
    api = UZUMInferenceAPI('models/uzum_nlp_v1')
    
    print("\n💡 Enter Uzbek or Russian reviews. Type 'exit' to quit.\n")
    
    while True:
        text = input("📝 Enter review: ").strip()
        
        if text.lower() == 'exit':
            print("Goodbye!")
            break
        
        if len(text) < 3:
            print("⚠️  Review too short. Minimum 3 characters.\n")
            continue
        
        prediction = api.predict(text, confidence_threshold=0.5)
        api.print_results(prediction, show_confidence=True)

# ============================================================================
# BATCH DEMO
# ============================================================================

def batch_demo():
    """Batch prediction demo"""
    api = UZUMInferenceAPI('models/uzum_nlp_v1')
    
    test_reviews = [
        "Mahsulot javob berdi, yetkazish tez, narxi mos. Tafsiya qilaman.",
        "Kacheli tovar edi, ketka chiqdi. Pul qaytarish so'roddim.",
        "Ruscha review: Товар хорошего качества, доставка быстрая. Рекомендую!",
        "Плохой товар, долго ждал доставку. Не доволен.",
    ]
    
    print("\n" + "="*80)
    print("📦 BATCH PREDICTION DEMO")
    print("="*80)
    
    results = api.predict_batch(test_reviews)
    
    for i, result in enumerate(results):
        print(f"\n--- Review {i+1} ---")
        api.print_results(result)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        batch_demo()
    else:
        interactive_demo()
