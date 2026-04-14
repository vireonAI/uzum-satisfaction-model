"""Quick test: verify v2 model loads and predicts via inference API"""
import sys
sys.path.insert(0, 'src')
from inference_api import UZUMInferenceAPI

api = UZUMInferenceAPI('models/uzum_nlp_v2')
result = api.predict('Sifat yaxshi, lekin qadoq yomon edi')

print("\nPredictions:")
for k, v in result['predictions'].items():
    print(f"  {k:25s}: pred={v['prediction']} conf={v['confidence']:.4f} t={v['threshold']}")
print(f"\nModel version: {result['model_version']}")
