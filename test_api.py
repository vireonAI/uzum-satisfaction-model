"""Quick test: verify backend analyze endpoint uses v2 model"""
import urllib.request
import json

# Test analyze endpoint
data = json.dumps({"text": "Mahsulot sifati ajoyib, narxi ham yaxshi"}).encode()
req = urllib.request.Request(
    'http://localhost:8000/api/analyze',
    data=data,
    headers={'Content-Type': 'application/json'}
)
resp = json.loads(urllib.request.urlopen(req).read())
print("=== Analyze Endpoint Test ===")
print(f"Text: {resp['text']}")
print(f"Script: {resp['script_type']}")
print(f"Overall: {resp['overall_sentiment']}")
print("\nFactors:")
for f in resp['factors']:
    print(f"  {f['factor']:25s}: pred={f['prediction']} conf={f['confidence']:.4f} label={f['label']}")

# Test model-performance endpoint
resp2 = json.loads(urllib.request.urlopen('http://localhost:8000/api/model-performance').read())
print(f"\n=== Model Performance ===")
for m in resp2['models']:
    print(f"  {m['model']:35s}: Macro F1 = {m['overall']['macro_f1']}")

# Test health endpoint
resp3 = json.loads(urllib.request.urlopen('http://localhost:8000/api/health').read())
print(f"\n=== Health ===")
print(f"  Model loaded: {resp3.get('model_loaded', 'N/A')}")
print(f"  ML available: {resp3.get('ml_available', 'N/A')}")
