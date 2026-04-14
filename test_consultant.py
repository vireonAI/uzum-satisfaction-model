"""Test consultant endpoint with None values (matching real product analysis output)"""
import urllib.request
import json
import time

time.sleep(3)  # Wait for server reload

# Simulate real analysis with None values (matching what the cached product returns)
test_data = {
    'analysis': {
        'status': 'success',
        'product_info': {'title': 'Test Product', 'review_count': 100, 'actual_rating': 4.2, 'category': 'Elektronika'},
        'health_analysis': {
            'health_score': 7.5,
            'predicted_rating': 4.1,
            'factor_breakdown': {'product_quality': 0.25, 'product_defects': 0.15},
            'top_problems': None  # This was causing the crash!
        },
        'benchmark': None,  # This too
        'timestamp': '2026-02-20'
    },
    'language': 'uz'
}

req = urllib.request.Request(
    'http://localhost:8000/api/consultant',
    data=json.dumps(test_data).encode(),
    headers={'Content-Type': 'application/json'}
)

resp = urllib.request.urlopen(req)
d = json.loads(resp.read())
print(f"Status: {d.get('status')}")
print(f"Verdict: {d.get('overall_verdict', 'N/A')[:120]}")
if d.get('status') == 'error':
    print(f"Error: {d.get('message')}")
