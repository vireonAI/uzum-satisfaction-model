import os
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path

PROJECT = Path(r'C:\Users\doniy\Desktop\BMI_V4_NLP')
sys.path.insert(0, str(PROJECT))

from src.inference_api import UZUMInferenceAPI

api = UZUMInferenceAPI(str(PROJECT / 'models' / 'uzum_nlp_v3'))

print()
print('=== V3 SMOKE TEST ===')
tests = [
    ('Mahsulot juda yaxshi, sifat ajoyib! Tavsiya qilaman.', '5-star positive'),
    ('Tovar rasmda boshqa edi. Qaytardim. Sifatsiz material.', '1-star negative'),
    ('yaxshi', 'Single word - noise'),
    ('Tez yetkazibdi, lekin qadoq ezilgan edi. Narxi mos.', 'Packaging complaint'),
    ('Ne stoit svoikh deneg. Kachestvo ochen plokhoe, brak.', 'Russian defect review'),
]

for text, label in tests:
    r = api.predict(text)
    active = [f for f, p in r['predictions'].items() if p['prediction'] == 1]
    confs = {f: round(p['confidence'], 3) for f, p in r['predictions'].items() if p['prediction'] == 1}
    print(f'[{label}]')
    print(f'  Text   : {text[:70]}')
    print(f'  Factors: {active}')
    print(f'  Conf   : {confs}')
    print()

print('Calibrators disabled for v3:', api.calibrators is None)
print('Model version is v3:', api.is_v3)
print('Thresholds:', api.thresholds)
