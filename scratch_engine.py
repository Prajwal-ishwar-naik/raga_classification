import traceback
import sys
import os

# Add backend to path
sys.path.append(os.path.abspath('backend'))

from backend.neural_raga_engine import HybridRagaVision

try:
    engine = HybridRagaVision()
    engine.analyze('data/day_ragas/BhairavUP.opus', duration=10, original_filename='BhairavUP.opus', file_id='test')
    print("Success")
except Exception as e:
    traceback.print_exc()
