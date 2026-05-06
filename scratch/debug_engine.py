import sys
import os
from pathlib import Path

# Setup paths
sys.path.append(os.path.join(os.getcwd(), "backend"))

from neural_raga_engine import HybridRagaVision

def test_engine():
    try:
        engine = HybridRagaVision()
        test_file = "data/Yaman/Yaman_vocal_01.wav"
        if not os.path.exists(test_file):
            print("Test file not found.")
            return
        
        print("\n[TEST] Running engine.analyze()...")
        result = engine.analyze(test_file, original_filename="Yaman_vocal_01.wav")
        print("\n[SUCCESS] Result keys:", result.keys())
        print("Prediction:", result["prediction"])
    except Exception as e:
        print("\n[FAILURE] Engine crashed!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_engine()
