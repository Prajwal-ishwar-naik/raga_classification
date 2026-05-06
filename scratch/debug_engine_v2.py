import sys
import os
import traceback
from pathlib import Path

# Setup paths
PROJECT_ROOT = os.getcwd()
sys.path.append(os.path.join(PROJECT_ROOT, "backend"))
sys.path.append(PROJECT_ROOT)

from neural_raga_engine import HybridRagaVision

def test_engine():
    print("Initializing HybridRagaVision...")
    try:
        engine = HybridRagaVision()
        test_file = "data/Yaman/Yaman_vocal_01.wav"
        
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            # Try to find any wav file
            for root, dirs, files in os.walk("data"):
                for f in files:
                    if f.endswith(".wav"):
                        test_file = os.path.join(root, f)
                        break
                if test_file: break
        
        print(f"\n[TEST] Running engine.analyze('{test_file}')...")
        result = engine.analyze(test_file, original_filename=os.path.basename(test_file))
        
        print("\n[SUCCESS] Result:")
        for k, v in result.items():
            if k == "spectrogram":
                print(f"  {k}: <list of length {len(v)}>")
            else:
                print(f"  {k}: {v}")
                
    except Exception as e:
        print("\n[FAILURE] Engine crashed!")
        traceback.print_exc()

if __name__ == "__main__":
    test_engine()
