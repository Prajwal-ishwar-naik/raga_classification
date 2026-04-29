import sys
import os
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from neural_raga_engine import HybridRagaVision

def test_pipeline():
    engine = HybridRagaVision()
    
    # Mock a filepath (need a real one to avoid librosa error)
    # Let's find one in data/day_ragas
    data_dir = Path(__file__).parent.parent / "data" / "day_ragas"
    if not data_dir.exists():
        print("Data directory not found. Skipping full test.")
        return
        
    wav_files = []
    for ext in ['*.wav', '*.opus', '*.mp3']:
        wav_files.extend(list(data_dir.glob(ext)))
    
    if not wav_files:
        print("No audio files found in data/day_ragas.")
        return
        
    test_file = str(wav_files[0])
    print(f"Testing with: {test_file}")
    
    try:
        result = engine.analyze(test_file, duration=5, original_filename="test_audio.wav")
        print("\n--- ANALYSIS RESULT ---")
        print(f"Prediction: {result['prediction']}")
        print(f"Therapy Scores: {result['therapy_recommendation']['therapy_scores']}")
        print(f"Therapy Tags: {result['therapy_recommendation']['tags']}")
        print(f"Pitch URL: {result['pitch_contour_url']}")
        print(f"Spectrogram URL: {result['spectrogram_url']}")
        
        # Check if files exist
        base_dir = Path(__file__).parent.parent
        pitch_path = base_dir / "static" / "pitch_test_audio.png"
        spec_path = base_dir / "static" / "spectrogram_test_audio.png"
        
        print(f"Pitch file exists: {pitch_path.exists()}")
        print(f"Spec file exists: {spec_path.exists()}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
