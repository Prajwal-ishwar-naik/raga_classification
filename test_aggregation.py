import sys
import os
import json

# Add backend to path for advanced_features
sys.path.append(os.path.join(os.getcwd(), "backend"))

from utils.chunking import get_chunks
from advanced_features import extract_all_features
from utils.aggregation import aggregate_features

def test_pipeline():
    test_file = "data/Yaman/Yaman_vocal_01.wav"
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    print(f"--- Processing File: {test_file} ---")
    
    # 1. Get chunks
    chunks = get_chunks(test_file)
    if not chunks:
        print("No chunks extracted.")
        return

    # Limit to first 3 chunks for faster test
    chunks = chunks[:3]
    print(f"Processing first {len(chunks)} chunks for testing aggregation...")

    chunk_features_list = []
    
    for i, chunk in enumerate(chunks):
        print(f"\nExtracting features for Chunk {i+1}...")
        # Use existing feature extraction
        # sr is 22050 as set in get_chunks
        res = extract_all_features(chunk, sr=22050)
        
        # Map to requested format
        meta = res["metadata"]
        chunk_feat = {
            "swara_distribution": meta["swara_distribution"],
            "dominant_notes": [meta["most_frequent"]],
            "transitions": meta["transitions"],
            "pitch_range": meta["pitch_range"],
            "tempo": meta["tempo"]
        }
        chunk_features_list.append(chunk_feat)

    # 2. Aggregate
    print("\n--- Aggregating Features ---")
    aggregated = aggregate_features(chunk_features_list)

    # 3. Print result
    print("\nFINAL AGGREGATED RESULT:")
    print(json.dumps(aggregated, indent=4))

if __name__ == "__main__":
    test_pipeline()
