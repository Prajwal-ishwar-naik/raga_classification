import sys
import os
import json

# Setup paths
sys.path.append(os.path.join(os.getcwd(), "backend"))

from utils.chunking import get_chunks
from advanced_features import extract_all_features
from utils.aggregation import aggregate_features
from core.classifier import classify_raga
from utils.pakad import detect_pakads


def test_full_pipeline():
    test_file = "data/Yaman/Yaman_vocal_01.wav"

    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found. Checking for any wav file...")
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith(".wav"):
                    test_file = os.path.join(root, file)
                    break
            if test_file:
                break

    if not test_file or not os.path.exists(test_file):
        print("No wav files found in data/ directory.")
        return

    print(f"=== FULL PIPELINE TEST ===")
    print(f"File: {test_file}")

    # -------------------------------
    # STEP 1: Chunking
    # -------------------------------
    print("\n[Step 1] Chunking audio...")
    chunks = get_chunks(test_file)

    if not chunks:
        print("Failed to extract chunks.")
        return

    test_chunks = chunks  # ✅ use all chunks
    print(f"Using {len(test_chunks)} chunks for classification.")

    # -------------------------------
    # STEP 2: Feature Extraction
    # -------------------------------
    print("\n[Step 2] Extracting features per chunk...")

    chunk_features_list = []
    full_swara_sequence = []

    for i, chunk in enumerate(test_chunks):
        print(f"  - Processing Chunk {i+1}...")

        res = extract_all_features(chunk, sr=22050)
        meta = res["metadata"]

        full_swara_sequence.extend(meta.get("swara_sequence", []))

        chunk_feat = {
            "swara_distribution": meta["swara_distribution"],
            "dominant_notes": [meta["most_frequent"]],
            "transitions": meta["transitions"],
            "pitch_range": meta["pitch_range"],
            "tempo": meta["tempo"]
        }

        chunk_features_list.append(chunk_feat)

    # -------------------------------
    # STEP 3: Aggregation
    # -------------------------------
    print("\n[Step 3] Aggregating features...")

    aggregated = aggregate_features(chunk_features_list)

    print(f"Detecting global pakads (Length: {len(full_swara_sequence)})...")
    aggregated["pakads"] = detect_pakads(full_swara_sequence, top_k=20)

    if aggregated["pakads"]:
        print(f"Top 5 Global Pakads: {aggregated['pakads'][:5]}")

    # -------------------------------
    # STEP 4: Classification
    # -------------------------------
    print("\n[Step 4] Classifying raga...")
    result = classify_raga(aggregated)

    # -------------------------------
    # FINAL OUTPUT
    # -------------------------------
    print("\n" + "=" * 50)
    print("        🎯 CLASSIFICATION RESULT")
    print("=" * 50)

    print(f"Prediction   : {result['prediction']}")
    print(f"Match Type   : {result.get('match_type', 'N/A')}")
    print(f"Confidence   : {result['confidence']:.4f}")

    if "note" in result:
        print(f"Note         : {result['note']}")

    # -------------------------------
    # EXPERT ANALYSIS (if exists)
    # -------------------------------
    print("\n[Expert Analysis]")

    analysis = result.get("analysis", {})

    if analysis.get("dominant_features"):
        print("Dominant Features:")
        for feat in analysis["dominant_features"]:
            print(f" - {feat}")

    if analysis.get("why_not_others"):
        print("\nRejection Logic:")
        for reason in analysis["why_not_others"]:
            print(f" - {reason}")

    # -------------------------------
    # ALTERNATIVES (SAFE PRINT)
    # -------------------------------
    if result.get("alternatives"):
        print("\n[Alternative Candidates]")
        for i, alt in enumerate(result["alternatives"], start=2):
            conf = alt.get("confidence", 0.0)  # ✅ SAFE
            print(f"{i}. {alt['raga']} → Confidence: {conf:.4f}")

    print("\n[Technical Summary]")
    print(f"Final Confidence: {result['confidence']:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    test_full_pipeline()