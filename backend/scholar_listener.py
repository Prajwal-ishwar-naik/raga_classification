 # -*- coding: utf-8 -*-
"""
Indian Classical Raga Classification System (v3) - THE SCHOLAR-LISTENER
======================================================================
Neuro-Symbolic Architecture:
1.  Neural/Signal: librosa.pyin for high-precision F0 tracking.
2.  Frequency Calibration: Cent-based tuning & tonic locking (Hz-level).
3.  Symbolic Transcription: Clean F0 into a sequence of Note Events.
4.  Grammatical Reasoning:
    - Note-set/Constraint matching.
    - Pakad (Characteristic phrase) detection.
    - Bigram (Transition) analysis.
5.  Explainability: Logic-based reasoning report.

Author: Antigravity AI
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. EXPANDED RAGA KNOWLEDGE BASE (Symbolic Logic)
# ==============================================================================
# Notes: Sa=0, re=1(k), Re=2, ga=3(k), Ga=4, Ma=5, Ma'=6(t), Pa=7, dha=8(k), Dha=9, ni=10(k), Ni=11
# Notes: Sa=0, re=1(k), Re=2, ga=3(k), Ga=4, Ma=5, Ma'=6(t), Pa=7, dha=8(k), Dha=9, ni=10(k), Ni=11
RAGA_DB_V3 = {
    "Bhairav": {
        "notes": [0, 1, 4, 5, 7, 8, 11], "vadi": 8, "samvadi": 1,
        "pakads": [[4, 5, 8], [8, 7, 5, 4, 1, 0]], "forbidden": [2, 3, 6, 9, 10], "time": "Dawn"
    },
    "Bhairavi": {
        "notes": [0, 1, 3, 5, 7, 8, 10], "vadi": 5, "samvadi": 0,
        "pakads": [[0, 10, 8, 7], [3, 1, 0]], "forbidden": [2, 4, 6, 9, 11], "time": "Morning"
    },
    "Yaman": {
        "notes": [0, 2, 4, 6, 7, 9, 11], "vadi": 4, "samvadi": 11,
        "pakads": [[11, 2, 4], [4, 6, 7], [7, 6, 4, 2, 0]], "forbidden": [1, 3, 5, 8, 10], "time": "Evening"
    },
    "Kedar": {
        "notes": [0, 2, 5, 6, 7, 9], "vadi": 5, "samvadi": 9,
        "pakads": [[5, 7, 6, 5], [0, 5], [7, 9, 7, 5]], "forbidden": [1, 3, 4, 8, 10, 11], "time": "Night"
    },
    "Malkauns": {
        "notes": [0, 3, 5, 8, 10], "vadi": 5, "samvadi": 0,
        "pakads": [[0, 3, 5, 3, 0], [8, 10, 8, 5]], "forbidden": [1, 2, 4, 6, 7, 9, 11], "time": "Midnight"
    },
    "Bageshri": {
        "notes": [0, 2, 3, 5, 7, 9, 10], "vadi": 3, "samvadi": 10,
        "pakads": [[10, 9, 5, 3], [5, 3, 2, 0]], "forbidden": [1, 4, 6, 8, 11], "time": "Night"
    },
    "Todi": {
        "notes": [0, 1, 3, 6, 7, 8], "vadi": 8, "samvadi": 3,
        "pakads": [[8, 7, 6, 5], [3, 1, 0], [6, 8, 10, 12]], "forbidden": [2, 4, 5, 9, 11], "time": "Morning"
    },
    "Jaunpuri": {
        "notes": [0, 2, 3, 5, 7, 8, 10], "vadi": 8, "samvadi": 3,
        "pakads": [[7, 8, 10, 7], [5, 3, 2, 0]], "forbidden": [1, 4, 6, 9, 11], "time": "Morning"
    }
}

SWARA_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa", "dha", "Dha", "ni", "Ni"]

# ==============================================================================
# 2. ADVANCED ACOUSTIC PROCESSING (Neural / Signal Layer)
# ==============================================================================
def estimate_tonic_advanced(f0, voiced):
    """
    Standard acoustic estimate.
    """
    valid = voiced & ~np.isnan(f0)
    if valid.sum() < 50: return 220.0
    f0v = f0[valid]
    midi = librosa.hz_to_midi(f0v)
    midi_mod = midi % 12.0
    hist, edges = np.histogram(midi_mod, bins=120, range=(0, 12))
    peak_idx = hist.argmax()
    peak_pc = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0
    
    dists = np.abs(midi_mod - peak_pc)
    dists = np.minimum(dists, 12.0 - dists)
    return float(np.median(f0v[dists < 0.3]))

def refine_tonic_symbolic(f0, voiced, initial_tonic):
    """
    PRO-LEVEL MULTI-HYPOTHESIS TONIC LOCKING.
    Tests all 12 candidate tonics and picks the one that maximizes
    the structural integrity of the resulting melodic profile.
    """
    valid = voiced & ~np.isnan(f0)
    if not np.any(valid): return initial_tonic
    
    candidates = []
    # Try shifts from -6 to +5 semitones around the acoustic estimate
    for shift in range(-6, 6):
        test_tonic = initial_tonic * (2 ** (shift / 12.0))
        
        # Calculate histogram for this test tonic
        rel_semi = 12.0 * np.log2(f0[valid] / test_tonic)
        pc = np.mod(np.round(rel_semi), 12.0)
        hist, _ = np.histogram(pc, bins=12, range=(0, 12))
        hist = hist / (hist.sum() + 1e-6)
        
        # Calculate 'Raga Conformity Score'
        best_raga_score = -1e9
        for r_name in RAGA_DB_V3:
            s, _ = score_raga_logic(hist, [], r_name)
            if s > best_raga_score: best_raga_score = s
            
        candidates.append((test_tonic, best_raga_score))
            
    # Success: The tonic that allows the audio to 'make sense' musically
    best_tonic = max(candidates, key=lambda x: x[1])[0]
    return best_tonic

def transcribe_notes(f0, voiced, tonic_hz):
    """
    Clean the raw F0 into a sequence of Note Events (Symbolic Layer).
    Uses a median filter to remove flutter and then quantizes.
    """
    valid_mask = voiced & ~np.isnan(f0)
    if not np.any(valid_mask): return []
    
    # Convert to relative semitones
    rel_semi = 12.0 * np.log2(f0 / tonic_hz)
    
    # Apply median filtering to smooth meends/vibrato
    from scipy.signal import medfilt
    # Handle NaNs for filtering
    rel_semi_filled = np.nan_to_num(rel_semi, nan=-100)
    smoothed = medfilt(rel_semi_filled, kernel_size=15)
    
    sequence = []
    current_note = None
    count = 0
    
    for i, val in enumerate(smoothed):
        if not voiced[i]:
            if current_note is not None and count > 6: # Faster transition detection
                sequence.append(int(round(current_note)) % 12)
            current_note = None
            count = 0
            continue
            
        note = round(val)
        if current_note == note:
            count += 1
        else:
            if current_note is not None and count > 6:
                sequence.append(int(current_note) % 12)
            current_note = note
            count = 1
            
    # Distinct non-consecutive notes only for grammar
    final_seq = []
    for n in sequence:
        if not final_seq or n != final_seq[-1]:
            final_seq.append(n)
    return final_seq

# ==============================================================================
# 3. NEURO-SYMBOLIC REASONING (Logic Layer)
# ==============================================================================
def score_raga_logic(hist, sequence, raga_name):
    """
    ULTRA-PRECISION RAGA LOGIC:
    Combines spectral note-set matching, forbidden frequency penalties, 
    and pakad fingerprinting.
    """
    info = RAGA_DB_V3[raga_name]
    score = 0.0
    reasoning = []
    
    # A. Note Presence Score (Weighted by Vadi/Samvadi)
    notes = set(info["notes"])
    for n in notes:
        weight = 4.0 if n == info.get("vadi") else (2.5 if n == info.get("samvadi") else 1.5)
        score += hist[n] * weight
    
    # B. Elimination Logic (Forbidden Swaras)
    forbidden = info.get("forbidden", [])
    for n in forbidden:
        if hist[n] > 0.03: # High intolerance for chromatic leakage
            score -= hist[n] * 15.0 # Severe penalty
    
    # C. Pakad / Signature Phrase Matching
    if sequence:
        seq_str = "".join([chr(65+n) for n in sequence])
        for pakad in info.get("pakads", []):
            p_str = "".join([chr(65+n) for n in pakad])
            if p_str in seq_str:
                score += 3.0 # Strong match for signature phrases
                reasoning.append(f"(+) Captured Signature: {'-'.join([SWARA_NAMES[n] for n in pakad])}")
    
    return float(score), reasoning

# ==============================================================================
# 4. MAIN EXPLANABLE PIPELINE
# ==============================================================================
def analyze_recording(filepath):
    print(f"\n[ANALYZING] {Path(filepath).name}")
    y, sr = librosa.load(filepath, sr=22050) # Full duration
    
    # 1. Acoustic / Neural Signal Extraction
    f0, voiced, _ = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    
    # 2. Advanced Hz-refinement (Hybrid Approach)
    initial_tonic = estimate_tonic_advanced(f0, voiced)
    tonic = refine_tonic_symbolic(f0, voiced, initial_tonic)
    
    if abs(tonic - initial_tonic) > 1.0:
        print(f"    Tonic Calibration: {initial_tonic:.2f} Hz -> {tonic:.2f} Hz (Symbolic Correction)")
    else:
        print(f"    Tonic Locked: {tonic:.2f} Hz")
    
    # 3. Create Note Histogram & Sequence
    # Histogram represents 'Time Spent' (Duration stats)
    valid = voiced & ~np.isnan(f0)
    semitones = 12.0 * np.log2(f0[valid] / tonic)
    pc = np.mod(np.round(semitones), 12.0)
    hist, _ = np.histogram(pc, bins=12, range=(0, 12))
    hist = hist / (hist.sum() + 1e-6)
    
    # Sequence represents 'Grammar' (Transitions)
    sequence = transcribe_notes(f0, voiced, tonic)
    print(f"    Note Sequence: {' '.join([SWARA_NAMES[n] for n in sequence[:15]])}...")
    
    # 4. Symbolic Reasoning
    best_raga = None
    best_score = -999
    best_report = []
    
    all_scores = {}
    for r_name in RAGA_DB_V3:
        score, report = score_raga_logic(hist, sequence, r_name)
        all_scores[r_name] = score
        if score > best_score:
            best_score = score
            best_raga = r_name
            best_report = report
            
    return {
        "label": Path(filepath).stem,
        "tonic": tonic,
        "prediction": best_raga,
        "score": best_score,
        "report": best_report,
        "hist": hist,
        "all_scores": all_scores
    }

def main():
    BASE = Path(__file__).parent
    PROJECT_ROOT = BASE.parent          # raga_classification/
    DATA_DIR = PROJECT_ROOT / "data"
    OUT_DIR = PROJECT_ROOT / "output"
    OUT_DIR.mkdir(exist_ok=True)
    
    wav_files = []
    for d in ["day_ragas", "night_ragas"]:
        p = DATA_DIR / d
        if p.exists(): wav_files += sorted(p.glob("*.wav"))
    
    results = []
    correct = 0
    total = len(wav_files)
    
    print("\n" + "="*80)
    print("   INDIAN RAGA CLASSIFIER v3 (Scholar-Listener)")
    print("="*80)

    for i, f in enumerate(wav_files):
        res = analyze_recording(str(f))
        gt = f.stem.replace("_UP", "").replace("_", "")
        
        is_correct = (res["prediction"] == gt)
        if is_correct: correct += 1
        
        results.append(res)
        
        status = "OK" if is_correct else "FAIL"
        print(f"\n    [{status}] | File: {f.name} | Truth: {gt}")
        print(f"    Verdict: {res['prediction']} (Score: {res['score']:.2f})")
        print(f"    Reasoning:")
        for line in res["report"][:5]:
            print(f"      * {line}")

    # Summary
    acc = (correct/total)*100 if total > 0 else 0
    print("\n" + "="*80)
    print(f"   FINAL ACCURACY: {correct}/{total} ({acc:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()
