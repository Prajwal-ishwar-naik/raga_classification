# -*- coding: utf-8 -*-
"""
Indian Classical Raga Classification System  (v2)
==================================================
Neuro-Symbolic pipeline:
  1. Audio feature extraction  (librosa - pyin, MFCC, spectral)
  2. F0-based pitch-class histogram (NOT CQT chroma - avoids spectral leakage)
  3. Tonic (Sa) auto-detection
  4. Multi-signal raga scoring:
       a) Weighted note matching  (present & absent notes)
       b) Vadi / Samvadi prominence bonus
       c) Forbidden-note penalty  (vivarjit swaras)
  5. Therapeutic & mood recommendations

Dataset:  4 Day + 9 Night ragas  (13 WAV files)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# 1.  RAGA KNOWLEDGE BASE
# ==============================================================================
# Semitone offsets from Sa (tonic = 0):
#   Sa=0  re=1  Re=2  ga=3  Ga=4  Ma=5  Ma'=6  Pa=7  dha=8  Dha=9  ni=10  Ni=11
RAGA_DB = {
    "Bhairav": {
        "aroha":   [0, 1, 4, 5, 7, 8, 11, 12],
        "avaroha": [12, 11, 8, 7, 5, 4, 1, 0],
        "vadi": 8, "samvadi": 1,
        "time": "Early Morning (6-9 AM)", "season": "Hemant (Winter)",
        "mood": "Peaceful - Devotional - Meditative",
        "therapy": [
            "Reduces anxiety and promotes calm, reflective state of mind.",
            "Associated with spiritual contemplation and dawn prayers.",
            "Can help with insomnia when heard in morning hours.",
        ],
    },
    "AhirBhairav": {
        "aroha":   [0, 1, 4, 5, 7, 9, 10, 12],
        "avaroha": [12, 10, 9, 7, 5, 4, 1, 0],
        "vadi": 5, "samvadi": 9,
        "time": "Early Morning (6-10 AM)", "season": "Hemant / Shishir",
        "mood": "Devotional - Melancholic - Introspective",
        "therapy": [
            "Deepens devotional feelings; used in classical healing rituals.",
            "Blend of Bhairav (komal re) and Kafi (komal ni) creates emotional balance.",
            "Supports grief processing and emotional release.",
        ],
    },
    "AlhaiyaBilawal": {
        "aroha":   [0, 2, 4, 5, 7, 9, 11, 12],
        "avaroha": [12, 11, 9, 7, 5, 4, 2, 0],
        "vadi": 9, "samvadi": 4,
        "time": "Morning (6 AM - 12 PM)", "season": "Vasant (Spring)",
        "mood": "Cheerful - Bright - Energetic",
        "therapy": [
            "Elevates mood and promotes optimism.",
            "Major-scale structure resonates with alertness and productivity.",
            "Helpful for depression and low motivation.",
        ],
    },
    "ShuddhaSaarang": {
        "aroha":   [0, 2, 6, 7, 9, 12],
        "avaroha": [12, 9, 7, 6, 2, 0],
        "vadi": 2, "samvadi": 7,
        "time": "Afternoon (12-3 PM)", "season": "Grishma (Summer)",
        "mood": "Focused - Contemplative - Serene",
        "therapy": [
            "Supports deep focus and mental clarity.",
            "Beneficial for concentration difficulties and ADHD.",
            "Calming for the mind during peak daytime stress.",
        ],
    },
    "Yaman": {
        "aroha":   [0, 2, 4, 6, 7, 9, 11, 12],
        "avaroha": [12, 11, 9, 7, 6, 4, 2, 0],
        "vadi": 4, "samvadi": 11,
        "time": "Early Night (7-10 PM)", "season": "Varsha (Monsoon)",
        "mood": "Romantic - Longing - Uplifting",
        "therapy": [
            "Creates beauty and longing ideal for creative work.",
            "Tivra Ma stimulates the heart chakra.",
            "Helpful for emotional opening and relationship healing.",
        ],
    },
    "Bageshri": {
        "aroha":   [0, 2, 3, 5, 7, 9, 10, 12],
        "avaroha": [12, 10, 9, 7, 5, 3, 2, 0],
        "vadi": 3, "samvadi": 10,
        "time": "Late Night (12-3 AM)", "season": "Sarad (Autumn)",
        "mood": "Romantic - Longing - Nostalgic",
        "therapy": [
            "Deepens emotional sensitivity and nostalgia.",
            "Used therapeutically for mild depression and loneliness.",
            "Resonates with feelings of love and longing.",
        ],
    },
    "Malkauns": {
        "aroha":   [0, 3, 5, 8, 10, 12],
        "avaroha": [12, 10, 8, 5, 3, 0],
        "vadi": 8, "samvadi": 3,
        "time": "Midnight (12-4 AM)", "season": "Hemant",
        "mood": "Grave - Powerful - Mysterious",
        "therapy": [
            "Most powerful raga for deep meditation.",
            "Slows mind and reduces chronic stress patterns.",
            "Invokes courage and helps overcome fear.",
        ],
    },
    "Chandrakauns": {
        "aroha":   [0, 3, 5, 8, 11, 12],
        "avaroha": [12, 11, 8, 5, 3, 0],
        "vadi": 8, "samvadi": 3,
        "time": "Midnight (11 PM - 2 AM)", "season": "Shishir (Winter)",
        "mood": "Mysterious - Reflective - Tranquil",
        "therapy": [
            "Creates deeply still, moonlit atmosphere.",
            "Supports meditation and inner reflection.",
            "Helps with emotional numbness or dissociation.",
        ],
    },
    "Bhatiyar": {
        "aroha":   [0, 1, 4, 6, 7, 9, 11, 12],
        "avaroha": [12, 11, 9, 7, 6, 4, 1, 0],
        "vadi": 1, "samvadi": 9,
        "time": "Pre-Dawn (4-6 AM)", "season": "Hemant",
        "mood": "Devotional - Introspective - Solemn",
        "therapy": [
            "Awakens spiritual awareness in pre-dawn hours.",
            "Grounding effect suitable for existential anxiety.",
        ],
    },
    "Bihaag": {
        "aroha":   [0, 4, 5, 6, 7, 11, 12],
        "avaroha": [12, 11, 9, 7, 5, 4, 2, 0],
        "vadi": 4, "samvadi": 11,
        "time": "Late Night (9 PM - 12 AM)", "season": "All seasons",
        "mood": "Sweet - Tender - Romantic",
        "therapy": [
            "Promotes feelings of love and sweetness.",
            "Mild antidepressant effect in evening.",
            "Enhances creative and artistic sensitivity.",
        ],
    },
    "Desh": {
        "aroha":   [0, 2, 4, 5, 7, 9, 11, 12],
        "avaroha": [12, 10, 9, 7, 5, 4, 2, 0],
        "vadi": 9, "samvadi": 4,
        "time": "Night (9 PM - 12 AM)", "season": "Varsha (Monsoon)",
        "mood": "Devotional - Nostalgic - Longing",
        "therapy": [
            "Evokes feelings of homeland and belonging.",
            "Supports grief work and homesickness.",
            "Generally uplifting despite its emotional depth.",
        ],
    },
    "Kedar": {
        "aroha":   [0, 4, 5, 6, 7, 12],
        "avaroha": [12, 9, 7, 6, 5, 4, 2, 0],
        "vadi": 5, "samvadi": 9,
        "time": "Night (9 PM - 12 AM)", "season": "All seasons",
        "mood": "Devotional - Solemn - Meditative",
        "therapy": [
            "Strong devotional quality suitable for prayer.",
            "Calms mental chatter and promotes surrender.",
            "Helpful for sleep onset when heard at night.",
        ],
    },
    "Sohini": {
        "aroha":   [0, 1, 4, 6, 7, 8, 11, 12],
        "avaroha": [12, 11, 8, 7, 6, 4, 1, 0],
        "vadi": 4, "samvadi": 8,
        "time": "Late Night (12-3 AM)", "season": "Varsha",
        "mood": "Romantic - Intense - Longing",
        "therapy": [
            "Intensifies emotional sensitivity and longing.",
            "Associated with the navarasa of shringara (love).",
            "Helpful for expressing suppressed emotions.",
        ],
    },
}

SWARA_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa",
               "dha", "Dha", "ni", "Ni"]


# ==============================================================================
# 2.  F0-BASED PITCH-CLASS HISTOGRAM  (the key improvement over CQT chroma)
# ==============================================================================
def f0_to_pitch_histogram(f0, voiced, tonic_hz, n_bins=12):
    """
    Build a 12-bin pitch-class histogram from the raw F0 track,
    normalised so that the tonic (Sa) sits at index 0.

    Unlike CQT chroma, this only counts *actual pitched frames*
    and avoids harmonic leakage that makes every bin non-zero.
    """
    valid = voiced & ~np.isnan(f0)
    if valid.sum() < 50:
        return np.ones(12) / 12.0

    f0v = f0[valid]

    # Convert to semitones relative to tonic
    semitones = 12.0 * np.log2(f0v / tonic_hz)
    # Wrap into [0, 12)
    pc = np.mod(semitones, 12.0)

    hist, _ = np.histogram(pc, bins=n_bins, range=(0, 12))
    hist = hist.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# ==============================================================================
# 3.  TONIC ESTIMATION
# ==============================================================================
def estimate_tonic(f0, voiced):
    """
    Multi-octave tonic estimation:
      - Convert valid F0 to MIDI (continuous)
      - Build a fine-grained pitch-class histogram (mod 12)
      - Pick the dominant pitch class
      - Return the median Hz of frames near that pitch class
    """
    valid = voiced & ~np.isnan(f0)
    if valid.sum() < 50:
        return 220.0

    f0v       = f0[valid]
    midi      = librosa.hz_to_midi(f0v)
    midi_mod  = midi % 12.0

    # Fine histogram (120 bins = 10-cent resolution)
    hist, edges = np.histogram(midi_mod, bins=120, range=(0, 12))
    peak_idx    = hist.argmax()
    peak_pc     = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0

    # Gather frequencies near that pitch class
    dists  = np.abs(midi_mod - peak_pc)
    # Handle wraparound  (e.g. 11.9 vs 0.1)
    dists  = np.minimum(dists, 12.0 - dists)
    nearby = f0v[dists < 0.4]

    if len(nearby) > 10:
        return float(np.median(nearby))
    return float(np.median(f0v))


# ==============================================================================
# 4.  FEATURE EXTRACTION
# ==============================================================================
def extract_features(filepath, sr=22050):
    """Extract features from a WAV file; returns a dict."""
    print(f"\n  [LOAD] {Path(filepath).name}")
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    dur   = len(y) / sr
    print(f"         Duration: {dur:.1f}s | SR: {sr}")

    # Pitch
    f0, voiced, _ = librosa.pyin(
        y, sr=sr, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"), hop_length=512,
    )
    valid_f0 = f0[voiced & ~np.isnan(f0)]

    # Tonic
    tonic_hz = estimate_tonic(f0, voiced)
    print(f"         Tonic: {tonic_hz:.1f} Hz")

    # F0-based pitch histogram (12 bins, tonic-normalised)
    pc_hist = f0_to_pitch_histogram(f0, voiced, tonic_hz)

    # MFCC
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)

    # Spectral
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rolloff  = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    zcr_val  = float(librosa.feature.zero_crossing_rate(y).mean())
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return {
        "filepath":     filepath,
        "duration":     round(dur, 2),
        "tonic_hz":     round(tonic_hz, 2),
        "f0_median":    round(float(np.median(valid_f0)), 2) if len(valid_f0) else 0,
        "voiced_ratio": round(float(voiced.mean()), 3),
        "pc_hist":      pc_hist.tolist(),         # <-- key feature
        "mfcc":         mfcc_mean.tolist(),
        "spectral_centroid": round(centroid, 2),
        "spectral_rolloff":  round(rolloff, 2),
        "zcr":          round(zcr_val, 5),
        "tempo":        round(float(np.atleast_1d(tempo)[0]), 2),
        "_f0":          f0,
        "_voiced":      voiced,
        "_sr":          sr,
    }


# ==============================================================================
# 5.  RAGA CLASSIFIER  (multi-signal scoring)
# ==============================================================================
def _raga_note_set(info):
    """Set of pitch classes (0-11) used by a raga."""
    return {n % 12 for n in set(info["aroha"]) | set(info["avaroha"])}


def classify_raga(features):
    """
    Score each raga against the F0-derived pitch-class histogram.

    Scoring combines three signals:
      A) Note-set match     : do the active pitch classes align?
      B) Absent-note penalty: energy in *forbidden* pitch classes hurts the score
      C) Vadi/Samvadi boost : king & queen notes should be prominent

    Returns sorted list of (name, score).
    """
    h = np.array(features["pc_hist"])   # 12-bin normalised histogram

    # Determine "active" swaras (above a relative threshold)
    threshold = 0.03   # 3% of total duration
    active_mask = (h > threshold).astype(float)

    scores = {}
    for name, info in RAGA_DB.items():
        raga_notes  = _raga_note_set(info)
        raga_vec    = np.zeros(12)
        for n in raga_notes:
            raga_vec[n] = 1.0
        absent_vec = 1.0 - raga_vec  # notes that should NOT appear

        # A) Positive match: energy in raga notes
        match_energy = sum(h[n] for n in raga_notes)

        # B) Penalty: energy in forbidden notes
        forbidden_energy = float(np.dot(h, absent_vec))

        # C) Vadi / Samvadi prominence
        vadi    = info["vadi"] % 12
        samvadi = info["samvadi"] % 12
        vadi_score = h[vadi]
        samv_score = h[samvadi]

        # D) Count of "extra active" notes not in raga  (structural mismatch)
        extra_notes = 0
        for i in range(12):
            if active_mask[i] > 0 and raga_vec[i] == 0:
                extra_notes += 1

        # E) Count of "missing" raga notes  (should be present but aren't)
        missing_notes = 0
        for n in raga_notes:
            if h[n] < 0.01:  # essentially absent
                missing_notes += 1

        # Combined score
        score = (
            match_energy * 1.0
            - forbidden_energy * 1.5
            + vadi_score * 0.3
            + samv_score * 0.15
            - extra_notes * 0.06
            - missing_notes * 0.04
        )

        scores[name] = round(score, 4)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ==============================================================================
# 6.  VISUALISATION
# ==============================================================================
def plot_analysis(features, ranked, out_dir, label):
    info = RAGA_DB.get(ranked[0][0], {})
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"Raga Analysis - {label}", fontsize=15, fontweight="bold", y=0.98)

    # --- Panel 1: Pitch contour ---
    ax1 = axes[0]
    f0  = features["_f0"];  voi = features["_voiced"];  sr = features["_sr"]
    t   = librosa.times_like(f0, sr=sr, hop_length=512)
    f0p = f0.copy().astype(float)
    f0p[~voi | np.isnan(f0p)] = np.nan
    ax1.plot(t, f0p, color="#4A90D9", lw=0.7, label="F0 (pyin)")
    ax1.axhline(features["tonic_hz"], color="red", ls="--", lw=1.5,
                label=f"Sa ({features['tonic_hz']:.0f} Hz)")
    ax1.set_ylabel("Frequency (Hz)"); ax1.set_title("Pitch Contour")
    ax1.legend(fontsize=8);  ax1.grid(True, alpha=0.3)

    # --- Panel 2: F0-based pitch-class histogram ---
    ax2 = axes[1]
    pc  = np.array(features["pc_hist"])
    colors = ["#909090"] * 12
    if info:
        rn = _raga_note_set(info)
        for i in rn: colors[i] = "#E87040"
        colors[info.get("vadi", 0) % 12]    = "#B01010"
        colors[info.get("samvadi", 0) % 12] = "#CC3030"
    ax2.bar(SWARA_NAMES, pc * 100, color=colors, edgecolor="white", lw=0.5)
    ax2.axhline(3, color="gray", ls=":", lw=1, label="3% threshold")
    ax2.set_ylabel("% of Voiced Time")
    ax2.set_title("Swara Duration Distribution  (orange = predicted raga | red = vadi)")
    ax2.legend(handles=[
        mpatches.Patch(color="#909090", label="Non-raga notes"),
        mpatches.Patch(color="#E87040", label="Raga notes"),
        mpatches.Patch(color="#B01010", label="Vadi (king)"),
        mpatches.Patch(color="gray",   label="Active threshold"),
    ], fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # --- Panel 3: Top-5 scores ---
    ax3 = axes[2]
    top5   = ranked[:5]
    names  = [r[0] for r in top5]
    scrs   = [r[1] for r in top5]
    bcols  = ["#E87040" if i == 0 else "#4A90D9" for i in range(len(names))]
    ax3.barh(names[::-1], scrs[::-1], color=bcols[::-1], edgecolor="white")
    for i, (n, s) in enumerate(zip(names[::-1], scrs[::-1])):
        ax3.text(max(s, 0) + 0.005, i, f"{s:.3f}", va="center", fontsize=9)
    ax3.set_xlabel("Score");  ax3.set_title("Top-5 Raga Matches")
    ax3.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"analysis_{label.replace(' ','_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight");  plt.close()
    return out_path


# ==============================================================================
# 7.  REPORT
# ==============================================================================
def generate_report(label, features, ranked):
    top_name, top_score = ranked[0]
    info = RAGA_DB.get(top_name, {})
    pc   = np.array(features["pc_hist"])
    active = [SWARA_NAMES[i] for i in range(12) if pc[i] > 0.03]
    return {
        "file":       label,
        "duration_s": features["duration"],
        "tonic_hz":   features["tonic_hz"],
        "tempo_bpm":  features["tempo"],
        "voiced_pct": round(features["voiced_ratio"] * 100, 1),
        "detected_swaras": active,
        "classification": {
            "predicted_raga": top_name,
            "confidence":     round(top_score, 4),
            "alternatives":   [(n, round(s, 4)) for n, s in ranked[1:4]],
        },
        "raga_properties": {
            "time_of_day": info.get("time", "?"),
            "season":      info.get("season", "?"),
            "mood":        info.get("mood", "?"),
        },
        "therapeutic_insights": info.get("therapy", []),
    }


# ==============================================================================
# 8.  MAIN
# ==============================================================================
def main():
    BASE    = Path(__file__).parent
    OUT_DIR = BASE / "output"
    OUT_DIR.mkdir(exist_ok=True)

    wav_files = []
    for folder in ["Day Raga", "Night Raga"]:
        p = BASE / folder
        if p.exists():
            wav_files += sorted(p.glob("*.wav"))

    print("\n" + "=" * 70)
    print("  Indian Classical Raga Classification System  (v2)")
    print(f"  {len(wav_files)} recordings found")
    print("=" * 70)

    all_reports = []
    correct = 0

    for wav_path in wav_files:
        label = wav_path.stem
        # Ground truth: extract raga name from filename (before _UP or _)
        gt = label.replace("_UP", "").replace("_", "")

        print(f"\n{'-'*60}")
        print(f"  File : {label}")
        print(f"  True : {gt}")
        print(f"{'-'*60}")

        feats  = extract_features(str(wav_path))
        ranked = classify_raga(feats)
        pred, score = ranked[0]
        info = RAGA_DB.get(pred, {})

        is_correct = (pred == gt)
        if is_correct:
            correct += 1

        tag = "OK" if is_correct else "MISS"
        print(f"\n  [{tag}]  Predicted: {pred}  (score: {score:.4f})")
        print(f"  Top-3: " + ", ".join(f"{n}({s:.3f})" for n, s in ranked[:3]))

        # Detected swaras
        pc = np.array(feats["pc_hist"])
        active = [SWARA_NAMES[i] for i in range(12) if pc[i] > 0.03]
        print(f"  Active swaras: {', '.join(active)}")

        print(f"\n  [TIME]    {info.get('time', '-')}")
        print(f"  [SEASON]  {info.get('season', '-')}")
        print(f"  [MOOD]    {info.get('mood', '-')}")
        print(f"  [THERAPY]")
        for tip in info.get("therapy", []):
            print(f"     * {tip}")

        plot_path = plot_analysis(feats, ranked, str(OUT_DIR), label)
        print(f"\n  [PLOT] {plot_path}")

        all_reports.append(generate_report(label, feats, ranked))

    # Save report
    rp = OUT_DIR / "raga_classification_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    # Summary
    total = len(wav_files)
    print("\n\n" + "=" * 74)
    print("  CLASSIFICATION SUMMARY")
    print("=" * 74)
    print(f"  {'File':<26} {'True Raga':<18} {'Predicted':<18} {'Score':>7} {'':>5}")
    print("  " + "-" * 72)
    for r in all_reports:
        cls  = r["classification"]
        gt = r["file"].replace("_UP", "").replace("_", "")
        ok = "OK" if cls["predicted_raga"] == gt else "MISS"
        print(f"  {r['file']:<26} {gt:<18} {cls['predicted_raga']:<18} "
              f"{cls['confidence']:>7.4f} {ok:>5}")

    acc = 100 * correct / total if total else 0
    print(f"\n  Accuracy: {correct}/{total} ({acc:.1f}%)")
    print(f"\n[DONE] Report -> {rp}")
    print(f"[DONE] Plots  -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
