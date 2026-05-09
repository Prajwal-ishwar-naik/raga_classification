import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
from pathlib import Path

def plot_pitch_contour(features, out_path, audio_path=None):
    """
    Plots the pitch contour (F0) over time.
    """
    f0 = features.get("_f0")
    voiced = features.get("_voiced")
    sr = features.get("_sr", 22050)
    
    if (f0 is None or voiced is None) and audio_path:
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        f0, voiced, _ = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    
    if f0 is None:
        return None
        
    plt.figure(figsize=(10, 4))
    t = librosa.times_like(f0, sr=sr, hop_length=512)
    f0_plot = f0.copy().astype(float)
    if voiced is not None:
        f0_plot[~voiced | np.isnan(f0_plot)] = np.nan
    else:
        f0_plot[np.isnan(f0_plot)] = np.nan
    
    plt.plot(t, f0_plot, color="#4A90D9", lw=1.5, label="Pitch (F0)")
    
    if "tonic_hz" in features:
        plt.axhline(features["tonic_hz"], color="red", ls="--", lw=1.5, label=f"Tonic (Sa): {features['tonic_hz']:.1f} Hz")
        
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Contour Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path

def plot_spectrogram(y, sr, out_path):
    """
    Generates and saves a Mel Spectrogram heatmap.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram Visualization")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path

import matplotlib.patches as mpatches

SWARA_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa", "dha", "Dha", "ni", "Ni"]

def plot_full_dashboard(features, ranked, out_path, label):
    """
    Generates a 3-panel analysis dashboard:
    1. Pitch Contour
    2. Swara Prominence Histogram
    3. Top-5 Raga Matches
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"Raga Analysis - {label}", fontsize=18, fontweight="bold", y=0.98, color="#2c3e50")

    # 1. Pitch Contour
    ax1 = axes[0]
    f0 = features.get("_f0")
    voiced = features.get("_voiced")
    sr = features.get("_sr", 22050)
    
    if f0 is not None:
        t = librosa.times_like(f0, sr=sr, hop_length=512)
        f0_plot = f0.copy().astype(float)
        if voiced is not None:
            f0_plot[~voiced | np.isnan(f0_plot)] = np.nan
        else:
            f0_plot[np.isnan(f0_plot)] = np.nan
        ax1.plot(t, f0_plot, color="#4A90D9", lw=0.7, label="Pitch (F0)")
    
    if "tonic_hz" in features:
        ax1.axhline(features["tonic_hz"], color="red", ls="--", lw=1.5, label=f"Tonic (Sa): {features['tonic_hz']:.1f} Hz")
    
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("Pitch Contour Analysis", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Swara Prominence
    ax2 = axes[1]
    swara_dist = features.get("swara_distribution", {})
    pc = np.array([swara_dist.get(s, 0) for s in SWARA_NAMES])
    # For now, we use a simple color scheme as we don't have the full RAGA_DB in visualizer
    colors = ["#4A90D9"] * 12
    # Highlight vadi if known (usually the first note in detected_swaras)
    ax2.bar(SWARA_NAMES, pc * 100, color=colors, edgecolor="white", lw=0.5)
    ax2.axhline(3, color="gray", ls=":", lw=1, label="Active Threshold")
    ax2.set_ylabel("Relative Energy (%)")
    ax2.set_title("Swara Prominence Distribution", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # 3. Temporal Classification Matches
    ax3 = axes[2]
    if not ranked:
        ranked = [("Day", 0.5), ("Night", 0.5)]
    top2 = ranked[:2]
    names = [r[0] for r in top2]
    scrs = [r[1] for r in top2]
    bcols = ["#E87040" if n == features.get("prediction", "Day") else "#4A90D9" for n in names]
    ax3.barh(names[::-1], scrs[::-1], color=bcols[::-1], edgecolor="white")
    for i, (n, s) in enumerate(zip(names[::-1], scrs[::-1])):
        ax3.text(max(s, 0) + 0.005, i, f"{s:.3f}", va="center", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Confidence Score")
    ax3.set_title("Temporal Classification (Day vs Night)", fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
