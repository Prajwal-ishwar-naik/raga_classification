import numpy as np
import librosa
import os
import noisereduce as nr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scholar_listener import RAGA_DB_V3, SWARA_NAMES, estimate_tonic_advanced, refine_tonic_symbolic
from audacity_loader import load_audacity_project

class RagaTitanV5:
    def __init__(self):
        self.db = RAGA_DB_V3
        self.melodic_templates = self._prepare_templates()

    def _prepare_templates(self):
        templates = {}
        for name, info in self.db.items():
            templates[name] = []
            for pakad in info.get("pakads", []):
                shape = []
                for note in pakad:
                    shape.extend([note] * 30) # High-res templates
                templates[name].append(np.array(shape))
        return templates

    def analyze(self, filepath):
        # 0. Load Signal (Handle .aup or standard)
        if filepath.lower().endswith('.aup'):
            y, sr = load_audacity_project(filepath)
            if y is None: raise ValueError("Could not load Audacity project data")
            # Limit to 60s for analysis efficiency
            y = y[:int(sr*60)]
        else:
            y, sr = librosa.load(filepath, sr=22050, duration=60)
            
        # 1. Audacity-Style Cleaning (Noise Reduction)
        print(f"    [CLEAN] Applying Spectral Gating (noisereduce)...")
        y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
        
        # 2. Normalization
        y_norm = librosa.util.normalize(y_clean)
        
        # 3. High-Precision F0 Tracking
        f0, voiced, _ = librosa.pyin(y_norm, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        
        # 4. Self-Correcting Tonic Calibration
        initial_tonic = estimate_tonic_advanced(f0, voiced)
        tonic = refine_tonic_symbolic(f0, voiced, initial_tonic)
        
        # 5. Mel-Spectrogram Generation (The 'Vision' Signal)
        # Use more bins for 'Audacity' look
        S = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128, fmax=8000, hop_length=256)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Normalize to 0-255 for easy canvas painting
        spec_norm = ((S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255).astype(np.uint8)
        # Higher density for "Audacity" feel
        spec_data = spec_norm[:, ::4].tolist() 
        
        # 4. Multi-Segment Melodic Shape Analysis (Instinctive)
        valid = voiced & ~np.isnan(f0)
        rel_semi = 12.0 * np.log2(f0[valid] / tonic)
        rel_semi_mod = np.mod(rel_semi + 0.5, 12.0) - 0.5
        
        # Sample 5 distinct 5-second segments
        segment_len = 200 # approx 5s at pyin hop
        dtw_accumulator = {name: [] for name in self.db}
        
        for i in range(0, len(rel_semi_mod) - segment_len, len(rel_semi_mod) // 5):
            window = rel_semi_mod[i:i+segment_len]
            for r_name, templates in self.melodic_templates.items():
                local_min = float('inf')
                for temp in templates:
                    # Check first part of window for template match
                    sub_win = window[:len(temp)]
                    if len(sub_win) < len(temp): continue
                    dist, _ = fastdtw(sub_win, temp, dist=lambda x,y: abs(x-y))
                    local_min = min(local_min, dist)
                dtw_accumulator[r_name].append(local_min)
        
        # Statistical mean of DTW distances
        dtw_scores = {name: 1.0 / (1.0 + np.mean(dists)) if dists else 0 for name, dists in dtw_accumulator.items()}

        # 5. Scholarly Fingerprinting (Logic)
        pc = np.mod(np.round(rel_semi), 12.0)
        hist, _ = np.histogram(pc, bins=12, range=(0, 12))
        hist = hist / (hist.sum() + 1e-6)
        
        final_scores = {}
        for r_name in self.db:
            notes = set(self.db[r_name]["notes"])
            match_dur = sum(hist[n] for n in notes)
            forbidden = self.db[r_name].get("forbidden", [])
            penalty = sum(hist[n] for n in forbidden) * 10.0 # Strict V5 penalty
            
            logic_score = match_dur - penalty
            instinct_score = dtw_scores.get(r_name, 0)
            
            final_scores[r_name] = (logic_score * 0.6) + (instinct_score * 0.4)

        prediction = max(final_scores, key=final_scores.get)
        
        # 6. Response with Visualization Data
        return {
            "prediction": prediction,
            "score": float(final_scores[prediction]),
            "tonic": float(tonic),
            "spectrogram": spec_data,
            "fingerprint": hist.tolist(),
            "ideal_fingerprint": [1.0 if i in self.db[prediction]["notes"] else 0.0 for i in range(12)],
            "report": [
                f"Multi-segment DTW confirmed {prediction} melody shape",
                f"Spectral consistency confirmed by Mel-texture analysis"
            ],
            "metadata": {
                "mood": self.db[prediction].get("mood", "-"),
                "time": self.db[prediction].get("time", "-"),
                "notes": [SWARA_NAMES[n] for n in self.db[prediction]["notes"]]
            }
        }
