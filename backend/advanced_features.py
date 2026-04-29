import numpy as np
import librosa
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

SWARA_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa", "dha", "Dha", "ni", "Ni"]

def extract_swaras(f0, voiced, tonic_hz, sr=22050, hop_length=512):
    valid_f0 = f0[voiced & ~np.isnan(f0)]
    if len(valid_f0) == 0:
        return {"detected": [], "unique": [], "most_frequent": ("None", 0)}
    
    semitones = 12.0 * np.log2(valid_f0 / tonic_hz)
    pc = np.mod(np.round(semitones), 12.0).astype(int)
    
    counts = Counter(pc)
    total = len(pc)
    
    unique_swaras = [SWARA_NAMES[i] for i in sorted(counts.keys())]
    frame_dur = hop_length / sr
    
    detected = []
    for i in sorted(counts.keys()):
        dur = counts[i] * frame_dur
        if dur > 0.05:  # filter noise
            detected.append(f"{SWARA_NAMES[i]} ({dur:.1f}s)")
            
    most_freq_idx = counts.most_common(1)[0][0]
    most_freq_pct = (counts[most_freq_idx] / total) * 100
    
    return {
        "detected": detected,
        "unique": unique_swaras,
        "most_frequent": (SWARA_NAMES[most_freq_idx], most_freq_pct)
    }

def extract_arohana_avarohana(f0, voiced, tonic_hz):
    valid_f0 = f0[voiced & ~np.isnan(f0)]
    if len(valid_f0) < 10:
        return {"arohana": "Unknown", "avarohana": "Unknown"}
    
    semitones = 12.0 * np.log2(valid_f0 / tonic_hz)
    pc = np.mod(np.round(semitones), 12.0).astype(int)
    
    seq = [pc[0]]
    for p in pc[1:]:
        if p != seq[-1]:
            seq.append(p)
            
    arohana_seqs = []
    avarohana_seqs = []
    
    curr_asc = [seq[0]]
    curr_desc = [seq[0]]
    
    for i in range(1, len(seq)):
        if seq[i] > seq[i-1]:
            curr_asc.append(seq[i])
        else:
            if len(curr_asc) > 2: arohana_seqs.append(curr_asc)
            curr_asc = [seq[i]]
            
        if seq[i] < seq[i-1]:
            curr_desc.append(seq[i])
        else:
            if len(curr_desc) > 2: avarohana_seqs.append(curr_desc)
            curr_desc = [seq[i]]
            
    if len(curr_asc) > 2: arohana_seqs.append(curr_asc)
    if len(curr_desc) > 2: avarohana_seqs.append(curr_desc)
    
    best_aro = max(arohana_seqs, key=len) if arohana_seqs else sorted(list(set(seq)))
    best_ava = max(avarohana_seqs, key=len) if avarohana_seqs else sorted(list(set(seq)), reverse=True)
    
    return {
        "arohana": " → ".join([SWARA_NAMES[i] for i in best_aro]),
        "avarohana": " → ".join([SWARA_NAMES[i] for i in best_ava])
    }

def detect_pakad(f0, voiced, tonic_hz):
    valid_f0 = f0[voiced & ~np.isnan(f0)]
    if len(valid_f0) < 10: return []
    semitones = 12.0 * np.log2(valid_f0 / tonic_hz)
    pc = np.mod(np.round(semitones), 12.0).astype(int)
    seq = [pc[0]]
    for p in pc[1:]:
        if p != seq[-1]:
            seq.append(p)
            
    ngrams = []
    for i in range(len(seq)-3):
        ngrams.append(tuple(seq[i:i+4]))
        
    counts = Counter(ngrams)
    top = counts.most_common(2)
    return [" ".join([SWARA_NAMES[n] for n in p[0]]) for p in top]

def analyze_gamakas(f0, voiced):
    valid_f0 = f0[voiced & ~np.isnan(f0)]
    if len(valid_f0) < 10:
        return {"oscillations": 0, "slides": "No", "avg_var": 0}
    
    diff = np.diff(valid_f0)
    oscillations = np.sum(np.diff(np.sign(diff[diff != 0])) != 0)
    slides = "Yes" if np.max(np.abs(diff)) > 15 else "No"
    avg_var = np.mean(np.abs(diff))
    
    return {"oscillations": int(oscillations), "slides": slides, "avg_var": round(avg_var, 1)}

def get_vadi_samvadi(f0, voiced, tonic_hz):
    valid_f0 = f0[voiced & ~np.isnan(f0)]
    if len(valid_f0) < 10:
        return {"vadi": "Unknown", "samvadi": "Unknown"}
    semitones = 12.0 * np.log2(valid_f0 / tonic_hz)
    pc = np.mod(np.round(semitones), 12.0).astype(int)
    counts = Counter(pc)
    top = counts.most_common(2)
    vadi = SWARA_NAMES[top[0][0]] if len(top) > 0 else "Unknown"
    samvadi = SWARA_NAMES[top[1][0]] if len(top) > 1 else "Unknown"
    return {"vadi": vadi, "samvadi": samvadi}

def get_pitch_dist(f0, voiced):
    valid_f0 = f0[voiced & ~np.isnan(f0)]
    if len(valid_f0) < 10:
        return {"min": 0, "max": 0, "range": 0}
    min_f = np.min(valid_f0)
    max_f = np.max(valid_f0)
    return {"min": round(min_f, 1), "max": round(max_f, 1), "range": round(max_f - min_f, 1)}

def get_note_transitions(f0, voiced, tonic_hz):
    valid_f0 = f0[voiced & ~np.isnan(f0)]
    if len(valid_f0) < 10:
        return {"most_common": "Unknown", "pct": 0}
    semitones = 12.0 * np.log2(valid_f0 / tonic_hz)
    pc = np.mod(np.round(semitones), 12.0).astype(int)
    seq = [pc[0]]
    for p in pc[1:]:
        if p != seq[-1]:
            seq.append(p)
    transitions = []
    for i in range(len(seq)-1):
        transitions.append((seq[i], seq[i+1]))
    counts = Counter(transitions)
    if not counts: return {"most_common": "Unknown", "pct": 0}
    most_common = counts.most_common(1)[0]
    pct = (most_common[1] / len(transitions)) * 100
    t_str = f"{SWARA_NAMES[most_common[0][0]]} → {SWARA_NAMES[most_common[0][1]]}"
    return {"most_common": t_str, "pct": round(pct)}

def get_tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if tempo.size == 0:
        return 0
    return round(float(np.atleast_1d(tempo)[0]))

def get_structure(y, sr):
    dur = len(y) / sr
    if dur < 20:
        return f"0-{int(dur)}s: Alap"
    else:
        return f"0-20s: Alap\n20-{int(dur)}s: Bandish"

def get_timbre(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    return {
        "mfcc_mean": np.round(np.mean(mfcc, axis=1)[:3], 2).tolist(),
        "centroid": round(float(np.mean(centroid)), 1),
        "zcr": round(float(np.mean(zcr)), 4)
    }

def extract_all_features(y, sr=22050):
    # Ultra-Fast Pitch Tracking with Yin
    f0 = librosa.yin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=1024)
    voiced_flag = np.ones_like(f0).astype(bool) # Yin doesn't return voiced flag directly in this version, or we can use energy threshold
    # Estimate voicing based on energy
    rms = librosa.feature.rms(y=y, hop_length=1024)[0]
    voiced_flag = rms > (np.mean(rms) * 0.5)
    # Ensure shapes match
    if len(voiced_flag) > len(f0): voiced_flag = voiced_flag[:len(f0)]
    if len(f0) > len(voiced_flag): f0 = f0[:len(voiced_flag)]
    
    valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
    tonic_hz = float(np.median(valid_f0)) if len(valid_f0) > 0 else 220.0
    
    swaras = extract_swaras(f0, voiced_flag, tonic_hz, sr)
    aro_ava = extract_arohana_avarohana(f0, voiced_flag, tonic_hz)
    pakad = detect_pakad(f0, voiced_flag, tonic_hz)
    gamakas = analyze_gamakas(f0, voiced_flag)
    vadi_samvadi = get_vadi_samvadi(f0, voiced_flag, tonic_hz)
    pitch_dist = get_pitch_dist(f0, voiced_flag)
    transitions = get_note_transitions(f0, voiced_flag, tonic_hz)
    tempo = get_tempo(y, sr)
    structure = get_structure(y, sr)
    timbre = get_timbre(y, sr)
    
    p1 = f"{pakad[0]}" if len(pakad) > 0 else 'None'
    p2 = f"2. {pakad[1]}" if len(pakad) > 1 else ''
    
    output = f"""
================ FEATURE ANALYSIS ================

🎵 Swaras:
Detected: {', '.join(swaras['detected'][:5])} ...
Unique: {', '.join(swaras['unique'])}
Most Frequent: {swaras['most_frequent'][0]} ({swaras['most_frequent'][1]:.0f}%)

📈 Arohana–Avarohana:
Arohana: {aro_ava['arohana']}
Avarohana: {aro_ava['avarohana']}

🎯 Pakad:
1. {p1}
{p2}

🎶 Gamakas:
Oscillations: {gamakas['oscillations']}
Slides detected: {gamakas['slides']}
Avg Pitch Variation: {gamakas['avg_var']} Hz

⭐ Vadi–Samvadi:
Vadi: {vadi_samvadi['vadi']}
Samvadi: {vadi_samvadi['samvadi']}

📊 Pitch Range:
Min: {pitch_dist['min']} Hz, Max: {pitch_dist['max']} Hz
Range: {pitch_dist['range']} Hz

🔁 Note Transitions:
Most Common: {transitions['most_common']} ({transitions['pct']}%)

🥁 Tempo:
BPM: {tempo}

🎼 Structure:
{structure}

🎧 Timbre:
MFCC Mean: {timbre['mfcc_mean']}...
Spectral Centroid: {timbre['centroid']}
ZCR: {timbre['zcr']}

================================================="""
    print(output)
    return {
        "text": output,
        "pitch_contour": valid_f0.tolist(),
        "_f0": f0,
        "_voiced": voiced_flag,
        "metadata": {
            "swaras": swaras['detected'],
            "unique_swaras": swaras['unique'],
            "most_frequent": swaras['most_frequent'][0],
            "vadi": vadi_samvadi['vadi'],
            "samvadi": vadi_samvadi['samvadi'],
            "tempo": tempo,
            "gamakas": gamakas,
            "pitch_range": pitch_dist['range'],
            "transitions": transitions
        }
    }
