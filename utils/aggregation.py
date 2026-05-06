import numpy as np
from collections import Counter

def aggregate_features(chunk_features_list):
    """
    Aggregates features from multiple audio chunks.
    
    Expected format of each chunk in chunk_features_list:
    {
        "swara_distribution": dict,
        "dominant_notes": list,
        "transitions": dict,
        "pitch_range": [min, max],
        "tempo": float
    }
    """
    if not chunk_features_list:
        return {
            "swara_distribution": {},
            "dominant_notes": [],
            "transitions": {},
            "pitch_range": [0, 0],
            "tempo": 0
        }

    num_chunks = len(chunk_features_list)
    
    # 1. Swara Distribution: Average and Normalize
    aggregated_swaras = {}
    for chunk in chunk_features_list:
        dist = chunk.get("swara_distribution", {})
        for swara, val in dist.items():
            aggregated_swaras[swara] = aggregated_swaras.get(swara, 0) + val
            
    # Calculate average
    for swara in aggregated_swaras:
        aggregated_swaras[swara] /= num_chunks
        
    # Normalize (ensure total = 1.0)
    total_val = sum(aggregated_swaras.values())
    if total_val > 0:
        for swara in aggregated_swaras:
            aggregated_swaras[swara] /= total_val

    # 2. Dominant Notes: Frequency count, top 2-3
    all_dominant = []
    for chunk in chunk_features_list:
        # Some chunks might have multiple dominant notes or just one
        notes = chunk.get("dominant_notes", [])
        if isinstance(notes, list):
            all_dominant.extend(notes)
        else:
            all_dominant.append(notes)
    
    counts = Counter(all_dominant)
    dominant_notes = [note for note, count in counts.most_common(3)]

    # 3. Transitions: Merge and Sum
    aggregated_transitions = {}
    for chunk in chunk_features_list:
        trans = chunk.get("transitions", {})
        for key, freq in trans.items():
            aggregated_transitions[key] = aggregated_transitions.get(key, 0) + freq

    # 4. Pitch Range: Global min/max
    mins = []
    maxs = []
    for chunk in chunk_features_list:
        pr = chunk.get("pitch_range", [0, 0])
        if pr[0] > 0: mins.append(pr[0])
        if pr[1] > 0: maxs.append(pr[1])
        
    global_min = min(mins) if mins else 0
    global_max = max(maxs) if maxs else 0

    # 5. Tempo: Average
    tempos = [chunk.get("tempo", 0) for chunk in chunk_features_list]
    avg_tempo = sum(tempos) / len(tempos) if tempos else 0

    # 6. Pakads: Merge and Sum
    aggregated_pakads = {}
    for chunk in chunk_features_list:
        p_list = chunk.get("pakads", [])
        for phrase, count in p_list:
            aggregated_pakads[phrase] = aggregated_pakads.get(phrase, 0) + count
    
    # Sort and take top 5
    top_pakads = sorted(aggregated_pakads.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "swara_distribution": {k: round(v, 4) for k, v in aggregated_swaras.items()},
        "dominant_notes": dominant_notes,
        "transitions": aggregated_transitions,
        "pitch_range": [round(global_min, 1), round(global_max, 1)],
        "tempo": round(avg_tempo, 1),
        "pakads": top_pakads
    }
