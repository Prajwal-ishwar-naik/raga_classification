import numpy as np

def compute_therapy_scores(features):
    """
    Computes Calm, Energy, and Focus scores based on musical features.
    Uses continuous mapping for higher sensitivity.
    """
    # Initialize raw scores
    calm = 5.0
    energy = 5.0
    focus = 5.0
    
    if not features:
        return {"calm_score": calm, "energy_score": energy, "focus_score": focus}

    metadata = features.get("metadata", {}) or {}
    tempo = metadata.get("tempo", 80) or 80
    pitch_range_raw = metadata.get("pitch_range", 150)
    
    pitch_range = 150
    if isinstance(pitch_range_raw, list) and len(pitch_range_raw) >= 2:
        if pitch_range_raw[0] is not None and pitch_range_raw[1] is not None:
            pitch_range = pitch_range_raw[1] - pitch_range_raw[0]
    elif isinstance(pitch_range_raw, (int, float)):
        pitch_range = pitch_range_raw
        
    # Advanced features
    gamakas = metadata.get("gamakas", {}) or {}
    slides = gamakas.get("slides", "No")
    oscillations = gamakas.get("oscillations", 0) or 0
    
    # Transitions
    transitions = metadata.get("transitions", {}) or {}
    trans_pct = transitions.get("pct", 0) or 0
    
    # 1. TEMPO LOGIC (Continuous)
    if tempo < 60:
        calm += 2.0
        focus += 1.0
    elif tempo < 90:
        calm += 1.0
        focus += 2.0
    elif tempo < 120:
        energy += 2.0
        focus += 1.0
    else:
        energy += 3.0
        calm -= 1.0

    # Ensure scores are in 0-10 range
    return {
        "calm_score": round(max(0, min(10, calm)), 1),
        "energy_score": round(max(0, min(10, energy)), 1),
        "focus_score": round(max(0, min(10, focus)), 1)
    }

def generate_therapy_recommendation(scores):
    c, e, f = scores["calm_score"], scores["energy_score"], scores["focus_score"]
    
    if c > e and c > f:
        return {"primary": "Stress Reduction & Meditation", "secondary": ["Anxiety Relief", "Deep Sleep Preparation"]}
    if e > c and e > f:
        return {"primary": "Mood Elevation & Vitality", "secondary": ["Morning Energy Boost", "Creative Flow"]}
    if f > c and f > e:
        return {"primary": "Cognitive Focus & Study", "secondary": ["Deep Work Support", "Mental Clarity"]}
    
    return {"primary": "General Emotional Balance", "secondary": ["Daily Wellness", "Equilibrium"]}

def generate_therapy_explanation(features, scores):
    metadata = (features.get("metadata", {}) or {}) if features else {}
    tempo = metadata.get("tempo", 80) or 80
    pitch_range_raw = metadata.get("pitch_range", 150)
    
    pitch_range = 150
    if isinstance(pitch_range_raw, list) and len(pitch_range_raw) >= 2:
        if pitch_range_raw[0] is not None and pitch_range_raw[1] is not None:
            pitch_range = pitch_range_raw[1] - pitch_range_raw[0]
    elif isinstance(pitch_range_raw, (int, float)):
        pitch_range = pitch_range_raw

    gamakas = metadata.get("gamakas", {}) or {}
    slides = gamakas.get("slides", "No")
    transitions = metadata.get("transitions", {}) or {}
    trans_str = transitions.get("most_common", "None") or "None"

    explanations = []
    
    if tempo < 85:
        explanations.append(f"The slow rhythmic pulse ({tempo} BPM) promotes a lower heart rate.")
    else:
        explanations.append(f"The steady rhythmic structure ({tempo} BPM) supports cognitive engagement.")
        
    if pitch_range > 200:
        explanations.append(f"Wide melodic range ({pitch_range:.0f} Hz) stimulates emotional depth.")
    else:
        explanations.append(f"Focused melodic range ({pitch_range:.0f} Hz) ensures a stable environment.")
        
    if slides == "Yes":
        explanations.append("Melodic slides (Gamakas) add an expressive quality that aids emotional processing.")
        
    explanations.append(f"The recurring '{trans_str}' note movement provides a familiar anchor for the mind.")
    
    return explanations

from raga_db import RAGA_DB_V3

RAGA_THERAPY_DB = {
    "Bhairav": {
        "rasa": "Peace & Serenity",
        "science_note": "Early morning ragas with Komal Re and Dha are known to stabilize cortisol levels.",
        "session_plan": ["Bhairav (15m Alap)", "Nat Bhairav (20m Gat)", "Ahir Bhairav (10m Conclusion)"]
    },
    "Yaman": {
        "rasa": "Love & Devotion",
        "science_note": "Teevra Ma in Yaman creates emotional depth, stimulating oxytocin.",
        "session_plan": ["Yaman (20m Alap)", "Yaman Kalyan (20m Gat)", "Hamsadhwani (15m Conclusion)"]
    },
    "Bhairavi": {
        "rasa": "Universal Compassion",
        "science_note": "Bhairavi's flat swara structure is extremely grounding.",
        "session_plan": ["Bhairavi (30m Thumri)", "Malkauns (15m Alap)", "Bhairavi (5m Bhajan)"]
    }
}

def get_therapy_output(features, raga_name="Unknown"):
    if not features:
        return {
            "therapy_scores": {"calm_score": 5.0, "energy_score": 5.0, "focus_score": 5.0},
            "recommendation": {"primary": "General Balance", "secondary": []},
            "explanation": ["No data available"],
            "raga_metadata": None,
            "session_plan": []
        }
        
    scores = compute_therapy_scores(features)
    recommendation = generate_therapy_recommendation(scores)
    explanation = generate_therapy_explanation(features, scores)
    
    raga_therapy = RAGA_THERAPY_DB.get(raga_name) or {}
    raga_static = RAGA_DB_V3.get(raga_name) or {}

    return {
        "therapy_scores": scores,
        "recommendation": recommendation,
        "explanation": explanation,
        "raga_name": raga_name,
        "raga_metadata": {
            "rasa": raga_therapy.get("rasa") or raga_static.get("rasa") or "Universal",
            "science_note": raga_therapy.get("science_note") or "Musical resonance aids in neural synchronization.",
            "optimal_time": raga_static.get("optimal_time") or "Varies",
            "vadi": SWARA_NAMES[raga_static["vadi"]] if raga_static.get("vadi") is not None else "N/A",
            "samvadi": SWARA_NAMES[raga_static["samvadi"]] if raga_static.get("samvadi") is not None else "N/A",
        },
        "session_plan": raga_therapy.get("session_plan") or ["Sample Raga 1", "Sample Raga 2", "Sample Raga 3"]
    }

SWARA_NAMES = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa", "dha", "Dha", "ni", "Ni"]
