import numpy as np

def compute_therapy_scores(features):
    """
    Computes Calm, Energy, and Focus scores based on musical features.
    Uses continuous mapping for higher sensitivity.
    """
    # Initialize raw scores
    calm = 0.0
    energy = 0.0
    focus = 0.0
    
    metadata = features.get("metadata", {})
    tempo = metadata.get("tempo", 80)
    pitch_range = metadata.get("pitch_range", 150)
    
    # Advanced features
    gamakas = metadata.get("gamakas", {})
    slides = gamakas.get("slides", "No")
    oscillations = gamakas.get("oscillations", 0)
    
    # Transitions
    transitions = metadata.get("transitions", {})
    trans_pct = transitions.get("pct", 0)
    
    # 1. TEMPO LOGIC (Continuous)
    if tempo < 60:
        calm += 5.0
        focus += 2.0
    elif tempo < 90:
        calm += (90 - tempo) / 30 * 3 + 2
        focus += 4.0
    elif tempo < 120:
        energy += (tempo - 90) / 30 * 3 + 2
        focus += 3.0
    else:
        energy += 6.0
        calm -= 1.0

    # 2. PITCH STABILITY (Continuous)
    # Range proxy for melodic activity
    if pitch_range < 100:
        calm += 3.0
        focus += 2.0
    elif pitch_range < 300:
        calm += (300 - pitch_range) / 200 * 2
        energy += (pitch_range - 100) / 200 * 3
    else:
        energy += 4.0
        focus += 1.0

    # 3. GAMAKAS & SLIDES
    if slides == "Yes":
        energy += 1.5
        calm -= 0.5
    else:
        calm += 2.0
        focus += 1.0
        
    if oscillations > 30:
        energy += 2.0
    elif oscillations > 10:
        energy += 1.0
    
    # 4. TRANSITION COMPLEXITY
    if trans_pct > 15:
        energy += 2.0
        focus -= 1.0
    elif trans_pct > 5:
        focus += 2.0
        energy += 0.5
    else:
        calm += 2.0

    # Base bias for stability
    calm = max(1.0, calm + 1.0)
    energy = max(1.0, energy + 1.0)
    focus = max(1.0, focus + 1.0)

    # Final Normalization to 0.0 - 10.0
    total = calm + energy + focus
    if total == 0: return {"calm_score": 3.3, "energy_score": 3.3, "focus_score": 3.3}
    
    # We want scores to represent "prominence" in a 0-10 range
    # Scaling to ensure they aren't all just 3.3
    # We use a non-linear boost to emphasize differences
    return {
        "calm_score": round(min(10.0, (calm / 8.0) * 10), 1),
        "energy_score": round(min(10.0, (energy / 8.0) * 10), 1),
        "focus_score": round(min(10.0, (focus / 8.0) * 10), 1)
    }

def generate_therapy_recommendation(scores):
    """
    Maps scores to therapeutic recommendations with more nuanced logic.
    """
    c = scores["calm_score"]
    e = scores["energy_score"]
    f = scores["focus_score"]
    
    # Determine dominant mode
    max_score = max(c, e, f)
    
    if max_score == c:
        primary = "Stress Relief, Meditation, and Deep Sleep"
        secondary = ["Anxiety Reduction", "Emotional Balancing", "Parasympathetic Activation"]
    elif max_score == e:
        primary = "Motivation, Physical Workout, and Vitality"
        secondary = ["Mood Elevation", "Active Engagement", "Dopaminergic Stimulation"]
    else:
        primary = "Study, Cognitive Concentration, and Productivity"
        secondary = ["Mental Clarity", "Memory Retention", "Neural Syncronization"]
        
    return {
        "primary": primary,
        "secondary": secondary
    }

def generate_therapy_explanation(features, scores):
    """
    Generates detailed, dynamic explanations.
    """
    metadata = features.get("metadata", {})
    tempo = metadata.get("tempo", 80)
    pitch_range = metadata.get("pitch_range", 150)
    gamakas = metadata.get("gamakas", {})
    slides = gamakas.get("slides", "No")
    transitions = metadata.get("transitions", {})
    trans_str = transitions.get("most_common", "None")

    explanations = []
    
    if tempo < 85:
        explanations.append(f"The slow rhythmic pulse ({tempo} BPM) promotes a lower heart rate and calm state.")
    elif tempo > 110:
        explanations.append(f"The high temporal density ({tempo} BPM) creates an uplifting, high-arousal energy.")
    else:
        explanations.append(f"The moderate, steady tempo ({tempo} BPM) supports alpha-wave brain activity for focus.")
        
    if pitch_range > 250:
        explanations.append(f"Wide pitch intervals ({pitch_range:.0f} Hz) stimulate emotional engagement and curiosity.")
    else:
        explanations.append(f"Minimal pitch variation ({pitch_range:.0f} Hz) ensures a stable, non-intrusive sonic environment.")
        
    if slides == "Yes":
        explanations.append("The presence of melodic slides (Gamakas) adds an expressive, human-like quality that aids empathy.")
    else:
        explanations.append("Clean note articulations help in maintaining clear cognitive boundaries during work.")
        
    explanations.append(f"The recurring '{trans_str}' note movement provides a familiar anchor for the mind.")
    
    return explanations

def get_therapy_output(features):
    """
    Main entry point for the therapy recommendation module.
    """
    if not features or "metadata" not in features:
        return {
            "therapy_scores": {"calm_score": 5.0, "energy_score": 5.0, "focus_score": 5.0},
            "recommendation": {"primary": "General Wellness", "secondary": ["Equilibrium"]},
            "explanation": ["Standard profile applied due to lack of deep feature metadata."]
        }
        
    scores = compute_therapy_scores(features)
    recommendation = generate_therapy_recommendation(scores)
    explanation = generate_therapy_explanation(features, scores)
    
    return {
        "therapy_scores": scores,
        "recommendation": recommendation,
        "explanation": explanation
    }
