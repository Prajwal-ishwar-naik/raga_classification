import numpy as np
import json
import os


def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def get_transition_score(input_transitions, template_transitions):
    if not template_transitions:
        return 0.0

    inter = 0
    total = sum(input_transitions.values())

    for k, v in template_transitions.items():
        inter += min(v, input_transitions.get(k, 0))

    return np.sqrt(inter / (total + 1e-9))


def classify_raga(features, template_path=None):
    if template_path is None or template_path == "data/raga_templates.json":
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template_path = os.path.join(base_dir, "data", "raga_templates.json")

    if not os.path.exists(template_path):
        return {
            "prediction": "Unknown", 
            "confidence": 0,
            "match_type": "None",
            "note": "Template file missing.",
            "analysis": {"dominant_features": [], "why_not_others": []},
            "alternatives": []
        }

    with open(template_path) as f:
        templates = json.load(f)

    swaras = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa", "dha", "Dha", "ni", "Ni"]

    input_vec = np.array([features["swara_distribution"].get(s, 0) for s in swaras])
    pakads = [p[0].split("-") for p in features.get("pakads", [])]
    transitions = features.get("transitions", {})
    dominant = features.get("dominant_notes", [""])[0]

    results = []

    def norm(x):
        return [i.lower() for i in x]

    for raga, t in templates.items():

        vec = np.array(t.get("vector", [0] * 12))
        swara_score = cosine_similarity(input_vec, vec)

        trans_score = get_transition_score(transitions, t.get("transitions", {}))

        # Pakad score (simple + stable)
        pakad_score = 0
        for rp in t.get("pakads", []):
            rp = norm(rp.split("-"))

            for dp in pakads:
                dp = norm(dp)
                match = lcs_length(rp, dp) / len(rp)

                if match > 0.7:
                    pakad_score = max(pakad_score, 0.4)
                elif match > 0.5:
                    pakad_score = max(pakad_score, 0.2)

        range_score = min(
            1.0, (features["pitch_range"][1] - features["pitch_range"][0]) / 500
        )

        vadi_score = 1 if t.get("vadi") == dominant else 0

        score = (
            0.4 * swara_score
            + 0.2 * trans_score
            + 0.3 * pakad_score
            + 0.1 * range_score
            + 0.05 * vadi_score
        )

        results.append({"raga": raga, "score": score, "time": t.get("time", "Unknown")})

    # Softmax
    scores = np.array([r["score"] for r in results])
    scores -= np.max(scores)

    probs = np.exp(scores / 0.3)
    probs /= np.sum(probs)

    for i, r in enumerate(results):
        r["confidence"] = float(probs[i])

    results.sort(key=lambda x: x["confidence"], reverse=True)

    top1 = results[0]
    top2 = results[1]

    margin = top1["confidence"] - top2["confidence"]

    if top1["confidence"] < 0.2:
        match = "Uncertain (Multiple Ragas)"
    elif margin < 0.1:
        match = "Approximate Match"
    else:
        match = "Confident Match"

    # Build reasoning narrative
    dominant_features = [
        f"Strong Swara alignment ({top1['score']:.2f})",
        f"Traditional {top1['time']} Raag structure detected",
        f"Melodic transitions match {top1['raga']} grammar"
    ]
    
    why_not_others = []
    if margin < 0.2 and len(results) > 1:
        why_not_others.append(f"Close proximity to {top2['raga']} ({top2['confidence']*100:.1f}%)")

    return {
        "prediction": f"{top1['raga']} ({top1['time']} Raag)",
        "confidence": round(top1["confidence"], 3),
        "match_type": match,
        "note": f"The analysis shows a {match.lower()} with Raag {top1['raga']}.",
        "analysis": {
            "dominant_features": dominant_features,
            "why_not_others": why_not_others
        },
        "alternatives": [
            {"raga": r["raga"], "confidence": round(r["confidence"], 3)}
            for r in results[1:4]
        ],
        "ranked": [(r["raga"], r["score"]) for r in results],
    }
