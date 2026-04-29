import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from therapy_engine import process_therapy

# Test Case 1: Low tempo, smooth
features_calm = {
    "metadata": {
        "tempo": 60,
        "gamakas": {"oscillations": 20, "avg_var": 5},
        "transitions": {"pct": 15}
    },
    "pitch_contour": [200, 205, 210, 208] # Low range
}

# Test Case 2: High tempo, high variance
features_energy = {
    "metadata": {
        "tempo": 130,
        "gamakas": {"oscillations": 200, "avg_var": 30},
        "transitions": {"pct": 5}
    },
    "pitch_contour": [200, 400, 600, 300] # High range
}

# Test Case 3: Moderate tempo, structured
features_focus = {
    "metadata": {
        "tempo": 90,
        "gamakas": {"oscillations": 40, "avg_var": 8},
        "transitions": {"pct": 35}
    },
    "pitch_contour": [250, 260, 255, 265] # Moderate range
}

def print_result(name, result):
    print(f"\n--- {name} ---")
    print(f"Scores: {result['therapy_scores']}")
    print(f"Primary: {result['recommendation']['primary_recommendation']}")
    print(f"Explanation: {result['explanation']}")

print_result("CALM TEST", process_therapy(features_calm))
print_result("ENERGY TEST", process_therapy(features_energy))
print_result("FOCUS TEST", process_therapy(features_focus))
