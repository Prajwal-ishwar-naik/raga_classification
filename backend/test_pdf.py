from pdf_generator import generate_report_pdf
import os

data = {
    "filename": "test.wav",
    "prediction": "Day Raga",
    "neural_mood": "Calm",
    "narrative": "This is a test narrative.",
    "therapy": {
        "therapy_scores": {"calm_score": 8, "energy_score": 4, "focus_score": 7},
        "recommendation": {"primary": "Meditation"},
        "explanation": ["Point 1", "Point 2"]
    },
    "detailed_features": {
        "swaras": {"unique": "Sa, Re, Ga"},
        "arohana_avarohana": {"arohana": "Sa Re Ga", "avarohana": "Ga Re Sa"},
        "tempo": {"bpm": 80}
    }
}

try:
    generate_report_pdf(data, "test_report.pdf")
    print("PDF Generated Successfully")
except Exception as e:
    print(f"PDF Generation Failed: {e}")
finally:
    if os.path.exists("test_report.pdf"):
        os.remove("test_report.pdf")
