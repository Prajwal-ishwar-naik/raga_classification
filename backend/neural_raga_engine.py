import torch
import librosa
import numpy as np
import os
from transformers import ClapModel, ClapProcessor
from pathlib import Path
from audacity_loader import load_audacity_project
# Import symbolic logic from scholar_listener
from scholar_listener import (
    RAGA_DB_V3, SWARA_NAMES, estimate_tonic_advanced, 
    refine_tonic_symbolic, score_raga_logic, transcribe_notes
)
from advanced_features import extract_all_features
from therapy_engine import get_therapy_output
from visualizer import plot_pitch_contour, plot_spectrogram, plot_full_dashboard

import sys
# Resolve project root (one level above backend/)
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

class HybridRagaVision:
    def __init__(self, model_id="laion/clap-htsat-fused"):
        print(f"[INIT] Loading Hybrid Neural-Symbolic Engine: {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # High-Precision Neural Mood Concepts (Global Context)
        self.neural_descriptions = [
            "Prabhat Samay early morning Hindustani classical raga with komal swaras and meditative drone",
            "Madhyahna bright midday Indian classical raga with sharp melodic movements",
            "Sayankal romantic evening Hindustani raga with deep resonance and flat notes",
            "Ratri deep meditative late night Indian classical raga performance with slow tempo"
        ]
        
        # Encode Neural Prompts
        print("[INIT] Encoding Latent Mood Concepts...")
        with torch.no_grad():
            inputs = self.processor(text=self.neural_descriptions, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            self.text_embeds = outputs.text_embeds if hasattr(outputs, 'text_embeds') else (outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0])
            self.text_embeds = self.text_embeds / self.text_embeds.norm(p=2, dim=-1, keepdim=True)

    def analyze(self, filepath, duration=30, original_filename="", file_id=None):
        """
        Hyper-Spectral Semantic-Acoustic Fusion Pipeline (Modular Refactor)
        """
        from utils.chunking import get_chunks
        from utils.aggregation import aggregate_features
        from core.classifier import classify_raga
        
        # --- PHASE 0: ROBUST TONIC LOCKING ---
        # Load a small sample for tonic estimation
        y_tonic, sr_tonic = librosa.load(filepath, sr=22050, duration=10)
        f0_t, voiced_t, _ = librosa.pyin(y_tonic, sr=sr_tonic, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        initial_tonic = estimate_tonic_advanced(f0_t, voiced_t)
        locked_tonic = refine_tonic_symbolic(f0_t, voiced_t, initial_tonic)
        print(f"[HYBRID] Tonic Locked: {locked_tonic:.2f} Hz")

        # --- PHASE 1: AUDIO CHUNKING ---
        chunks = get_chunks(filepath, sr=22050, chunk_size=20, step_size=10)
        if not chunks:
            print("[ERROR] No audio chunks created.")
            raise ValueError("Audio file too short or invalid")

        # --- PHASE 2: FEATURE EXTRACTION (PER CHUNK) ---
        chunk_features = []
        full_swara_sequence = []
        first_chunk_text = ""
        first_f0 = None
        first_voiced = None
        pitch_contour = []
        
        for i, chunk in enumerate(chunks):
            if i * 10 > (duration if duration else 120): break
            
            features = extract_all_features(chunk, sr=22050, tonic_hz=locked_tonic)
            chunk_features.append(features["metadata"])
            full_swara_sequence.extend(features["metadata"].get("swara_sequence", []))
            if i == 0:
                first_chunk_text = features.get("text", "")
                detailed_features = features.get("detailed_features", {})
                first_f0 = features.get("_f0")
                first_voiced = features.get("_voiced")
                pitch_contour = features.get("pitch_contour", [])

        # --- PHASE 3: AGGREGATION ---
        aggregated = aggregate_features(chunk_features)
        # Add the global sequence for pakad detection
        aggregated["swara_sequence"] = full_swara_sequence

        # --- PHASE 4: HYBRID CLASSIFICATION ---
        res = classify_raga(aggregated)
        
        # === NEW FEATURES (SA STABILITY, NYAS, CONFIDENCE) ===
        from collections import Counter
        
        # 1. Sa Stability Score
        global_seq = aggregated.get("swara_sequence", [])
        sa_stability = round(global_seq.count("Sa") / max(len(global_seq), 1), 3) if global_seq else 0.0
        
        # 2. Nyas Swaras
        phrase_endings = [chunk.get("swara_sequence", [])[-1] for chunk in chunk_features if chunk.get("swara_sequence")]
        nyas_swaras = [note for note, count in Counter(phrase_endings).most_common(2)]
        
        # 3. Confidence Explanation
        confidence_reason = []
        if res["confidence"] > 0.7:
            confidence_reason.append("Strong swara alignment")
        elif res["confidence"] > 0.4:
            confidence_reason.append("Moderate swara alignment")
            
        if aggregated.get("pakads"):
            confidence_reason.append("Strong pakad match")
            
        if aggregated.get("transitions"):
            confidence_reason.append("Melodic transitions match grammar")
            
        detailed_features["advanced_analytics"] = {
            "sa_stability": sa_stability,
            "nyas_swaras": nyas_swaras,
            "confidence_reason": confidence_reason
        }
        
        # Inject into metadata
        aggregated["sa_stability"] = sa_stability
        aggregated["nyas_swaras"] = nyas_swaras
        aggregated["confidence_reason"] = confidence_reason
        # =====================================================
        
        # --- PHASE 5: NEURAL MOOD (CLAP) ---
        # Use first 5s for mood context
        mood_audio = chunks[0][:5*22050]
        mood_audio_48k = librosa.resample(mood_audio, orig_sr=22050, target_sr=48000)
        
        neural_mood = "Unknown"
        confidence = 0.5
        try:
            inputs = self.processor(audio=mood_audio_48k, return_tensors="pt", sampling_rate=48000).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_audio_features(**inputs)
                audio_embeds = outputs.audio_embeds if hasattr(outputs, 'audio_embeds') else (outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0])
                audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
                similarity = (audio_embeds @ self.text_embeds.T).squeeze(0)
                probs = torch.nn.functional.softmax(similarity * 10, dim=-1).cpu().numpy()
                neural_mood = "Day" if (probs[0] + probs[1]) > (probs[2] + probs[3]) else "Night"
                confidence = float(max(probs[0]+probs[1], probs[2]+probs[3]))
        except Exception as e:
            print(f"[NEURAL MOOD ERROR] {e}")

        # --- PHASE 6: VISUALIZATION ---
        image_url = None
        spectrogram_url = None
        if original_filename:
            stem = Path(original_filename).stem
            spec_path = BASE_DIR / "static" / f"spec_{stem}.png"
            dash_path = BASE_DIR / "static" / f"dash_{stem}.png"
            output_image_path = BASE_DIR / "output" / f"analysis_{stem}.png"
            
            try:
                plot_spectrogram(chunks[0], 22050, str(spec_path))
                spectrogram_url = f"/static/spec_{stem}.png"
                
                # Check if a pre-generated analysis image exists in the output folder
                if output_image_path.exists():
                    image_url = f"/output/analysis_{stem}.png"
                else:
                    # Fallback: Generate Full Dashboard dynamically
                    dashboard_features = {
                        "_f0": first_f0,
                        "_voiced": first_voiced,
                        "swara_distribution": aggregated.get("swara_distribution", {}),
                        "prediction": res.get("prediction", "Unknown")
                    }
                    confidence = res.get("confidence", 0.5)
                    pred = res.get("prediction", "Day")
                    if pred == "Day":
                        ranked = [("Day", confidence), ("Night", 1.0 - confidence)]
                    else:
                        ranked = [("Night", confidence), ("Day", 1.0 - confidence)]
                    
                    plot_full_dashboard(dashboard_features, ranked, str(dash_path), stem)
                    image_url = f"/static/dash_{stem}.png"
                
            except Exception as e:
                print(f"[VISUALIZATION ERROR] {e}")

        # Generate UI Spectrogram Data
        S = librosa.feature.melspectrogram(y=chunks[0][:10*22050], sr=22050, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        spec_data = ((S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255).astype(np.uint8).tolist()

        # Final narrative construction
        narrative = res.get("note", "") + "\n\n" + "\n".join(res["analysis"]["dominant_features"])
        if res["analysis"]["why_not_others"]:
            narrative += "\n\nRejections:\n" + "\n".join(res["analysis"]["why_not_others"])

        return {
            "prediction": res["prediction"],
            "neural_prediction": res["prediction"],
            "neural_mood": neural_mood,
            "detected_raag": res["prediction"],
            "confidence": res["confidence"],
            "neural_confidence": res["confidence"], # Added for frontend compatibility
            "logic_score": res["confidence"],
            "spectrogram": spec_data,
            "image_url": image_url,
            "narrative": narrative,
            "metadata": {
                "time": neural_mood,
                "swaras": sorted(list(aggregated.get("swara_distribution", {}).keys())), # Convert to list for chips
                "advanced_features": {
                    **aggregated,
                    "pitch_range": aggregated["pitch_range"][1] - aggregated["pitch_range"][0] # Convert to scalar for UI
                }
            },
            "report": res["analysis"]["dominant_features"],
            "detailed_features": detailed_features,
            "pitch_contour_data": pitch_contour,
            "swara_distribution_data": aggregated.get("swara_distribution", {}),
            "therapy": get_therapy_output({"metadata": aggregated}),
            "therapy_recommendation": get_therapy_output({"metadata": aggregated}),
            "spectrogram_url": spectrogram_url
        }

    def cognitive_reasoning(self, raga, mood, confidence, logic, swaras):
        """
        Hyper-Advanced Reasoning Bridge: Uses Ollama if available, else highly specialized template.
        """
        import requests
        prompt = (
            f"As an AI Musicologist expert in Hindustani Classical Music, explain this result:\n"
            f"- Identified Raag: {raga}\n"
            f"- Neural Mood Context: {mood}\n"
            f"- Confidence: {confidence*100:.1f}%\n"
            f"- Evidence: {', '.join(logic)}\n"
            f"- Swaras Detected: {', '.join(swaras)}\n\n"
            f"Provide a brief, professional, and insightful musical analysis (3-4 sentences)."
        )

        # Attempt Ollama reasoning
        try:
            r = requests.post("http://localhost:11434/api/generate", 
                              json={"model": "llama3", "prompt": prompt, "stream": False},
                              timeout=2.0)
            if r.status_code == 200:
                print(f"[COGNITIVE] Reasoning generated by Ollama for {raga}.")
                return r.json()["response"]
        except Exception:
            pass
        
        # Fallback optimized template reasoning
        print("[COGNITIVE] Ollama offline. Using template reasoning.")
        base = f"The {mood} mood was detected via multi-point temporal sampling ({confidence*100:.1f}%). "
        if raga != "Unknown":
            base += f"Raag {raga} was identified logically by the Swara presence of {', '.join(swaras)}. "
            base += f"Grammatical evidence: {' '.join(logic[:2])}."
        else:
            base += "Melodic features were identified but didn't match a specific raga signature precisely."
        return base
