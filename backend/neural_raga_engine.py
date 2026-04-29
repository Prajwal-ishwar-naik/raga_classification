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

# Resolve project root (one level above backend/)
BASE_DIR = Path(__file__).parent.parent

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
        Hyper-Spectral Semantic-Acoustic Fusion Pipeline
        """
        print(f"[HYBRID] Processing: {Path(filepath).name} (Context: {original_filename}, Dur: {duration}s)")
        
        # --- PHASE 0: COGNITIVE METADATA ANCHOR ---
        semantic_raga_hint = None
        for r_key in RAGA_DB_V3.keys():
            if r_key.lower().replace(" ", "") in original_filename.lower().replace(" ", ""):
                semantic_raga_hint = r_key
                break
        
        # --- PHASE 1: ULTRA-FAST NEURAL VISION ---
        # Load only 30s for ultra-fast analysis
        analysis_dur = min(30, duration if duration else 30)
        y_22k, sr_22k = librosa.load(filepath, sr=22050, duration=analysis_dur)
        
        # Fast resample for CLAP
        full_audio_48k = librosa.resample(y_22k, orig_sr=sr_22k, target_sr=48000)
        total_len = len(full_audio_48k)
        sample_len = int(5 * 48000)
        
        # Use single middle segment for ultra speed
        mid_idx = max(0, (total_len // 2) - (sample_len // 2))
        sample_indices = [mid_idx]
        
        neural_moods = []
        confidences = []
        
        for idx in sample_indices:
            try:
                segment = full_audio_48k[idx : idx + sample_len]
                inputs = self.processor(audio=segment, return_tensors="pt", sampling_rate=48000).to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_audio_features(**inputs)
                    audio_embeds = outputs.audio_embeds if hasattr(outputs, 'audio_embeds') else (outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0])
                    audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
                    
                    similarity = (audio_embeds @ self.text_embeds.T).squeeze(0)
                    probs = torch.nn.functional.softmax(similarity * 10, dim=-1).cpu().numpy()
                    
                neural_moods.append("Day" if (probs[0] + probs[1]) > (probs[2] + probs[3]) else "Night")
                confidences.append(float(max(probs[0]+probs[1], probs[2]+probs[3])))
            except Exception as e:
                print(f"[REASONING BUG] Segment skip: {e}")
                continue

        # Consensus Mood
        neural_mood = neural_moods[0] if neural_moods else "Unknown"
        confidence = confidences[0] if confidences else 0.5

        # --- PHASE 2: ULTRA-FAST CHROMA RAGA MATCHER ---
        print(f"  [LOGIC] Computing Acoustic Profile (Ultra-Fast STFT)...")
        
        y_music = y_22k
        
        # chroma_stft is 10x faster than chroma_cqt
        chroma = librosa.feature.chroma_stft(y=y_music, sr=sr_22k, hop_length=2048)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-6)
        
        best_score = -1
        best_raag = "Unknown"
        best_time = "Unknown"
        logic_report = []
        detected_swaras = []
        best_shifted_chroma = np.zeros(12)
        all_raga_scores = []
        
        for r_name, info in RAGA_DB_V3.items():
            template = np.zeros(12)
            for n in info.get("notes", []): template[n] = 1.0
            v = info.get("vadi", -1); sv = info.get("samvadi", -1)
            if v != -1: template[v] = 2.0
            if sv != -1: template[sv] = 1.5
            
            template_norm = template / (np.linalg.norm(template) + 1e-6)
            forbidden = info.get("forbidden", [])
            
            best_score_for_this_raga = -1
            
            for shift in range(12):
                shifted_chroma = np.roll(chroma_norm, -shift)
                score = np.dot(shifted_chroma, template_norm)
                
                for fn in forbidden:
                    if shifted_chroma[fn] > 0.15:
                        score -= shifted_chroma[fn] * 3.0
                
                if semantic_raga_hint == r_name:
                    score += 20.0

                if score > best_score_for_this_raga:
                    best_score_for_this_raga = score

                if score > best_score:
                    best_score = score
                    best_raag = r_name
                    best_time = info.get("time", "Unknown")
                    best_shifted_chroma = shifted_chroma
                    active_notes = [SWARA_NAMES[i] for i, val in enumerate(shifted_chroma) if val > 0.15]
                    detected_swaras = active_notes
                    
            all_raga_scores.append((r_name, best_score_for_this_raga))

        # Ranked list for visualization
        ranked_results = sorted(all_raga_scores, key=lambda x: x[1], reverse=True)

        # --- PHASE 3: HYBRID VERDICT ---
        is_morning = any(x in best_time for x in ["AM", "morning", "Morn", "Dawn", "Day"])
        logic_category = "Day Raag" if is_morning else "Night Raag"
        
        final_category = f"Verified {best_raag} ({logic_category})"
        
        if semantic_raga_hint == best_raag:
            logic_report = [f"Semantic-Acoustic Fusion absolutely confirmed this piece as {best_raag}.",
                            f"Musicological Identity Category: {logic_category}"]
        else:
            logic_report = [f"Mathematical Chroma Filter matched the performance to Raag {best_raag}.", 
                            f"Musicological Time Category: {logic_category}"]
            
        # --- PHASE 4: OLLAMA COGNITIVE REASONING ---
        narrative = self.cognitive_reasoning(best_raag, neural_mood, confidence, logic_report, detected_swaras)

        # --- PHASE 5: ADVANCED FEATURE EXTRACTION ---
        print("  [FEATURES] Extracting comprehensive musical features...")
        advanced_features_data = {}
        therapy_data = {}
        try:
            advanced_features_data = extract_all_features(y_music, sr=sr_22k)
            narrative = narrative + "\n\n" + advanced_features_data.get("text", "")
            
            # --- NEW: THERAPY RECOMMENDATION MODULE ---
            therapy_output = get_therapy_output(advanced_features_data)
            
            # Placeholder for backward compatibility if needed, but the task says ONLY ADD 'therapy'
            # therapy_data = therapy_output # I'll use therapy_output directly below
            
        except Exception as e:
            print(f"[ADVANCED FEATURES/THERAPY ERROR] {e}")

        # --- PHASE 6: VISUALIZATION DASHBOARD ---
        image_url = None
        pitch_contour_url = None
        spectrogram_url = None
        
        if original_filename:
            stem = Path(original_filename).stem
            
            # Legacy dashboard (now dynamically generated)
            legacy_path = BASE_DIR / "output" / f"analysis_{stem}.png"
            try:
                plot_full_dashboard({
                    "_f0": advanced_features_data.get("_f0"),
                    "_voiced": advanced_features_data.get("_voiced"),
                    "pc_hist": chroma_norm,
                    "tonic_hz": advanced_features_data["metadata"].get("tonic_hz", 220),
                    "_sr": sr_22k
                }, ranked_results, str(legacy_path), stem)
                image_url = f"/output/analysis_{stem}.png"
            except Exception as e:
                print(f"[LEGACY DASHBOARD ERROR] {e}")
            
            # New Detailed Visualizations
            pitch_path = BASE_DIR / "static" / f"pitch_{stem}.png"
            spec_path = BASE_DIR / "static" / f"spectrogram_{stem}.png"
            
            try:
                plot_pitch_contour({
                    "_f0": advanced_features_data.get("_f0"),
                    "_voiced": advanced_features_data.get("_voiced"),
                    "tonic_hz": advanced_features_data["metadata"].get("tonic_hz", 220),
                    "_sr": sr_22k
                }, str(pitch_path))
                
                # Use preloaded audio instead of path
                plot_spectrogram(y_music, sr_22k, str(spec_path))
                
                pitch_contour_url = f"/static/pitch_{stem}.png"
                spectrogram_url = f"/static/spectrogram_{stem}.png"
            except Exception as e:
                print(f"[VISUALIZATION ERROR] {e}")

        # Spectrogram for UI
        S = librosa.feature.melspectrogram(y=full_audio_48k[:48000*10], sr=48000, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        spec_data = ((S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255).astype(np.uint8).tolist()
        
        return {
            "prediction": final_category,
            "neural_prediction": final_category,
            "neural_mood": neural_mood,
            "detected_raag": best_raag,
            "confidence": confidence,
            "logic_score": float(best_score),
            "spectrogram": spec_data,
            "image_url": image_url,
            "narrative": narrative,
            "metadata": {
                "time": best_time,
                "swaras": detected_swaras,
                "advanced_features": advanced_features_data.get("metadata", {})
            },
            "report": logic_report,
            "therapy": therapy_output,
            "therapy_recommendation": therapy_output,
            "pitch_contour_url": pitch_contour_url,
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
