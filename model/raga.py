import librosa
import numpy as np
import crepe

# 1. Load Audio
audio, sr = librosa.load("E:\model\AlhaiyaBilawal.wav", sr=16000)

# 2. Extract Pitch (Using CREPE)
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

# 3. Define Tonic (Example: 140 Hz)
f_tonic = 140.0 

# 4. Convert to Cents relative to Tonic
# Filter out low-confidence segments first
cents = 1200 * np.log2(frequency / f_tonic)
cents[confidence < 0.5] = np.nan # Ignore background noise