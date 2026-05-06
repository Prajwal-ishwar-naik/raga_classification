import librosa
import numpy as np
import os

def get_chunks(file_path, sr=22050, chunk_size=20, step_size=10):
    """
    Loads audio, skips the first 10 seconds, and extracts overlapping chunks.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        chunk_size (int): Duration of each chunk in seconds.
        step_size (int): Step size (overlap) in seconds.
        
    Returns:
        List[np.ndarray]: List of audio chunks.
    """
    try:
        # Load audio (sr=22050)
        # We load the whole file first to calculate duration easily, 
        # but skip the first 10 seconds as requested.
        # librosa.load 'offset' parameter skips the beginning.
        audio, _ = librosa.load(file_path, sr=sr, offset=10)
        
        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"Loaded: {os.path.basename(file_path)} | Total duration (after skip): {duration:.2f}s")
        
        # Convert seconds to samples
        chunk_samples = int(chunk_size * sr)
        step_samples = int(step_size * sr)
        
        chunks = []
        
        # If audio is too short for even one chunk
        if len(audio) < chunk_samples:
            print(f"Warning: Audio too short for chunking. Samples: {len(audio)}")
            return []
            
        # Sliding window
        start = 0
        while start + chunk_samples <= len(audio):
            chunk = audio[start : start + chunk_samples]
            chunks.append(chunk)
            start += step_samples
            
        print(f"Successfully created {len(chunks)} chunks.")
        return chunks
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

if __name__ == "__main__":
    # Internal test
    pass
