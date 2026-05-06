import librosa
import numpy as np
import os

def get_chunks(file_path):
    """
    Load audio with librosa (sr=22050), skip first 10 seconds,
    and split into 20-second chunks with a 10-second step (overlap).
    
    Args:
        file_path (str): Path to the .wav file.
        
    Returns:
        List[np.ndarray]: List of audio chunks as numpy arrays.
    """
    sr = 22050
    skip_sec = 10
    chunk_sec = 20
    step_sec = 10
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return []

    # Load audio
    # librosa.load 'offset' parameter skips the beginning of the file
    try:
        y, _ = librosa.load(file_path, sr=sr, offset=skip_sec)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

    # Calculate number of samples per chunk and step
    chunk_samples = int(chunk_sec * sr)
    step_samples = int(step_sec * sr)
    
    chunks = []
    
    # Extract chunks using a sliding window
    # We only take full 20-second chunks
    for i in range(0, len(y) - chunk_samples + 1, step_samples):
        chunk = y[i : i + chunk_samples]
        chunks.append(chunk)
        
    return chunks

if __name__ == "__main__":
    # Test with an existing file
    test_file = "data/Yaman/Yaman_vocal_01.wav"
    if os.path.exists(test_file):
        print(f"Testing get_chunks with: {test_file}")
        chunks = get_chunks(test_file)
        print(f"Total chunks extracted: {len(chunks)}")
        if chunks:
            print(f"Chunk 1 size: {chunks[0].shape} (Expected: {20 * 22050})")
            print(f"Chunk 2 size: {chunks[1].shape}")
            # Verify overlap: Chunk 2 starts 10 seconds in, so it should share half its data with Chunk 1
    else:
        print("Test file not found. Please ensure 'data/Yaman/Yaman_vocal_01.wav' exists.")
