import os
import json

def load_raga_dataset(root_dir="data"):
    """
    Loads .wav files from raga subfolders in the root directory.
    Returns a dictionary where keys are raga labels and values are lists of file paths.
    """
    dataset = {}
    
    if not os.path.exists(root_dir):
        print(f"Error: Root folder '{root_dir}' not found.")
        return dataset

    # Get all subdirectories
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for raga_label in subdirs:
        raga_path = os.path.join(root_dir, raga_label)
        # Find all .wav files in this folder
        wav_files = [
            os.path.join(root_dir, raga_label, f).replace("\\", "/") 
            for f in os.listdir(raga_path) 
            if f.lower().endswith(".wav")
        ]
        
        if wav_files:
            dataset[raga_label] = wav_files

    # Print requested statistics
    print("---------------------------------")
    print(f"Total ragas found: {len(dataset)}")
    print("Files per raga:")
    for raga, files in dataset.items():
        print(f" - {raga}: {len(files)} files")
    print("---------------------------------")

    return dataset

if __name__ == "__main__":
    # If running from project root, use 'data'
    # The script is in 'backend/', so we might need to go up one level if called directly
    # but usually we run from project root.
    
    # Check if 'data' exists in current dir or parent
    target_data = "data"
    if not os.path.exists(target_data) and os.path.exists("../data"):
        target_data = "../data"
        
    data_dict = load_raga_dataset(target_data)
    
    # Verify return format
    # print("\nDataset Dictionary:")
    # print(json.dumps(data_dict, indent=4))
