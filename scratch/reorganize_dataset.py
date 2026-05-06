import os
import shutil

root = r"d:\SUDHIR\PU\COMP\raga_classification"
dataset_dir = os.path.join(root, "dataset")
data_dir = os.path.join(root, "data")

ragas = ["Yaman", "Bhairav", "Bageshri", "Bihag", "Chandrakauns", "Malkauns"]

if not os.path.exists(data_dir):
    print(f"Creating {data_dir}...")
    os.makedirs(data_dir)
else:
    print(f"{data_dir} already exists.")

for raga in ragas:
    src = os.path.join(dataset_dir, raga)
    dst = os.path.join(data_dir, raga)
    
    if os.path.exists(src):
        print(f"Processing raga: {raga}")
        # Create dst if it doesn't exist
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        # Collect all .wav files from src recursively
        all_wavs = []
        for r, d, f in os.walk(src):
            for file in f:
                if file.lower().endswith(".wav"):
                    all_wavs.append(os.path.join(r, file))
        
        # Move all .wav files to dst root
        for wav_path in all_wavs:
            file_name = os.path.basename(wav_path)
            target_path = os.path.join(dst, file_name)
            
            # If target already exists, we might need a unique name but let's assume unique for now
            # or handle collision by prepending parent folder name if needed.
            # But the requirement is data/<raga>/<audio>.wav
            if os.path.exists(target_path) and wav_path != target_path:
                print(f"Warning: {target_path} already exists. Skipping or overwriting?")
                # Overwriting for now as it's likely the same file or a duplicate
                shutil.move(wav_path, target_path)
            else:
                shutil.move(wav_path, target_path)
        
        # Now remove the old src raga folder entirely
        print(f"Removing source folder {src}...")
        shutil.rmtree(src)
    else:
        print(f"Source raga folder {src} not found, checking if it's already in data...")
        if not os.path.exists(dst):
            print(f"Error: Raga {raga} not found in dataset or data.")

# Final cleanup of dst folders to ensure only .wav files and no subdirs
for raga in ragas:
    dst = os.path.join(data_dir, raga)
    if os.path.exists(dst):
        for item in os.listdir(dst):
            item_path = os.path.join(dst, item)
            if os.path.isdir(item_path):
                print(f"Removing nested folder: {item_path}")
                shutil.rmtree(item_path)
            elif not item.lower().endswith(".wav"):
                print(f"Removing non-wav file: {item_path}")
                os.remove(item_path)

# Verify and print
print("\nFinal Structure Verification:")
count = 0
for raga in ragas:
    raga_path = os.path.join(data_dir, raga)
    if os.path.exists(raga_path):
        files = os.listdir(raga_path)
        for file in files:
            if file.lower().endswith(".wav"):
                print(f"data/{raga}/{file}")
                count += 1
print(f"\nTotal .wav files moved: {count}")
