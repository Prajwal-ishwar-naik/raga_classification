import os
from utils.chunking import get_chunks

data_path = "data"

if not os.path.exists(data_path):
    print(f"Error: {data_path} directory not found.")
else:
    for raga in os.listdir(data_path):
        raga_path = os.path.join(data_path, raga)

        if not os.path.isdir(raga_path):
            continue

        print(f"--- Raga: {raga} ---")
        for file in os.listdir(raga_path):
            if file.endswith(".wav"):
                file_path = os.path.join(raga_path, file)

                print(f"Processing: {file_path}")

                chunks = get_chunks(file_path)

                print(f"{file} -> {len(chunks)} chunks\n")
