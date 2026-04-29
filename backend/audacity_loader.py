import os
import xml.etree.ElementTree as ET
import numpy as np
import librosa
from pathlib import Path

def load_audacity_project(aup_path):
    """
    Parses an Audacity .aup file and loads the audio data from the _data directory.
    Supports Audacity 1.x/2.x project format.
    """
    aup_path = Path(aup_path)
    if not aup_path.exists():
        raise FileNotFoundError(f"Project file not found: {aup_path}")

    # Use a simpler approach: many .aup files use a namespace that makes findall tricky
    tree = ET.parse(aup_path)
    root = tree.getroot()
    
    # Audacity version check or namespace handling
    ns = ""
    if '}' in root.tag:
        ns = root.tag.split('}')[0] + '}'

    proj_name = root.attrib.get('projname')
    if not proj_name:
        proj_name = aup_path.stem + "_data"
        
    data_dir = aup_path.parent / proj_name
    if not data_dir.exists():
        # Try fallback: some projects don't have "_data" in the projname attribute
        data_dir = aup_path.parent / (aup_path.stem + "_data")
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # For Raga classification, we usually only care about the first mono track
    # or we mixdown. For simplicity, we'll take the first 'wavetrack'.
    wavetrack = root.find(f'.//{ns}wavetrack')
    if wavetrack is None:
        raise ValueError("No audio tracks found in project.")

    rate = float(wavetrack.attrib.get('rate', 44100))
    all_chunks = []

    # Iterate through clips and sequences
    for sequence in wavetrack.findall(f'.//{ns}sequence'):
        for waveblock in sequence.findall(f'.//{ns}waveblock'):
            file_info = waveblock.find(f'.//{ns}simpleblockfile')
            if file_info is not None:
                filename = file_info.attrib.get('filename')
                # Filename pattern: e0000abc.au
                # Path: e00/d00/e0000abc.au (first 3 chars determine folder, then d + 4th/5th char?)
                # Actually, standard Audacity 2.x is:
                # {data_dir}/{dXX}/{dYY}/{filename}
                # Where dXX is e00, dYY is d00, d01 etc.
                
                # Let's use a robust search because naming can vary slightly
                # Format is usually: e{2hex}/d{2hex}/e{8hex}.au
                # But we can just search for it or use the standard rule.
                
                # Rule: e{first 2 indices of 0000 part}/d{next 2}/filename
                # Filename: e 00 00 abc .au
                # Indices:  0 12 34 
                folder1 = filename[0:3] # e00
                folder2 = "d" + filename[3:5] # d00
                
                block_path = data_dir / folder1 / folder2 / filename
                
                if block_path.exists():
                    try:
                        y, _ = librosa.load(str(block_path), sr=None)
                        all_chunks.append(y)
                    except Exception as e:
                        print(f"Error loading block {filename}: {e}")
                else:
                    # Fallback recursive search if directory structure unexpected
                    matches = list(data_dir.rglob(filename))
                    if matches:
                        y, _ = librosa.load(str(matches[0]), sr=None)
                        all_chunks.append(y)
                    else:
                        print(f"Warning: block file not found: {filename}")

    if not all_chunks:
        return None, 0

    return np.concatenate(all_chunks), rate

if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        y, sr = load_audacity_project(sys.argv[1])
        if y is not None:
            print(f"Loaded {len(y)} samples @ {sr}Hz")
