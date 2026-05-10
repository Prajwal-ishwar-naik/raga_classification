
import os
import sys
from pathlib import Path

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_engine import RagaChatEngine
    print("Import successful")
    engine = RagaChatEngine()
    print("Initialization successful")
    print(f"Chunk count: {engine.get_chunk_count()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
