# Backend Engine

This directory contains the core processing logic for the Raga Classification system.

## Components

- `server.py`: FastAPI server handling web requests.
- `neural_raga_engine.py`: Main hybrid engine combining CLAP neural embeddings with symbolic logic.
- `scholar_listener.py`: Raga database and symbolic transcription logic.
- `titan_engine.py`: Alternative DTW-based melodic matching engine.
- `advanced_features.py`: Extraction of musical features (swaras, arohana/avarohana, gamakas, etc.).
- `audacity_loader.py`: Utility to load Audacity project files.
- `run_raga.py`: CLI script for batch processing audio files.

## Usage

Run the server:
```bash
python server.py
```

Run batch analysis:
```bash
python run_raga.py
```
