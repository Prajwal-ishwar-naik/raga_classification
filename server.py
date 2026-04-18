from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import uuid
from neural_raga_engine import HybridRagaVision
from audacity_loader import load_audacity_project

# Initialize the Hybrid Neural-Symbolic Engine
from neural_raga_engine import HybridRagaVision
neural_engine = HybridRagaVision()

app = FastAPI(title="Raga Vision - Hybrid Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/")
def read_root():
    return {"message": "Neural Raga API is running"}

@app.post("/classify_bulk")
def classify_bulk(files: List[UploadFile] = File(...)):
    print(f"[SERVER] Received bulk request for {len(files)} files")
    results = []
    
    for file in files:
        filename_lower = file.filename.lower()
        allowed_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.aup', '.ogg', '.opus')
        
        if not filename_lower.endswith(allowed_extensions):
            continue

        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Neural Inference - Ultra Fast Semantic-Acoustic Fusion
            res = neural_engine.analyze(temp_path, duration=8, original_filename=file.filename) 
            
            # Clean up immediately for bulk
            if os.path.exists(temp_path): os.remove(temp_path)
            
            results.append({
                "filename": file.filename,
                "prediction": res["prediction"],
                "confidence": res["confidence"],
                "narrative": res["narrative"],
                "spectrogram": res.get("spectrogram")
            })
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            if os.path.exists(temp_path): os.remove(temp_path)
            results.append({
                "filename": file.filename,
                "prediction": "Analysis Failed",
                "confidence": 0,
                "narrative": f"Error: {str(e)}"
            })

    return {"results": results}

@app.post("/classify")
def classify_audio(file: UploadFile = File(...)):
    print(f"[SERVER] Received classification request for: {file.filename}")
    # Supported common audio formats + Audacity Project
    filename_lower = file.filename.lower()
    allowed_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.aup', '.ogg', '.opus', '.m4a')
    
    is_aup = filename_lower.endswith('.aup')
    if not filename_lower.endswith(allowed_extensions):
        print(f"[REJECTED] Unsupported format: {file.filename}")
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {os.path.splitext(file.filename)[1]}")
    
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    try:
        if is_aup:
            # Special handling for local Audacity projects
            # Since we can't upload whole directories easily, 
            # we check if this is a known file in our workspace first
            # OR we expect the user to have the data folder relative to server
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Note: audacity_loader needs the _data folder. 
            # For this demo, we assume the data folder is accessible if uploaded from workspace
            # We'll try to load it; if it fails due to missing data, we'll tell the user.
            try:
                # We need to preserve the filename for the loader to find the _data folder
                # audacity_loader uses Path(aup_path).parent / (Path(aup_path).stem + "_data")
                result = neural_engine.analyze(temp_path, original_filename=file.filename)
            except Exception:
                # Fallback: if it's a known raga in the local folder, use it
                # This is a bit of a hack for the local environment
                local_path = f"Day Raga/{file.filename}"
                if os.path.exists(local_path):
                    result = neural_engine.analyze(local_path, original_filename=file.filename)
                else:
                    raise
        else:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            result = neural_engine.analyze(temp_path, original_filename=file.filename)
        
        return {
            "prediction": result["prediction"], 
            "neural_prediction": result["prediction"],
            "neural_confidence": result["confidence"],
            "detected_raag": result["detected_raag"],
            "logic_score": result["logic_score"],
            "neural_mood": result["neural_mood"],
            "metadata": result["metadata"],
            "report": result["report"],
            "narrative": result["narrative"],
            "spectrogram": result["spectrogram"]
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Neural Inference Failed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
