from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import uuid
from pathlib import Path
from neural_raga_engine import HybridRagaVision
from audacity_loader import load_audacity_project
from pdf_generator import generate_report_pdf
from feedback import save_feedback
from rag_engine import RagaChatEngine

class FeedbackRequest(BaseModel):
    filename: str
    predicted_raga: str
    correct_raga: str

class ChatRequest(BaseModel):
    question: str
    filename: Optional[str] = None

class IndexPDFRequest(BaseModel):
    filename: str


# Resolve project root (one level above backend/)
BASE_DIR = Path(__file__).parent.parent

# Initialize the Hybrid Neural-Symbolic Engine
neural_engine = HybridRagaVision()

# Initialize the RAG Chatbot Engine
rag_engine = RagaChatEngine()
rag_engine.index_raga_knowledge()  # Index music theory on startup

app = FastAPI(title="Raga Vision - Hybrid Intelligence")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/output", StaticFiles(directory=str(BASE_DIR / "output")), name="output")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = str(BASE_DIR / "uploads")
STATIC_DIR = str(BASE_DIR / "static")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)


@app.get("/")
def read_root():
    return {"message": "Neural Raga API is running"}


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    try:
        save_feedback(request.filename, request.predicted_raga, request.correct_raga)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================= RAG CHATBOT ENDPOINTS =======================

@app.post("/index_pdf")
def index_pdf(request: IndexPDFRequest):
    """Index a generated PDF report + images into the vector store."""
    filename = request.filename
    stem = Path(filename).stem
    pdf_path = str(BASE_DIR / "static" / f"report_{stem}.pdf")

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF report not found. Please download the PDF first.")

    # Find related images
    image_paths = []
    for prefix in ["spec_", "dash_"]:
        img_path = str(BASE_DIR / "static" / f"{prefix}{stem}.png")
        if os.path.exists(img_path):
            image_paths.append(img_path)

    result = rag_engine.index_pdf(pdf_path, filename=stem, image_paths=image_paths)
    return result

@app.post("/chat")
def chat(request: ChatRequest):
    """Query the RAG chatbot with a question about the analysis."""
    result = rag_engine.query(request.question, filename=request.filename)
    return result

@app.get("/rag_status")
def rag_status():
    """Check RAG engine status and chunk count."""
    return {
        "status": "active",
        "total_chunks": rag_engine.get_chunk_count()
    }

@app.post("/classify_bulk")
def classify_bulk(files: List[UploadFile] = File(...), lang: str = "en"):
    print(f"[SERVER] Received bulk request for {len(files)} files in language {lang}")
    results = []

    for file in files:
        filename_lower = file.filename.lower()
        allowed_extensions = (".wav", ".mp3", ".m4a", ".flac", ".aup", ".ogg", ".opus")

        if not filename_lower.endswith(allowed_extensions):
            continue

        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Neural Inference - Ultra Fast Semantic-Acoustic Fusion
            res = neural_engine.analyze(
                temp_path, duration=8, original_filename=file.filename, file_id=file_id, lang=lang
            )

            # Clean up immediately for bulk
            if os.path.exists(temp_path):
                os.remove(temp_path)

            formatted_pred = f"{res['prediction']} Raga"
            results.append(
                {
                    "filename": file.filename,
                    "prediction": formatted_pred,
                    "confidence": res["confidence"],
                    "narrative": res["narrative"],
                    "spectrogram": res.get("spectrogram"),
                    "detailed_features": res.get("detailed_features"),
                    "pitch_contour_data": res.get("pitch_contour_data", []),
                    "swara_distribution_data": res.get("swara_distribution_data", {}),
                    "image_url": res.get("image_url"),
                    "therapy_recommendation": res.get("therapy"),
                    "therapy": res.get("therapy"),
                }
            )
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            results.append(
                {
                    "filename": file.filename,
                    "prediction": "Analysis Failed",
                    "confidence": 0,
                    "narrative": f"Error: {str(e)}",
                    "metadata": {"swaras": []},
                    "therapy": {
                        "recommendation": {"primary": "N/A", "secondary": []},
                        "therapy_scores": {"calm_score": 0, "energy_score": 0, "focus_score": 0},
                        "explanation": [],
                        "session_plan": [],
                        "raga_metadata": None
                    },
                    "report": [f"Technical Error: {str(e)}"],
                    "pitch_contour_data": [],
                    "swara_distribution_data": {},
                }
            )

    return {"results": results}


# Add Request model for PDF generation
class PDFRequest(BaseModel):
    data: Dict[str, Any]

@app.post("/download_pdf")
async def download_pdf(request: PDFRequest):
    try:
        data = request.data
        filename = data.get("filename", "report")
        stem = Path(filename).stem
        # Use BASE_DIR to ensure we point to the correct static folder
        pdf_path = BASE_DIR / "static" / f"report_{stem}.pdf"
        
        generate_report_pdf(data, str(pdf_path))
        
        return FileResponse(
            str(pdf_path), 
            media_type="application/pdf", 
            filename=f"RagaVision_Report_{stem}.pdf"
        )
    except Exception as e:
        print(f"PDF Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.post("/classify")
def classify_audio(file: UploadFile = File(...), lang: str = "en"):
    print(f"[SERVER] Received classification request for: {file.filename} in language {lang}")
    filename_lower = file.filename.lower()
    allowed_extensions = (
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".aup",
        ".ogg",
        ".opus",
        ".m4a",
    )

    is_aup = filename_lower.endswith(".aup")
    if not filename_lower.endswith(allowed_extensions):
        print(f"[REJECTED] Unsupported format: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {os.path.splitext(file.filename)[1]}",
        )

    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

    try:
        if is_aup:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            try:
                result = neural_engine.analyze(
                    temp_path, original_filename=file.filename, file_id=file_id, lang=lang
                )
            except Exception:
                # Fallback: check local day_ragas folder
                local_path = str(BASE_DIR / "data" / "day_ragas" / file.filename)
                if os.path.exists(local_path):
                    result = neural_engine.analyze(
                        local_path, original_filename=file.filename, file_id=file_id, lang=lang
                    )
                else:
                    raise
        else:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            result = neural_engine.analyze(
                temp_path, original_filename=file.filename, file_id=file_id, lang=lang
            )

        formatted_pred = f"{result['prediction']} Raga"

        # Auto-generate and index PDF for RAG chatbot
        try:
            stem = Path(file.filename).stem
            pdf_path = str(BASE_DIR / "static" / f"report_{stem}.pdf")
            full_result = {
                "filename": file.filename,
                "prediction": formatted_pred,
                "neural_mood": result["neural_mood"],
                "confidence": result["confidence"],
                "metadata": result["metadata"],
                "report": result["report"],
                "narrative": result["narrative"],
                "detailed_features": result.get("detailed_features"),
                "image_url": result.get("image_url"),
                "therapy": result.get("therapy"),
            }
            generate_report_pdf(full_result, pdf_path)
            # Auto-index into RAG
            image_paths = []
            for prefix in ["spec_", "dash_"]:
                img = str(BASE_DIR / "static" / f"{prefix}{stem}.png")
                if os.path.exists(img):
                    image_paths.append(img)
            rag_engine.index_pdf(pdf_path, filename=stem, image_paths=image_paths)
        except Exception as e:
            print(f"[RAG AUTO-INDEX] Non-critical error: {e}")

        return {
            "prediction": formatted_pred,
            "neural_prediction": formatted_pred,
            "neural_confidence": result["confidence"],
            "detected_raag": formatted_pred,
            "filename": file.filename,
            "logic_score": result["logic_score"],
            "neural_mood": result["neural_mood"],
            "metadata": result["metadata"],
            "report": result["report"],
            "narrative": result["narrative"],
            "spectrogram": result["spectrogram"],
            "detailed_features": result.get("detailed_features"),
            "image_url": result.get("image_url"),
            "pitch_contour_data": result.get("pitch_contour_data", []),
            "swara_distribution_data": result.get("swara_distribution_data", {}),
            "therapy_recommendation": result.get("therapy"),
            "therapy": result.get("therapy"),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Neural Inference Failed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
