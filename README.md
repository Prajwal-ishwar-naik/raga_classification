# Raga Classification System

A neuro-symbolic pipeline for Indian Classical Raga classification using Hybrid Intelligence.

## Project Structure

- `backend/`: Core logic, engines, and FastAPI server.
- `data/`: Audio datasets and annotations.
- `frontend/`: React-based user interface.
- `model/`: Experimental models and prototypes.
- `output/`: Generated analysis reports and visualizations.
- `static/`: Static assets for the web server.
- `uploads/`: Temporary storage for uploaded audio files.

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js & npm

### Installation

1. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

1. Start the backend server:
   ```bash
   cd backend
   python server.py
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```

## Documentation

Each major directory contains its own `README.md` with specific details.
