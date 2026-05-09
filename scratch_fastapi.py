import sys
import os
import traceback
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath('backend'))
from backend.server import app

client = TestClient(app)

try:
    with open('data/day_ragas/BhairavUP.opus', 'rb') as f:
        response = client.post("/classify", files={"file": ("BhairavUP.opus", f, "audio/opus")})
    print(response.status_code)
    print(response.text)
except Exception as e:
    traceback.print_exc()
