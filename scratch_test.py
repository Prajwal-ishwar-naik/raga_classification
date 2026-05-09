import requests
import sys

try:
    print("Sending POST request to /classify...")
    r = requests.post(
        'http://127.0.0.1:8000/classify', 
        files={'file': open('data/day_ragas/BhairavUP.opus', 'rb')}
    )
    print(f"Status Code: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Request failed: {e}")
