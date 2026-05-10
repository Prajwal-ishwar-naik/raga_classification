import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "feedback.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_raga TEXT,
            correct_raga TEXT,
            timestamp DATETIME
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(filename, predicted, correct):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedback (filename, predicted_raga, correct_raga, timestamp) VALUES (?, ?, ?, ?)",
        (filename, predicted, correct, datetime.now())
    )
    conn.commit()
    conn.close()
    return True

# Initialize on import
init_db()
