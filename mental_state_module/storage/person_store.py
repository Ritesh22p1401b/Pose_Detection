import sqlite3
from pathlib import Path

DB_PATH = Path("mental_state_module/storage/database.sqlite")


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def get_all_persons():
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT person_id, first_emotion, final_verdict FROM person")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_emotions(person_id):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT emotion, confidence, timestamp
        FROM emotion_log
        WHERE person_id=?
        ORDER BY timestamp
    """, (person_id,))
    rows = cur.fetchall()
    conn.close()
    return rows
