import sqlite3
from datetime import datetime
from pathlib import Path

from mental_state_module.core.emotion_buffer import EmotionBuffer
from mental_state_module.core.mental_analyzer import MentalAnalyzer

DB_PATH = Path("mental_state_module/storage/database.sqlite")
_BUFFERS = {}


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def ingest_emotion(person_id, emotion, confidence):
    if person_id not in _BUFFERS:
        _BUFFERS[person_id] = EmotionBuffer()

    buffer = _BUFFERS[person_id]
    buffer.add(emotion)

    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO emotion_log (person_id, emotion, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (person_id, emotion, confidence, datetime.utcnow().isoformat()))

    if buffer.first_emotion:
        cur.execute("""
            UPDATE person
            SET first_emotion=?
            WHERE person_id=? AND first_emotion IS NULL
        """, (buffer.first_emotion, person_id))

    conn.commit()
    conn.close()

    if buffer.is_ready():
        verdict = MentalAnalyzer().analyze(
            buffer.first_emotion,
            list(buffer.emotions)
        )

        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            UPDATE person SET final_verdict=?
            WHERE person_id=?
        """, (verdict, person_id))
        conn.commit()
        conn.close()

        return verdict

    return None
