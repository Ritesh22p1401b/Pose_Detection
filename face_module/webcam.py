import sys
import os
from collections import defaultdict

# --------------------------------------------------
# ADD PROJECT ROOT TO sys.path
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis

from face_matcher import cosine_similarity
from emotion_module.emotion_link.emotion_adapter import EmotionAdapter


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
TRACKER_TTL = 60
DETECT_INTERVAL = 5
IOU_THRESHOLD = 0.3

EMOTION_UPDATE_INTERVAL = 12
EMOTION_HISTORY_SIZE = 15

MIN_FACE_SIZE = 48   # DO NOT LOWER (hard safety floor)


# --------------------------------------------------
# TRACKER FACTORY
# --------------------------------------------------
def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()

    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()

    raise RuntimeError(
        "No OpenCV tracker found. Install opencv-contrib-python in face venv."
    )


# --------------------------------------------------
# ADAPTIVE CONFIDENCE (0.20 → 0.45)
# --------------------------------------------------
def adaptive_emotion_confidence(face_size):
    """
    Dynamic confidence threshold based on face size (distance).
    Returns None if emotion should be skipped.
    """
    if face_size >= 120:
        return 0.45
    elif face_size >= 100:
        return 0.40
    elif face_size >= 80:
        return 0.35
    elif face_size >= 60:
        return 0.25
    elif face_size >= MIN_FACE_SIZE:
        return 0.20
    else:
        return None  # too far, do not attempt emotion


# --------------------------------------------------
# TEMPORAL EMOTION AGGREGATION
# --------------------------------------------------
def dominant_emotion(history):
    scores = defaultdict(float)

    for emo, weight in history:
        scores[emo] += weight

    if not scores:
        return "Unknown"

    return max(scores, key=scores.get)


# --------------------------------------------------
# VIDEO FINDER
# --------------------------------------------------
class VideoFinder:
    def __init__(self, person_db):
        self.person_db = person_db
        self.frame_count = 0
        self.trackers = []

        self.emotion_engine = EmotionAdapter()

        providers = ort.get_available_providers()
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    # --------------------------------------------------
    def adaptive_threshold(self, face_size):
        if face_size < 55:
            return 0.19
        elif face_size < 75:
            return 0.22
        elif face_size < 95:
            return 0.25
        elif face_size < 130:
            return 0.35
        else:
            return 0.45

    def best_match(self, embedding):
        best_score = 0.0
        best_name = None

        for name, ref in self.person_db.items():
            score = cosine_similarity(embedding, ref)
            if score > best_score:
                best_score = score
                best_name = name

        return best_score, best_name

    def iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        union = aw * ah + bw * bh - inter

        return inter / union

    # --------------------------------------------------
    # MAIN VIDEO LOOP
    # --------------------------------------------------
    def detect_frame(self, frame):
        self.frame_count += 1
        active = []

        # -------- TRACK EXISTING --------
        for t in self.trackers:
            ok, bbox = t["tracker"].update(frame)
            t["ttl"] -= 1

            if not ok or t["ttl"] <= 0:
                continue

            x, y, w, h = map(int, bbox)
            t["bbox"] = (x, y, w, h)

            # -------- EMOTION (VIDEO-BASED) --------
            t["emotion_counter"] += 1

            if t["emotion_counter"] % EMOTION_UPDATE_INTERVAL == 0:
                face_size = min(w, h)
                conf_thresh = adaptive_emotion_confidence(face_size)

                if conf_thresh is not None:
                    roi = frame[y:y+h, x:x+w]

                    if roi.size >= MIN_FACE_SIZE * MIN_FACE_SIZE:
                        emotion, conf = self.emotion_engine.predict(roi)

                        if emotion != "Unknown" and conf >= conf_thresh:
                            # weight emotion by confidence and face size
                            weight = conf * (face_size / 120.0)
                            t["emotion_history"].append((emotion, weight))

                            # sliding window
                            t["emotion_history"] = t["emotion_history"][-EMOTION_HISTORY_SIZE:]

                            # final emotion
                            t["emotion"] = dominant_emotion(t["emotion_history"])

            color = (0, 255, 0) if t["matched"] else (0, 0, 255)
            label = f"{t['name']} | {t['emotion']}"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            active.append(t)

        self.trackers = active

        # -------- DETECT NEW FACES --------
        if self.frame_count % DETECT_INTERVAL != 0:
            return frame

        for face in self.app.get(frame):
            x1, y1, x2, y2 = map(int, face.bbox)
            w, h = x2 - x1, y2 - y1
            size = min(w, h)

            if any(self.iou((x1, y1, w, h), t["bbox"]) > IOU_THRESHOLD
                   for t in self.trackers):
                continue

            score, name = self.best_match(face.embedding)
            matched = score >= self.adaptive_threshold(size)

            tracker = create_tracker()
            tracker.init(frame, (x1, y1, w, h))

            self.trackers.append({
                "tracker": tracker,
                "bbox": (x1, y1, w, h),
                "ttl": TRACKER_TTL,
                "matched": matched,
                "name": name if matched else "Unknown",
                "emotion": "Detecting...",
                "emotion_history": [],
                "emotion_counter": 0
            })

        return frame
