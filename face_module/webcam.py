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
from gender_age_module.gender_age_client import GenderAgeClient


# --------------------------------------------------
# CONFIG (UNCHANGED)
# --------------------------------------------------
TRACKER_TTL = 60
DETECT_INTERVAL = 5
IOU_THRESHOLD = 0.3

EMOTION_UPDATE_INTERVAL = 3
MIN_FACE_SIZE = 32
EMOTION_STORE_CONF = 0.25

GENDER_AGE_MIN_FACE = 80   # ✅ REQUIRED FOR STABLE SINGLE-FRAME PREDICTION


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
    raise RuntimeError("opencv-contrib-python required")


def dominant_emotion(history):
    scores = defaultdict(float)
    for emo, weight in history:
        scores[emo] += weight
    return max(scores, key=scores.get) if scores else "Unknown"


# --------------------------------------------------
class VideoFinder:
    def __init__(self, person_db):
        self.person_db = person_db
        self.frame_count = 0
        self.trackers = []

        self.last_known_emotion = {}
        self.last_known_expression = defaultdict(lambda: "Unknown")  # ✅ SAFE
        self.last_known_gender_age = {}

        self.enable_emotion = True

        self.emotion_engine = EmotionAdapter()

        # --------------------------------------------------
        # SAFE Gender/Age initialization
        # --------------------------------------------------
        try:
            self.gender_age_engine = GenderAgeClient()
        except Exception as e:
            print("[VideoFinder] GenderAge disabled:", e)
            self.gender_age_engine = None

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
        best_score, best_name = 0.0, None
        for name, ref in self.person_db.items():
            score = cosine_similarity(embedding, ref)
            if score > best_score:
                best_score, best_name = score, name
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
    def detect_frame(self, frame):
        self.frame_count += 1
        active = []

        active_known_profiles = {
            t["name"] for t in self.trackers
            if t["matched"] and t["name"] != "Unknown"
        }

        emotion_allowed = (
            self.enable_emotion and
            len(active_known_profiles) == 1
        )

        for t in self.trackers:
            ok, bbox = t["tracker"].update(frame)
            t["ttl"] -= 1

            if not ok or t["ttl"] <= 0:
                final_emotion = dominant_emotion(t["emotion_history"])
                if final_emotion != "Unknown":
                    self.last_known_emotion[t["name"]] = final_emotion
                    self.last_known_expression[t["name"]] = final_emotion  # ✅ FIX
                continue

            x, y, w, h = map(int, bbox)
            t["bbox"] = (x, y, w, h)

            # ---------------- EMOTION ----------------
            t["emotion_counter"] += 1
            if emotion_allowed and t["emotion_counter"] % EMOTION_UPDATE_INTERVAL == 0:
                if min(w, h) >= MIN_FACE_SIZE:
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        emo, conf = self.emotion_engine.predict(roi)
                        if emo != "Unknown":
                            t["expression_now"] = emo
                        if conf >= EMOTION_STORE_CONF:
                            t["emotion_history"].append((emo, conf))

            # ---------------- GENDER / AGE (ONE FRAME ONLY) ----------------
            if (
                self.gender_age_engine
                and t["matched"]
                and t["name"] not in self.last_known_gender_age
                and min(w, h) >= GENDER_AGE_MIN_FACE  # ✅ CRITICAL FIX
            ):
                age, gender = self.gender_age_engine.predict(
                    frame[y:y+h, x:x+w]
                )

                if age is not None:
                    self.last_known_gender_age[t["name"]] = (age, gender)
                else:
                    # store once to avoid silent permanent failure
                    self.last_known_gender_age[t["name"]] = (None, "Unknown")

            if t["matched"]:
                label = f"{t['name']} | {t['expression_now']}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            active.append(t)

        self.trackers = active

        # ---------------- FACE DETECTION ----------------
        if self.frame_count % DETECT_INTERVAL == 0:
            for face in self.app.get(frame):
                x1, y1, x2, y2 = map(int, face.bbox)
                w, h = x2 - x1, y2 - y1

                if any(self.iou((x1, y1, w, h), t["bbox"]) > IOU_THRESHOLD
                       for t in self.trackers):
                    continue

                score, name = self.best_match(face.embedding)
                matched = score >= self.adaptive_threshold(min(w, h))

                tracker = create_tracker()
                tracker.init(frame, (x1, y1, w, h))

                self.trackers.append({
                    "tracker": tracker,
                    "bbox": (x1, y1, w, h),
                    "ttl": TRACKER_TTL,
                    "matched": matched,
                    "name": name if matched else "Unknown",
                    "expression_now": "Analyzing...",
                    "emotion_history": [],
                    "emotion_counter": 0
                })

        # ---------------- BOTTOM LEFT GENDER / AGE ----------------
        y_pos = frame.shape[0] - 20
        for name, (age, gender) in self.last_known_gender_age.items():
            age_txt = "--" if age is None else str(age)
            cv2.putText(
                frame,
                f"{name}: {gender}, {age_txt}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            y_pos -= 24

        return frame
