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
from age_module.age_client import AgeClient
from theft_module.theft_client import TheftClient

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
TRACKER_TTL = 60
DETECT_INTERVAL = 5
IOU_THRESHOLD = 0.3

EMOTION_UPDATE_INTERVAL = 3
MIN_FACE_SIZE = 32
EMOTION_STORE_CONF = 0.25

GENDER_MIN_FACE = 45


# --------------------------------------------------
def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
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
        self.quick_verify_mode = (
            len(person_db) == 1 and "QuickPerson" in person_db
        )

        self.frame_count = 0
        self.trackers = []
        self.last_known_expression = defaultdict(lambda: "Unknown")

        self.enable_emotion = True
        self.emotion_engine = EmotionAdapter()

        try:
            self.gender_engine = GenderAgeClient()
        except Exception as e:
            print("[VideoFinder] Gender disabled:", e)
            self.gender_engine = None

        try:
            self.age_engine = AgeClient()
        except Exception as e:
            print("[Age Adapter] Age disabled:", e)
            self.age_engine = None

        try:
            self.theft_engine = TheftClient()
        except Exception as e:
            print("[Theft Adapter] Theft disabled:", e)
            self.theft_engine = None

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

        weapon_status = "nothing"

        for t in self.trackers:
            ok, bbox = t["tracker"].update(frame)
            t["ttl"] -= 1

            if not ok or t["ttl"] <= 0:
                final_emotion = dominant_emotion(t["emotion_history"])
                if final_emotion != "Unknown":
                    self.last_known_expression[t["name"]] = final_emotion
                continue

            x, y, w, h = map(int, bbox)
            fh, fw = frame.shape[:2]
            x, y = max(0, x), max(0, y)
            w, h = min(w, fw - x), min(h, fh - y)

            t["bbox"] = (x, y, w, h)
            face_size = min(w, h)

            # ---------------- EMOTION ----------------
            t["emotion_counter"] += 1
            if self.enable_emotion and t["emotion_counter"] % EMOTION_UPDATE_INTERVAL == 0:
                if face_size >= MIN_FACE_SIZE:
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        emo, conf = self.emotion_engine.predict(roi)
                        if emo != "Unknown":
                            t["expression_now"] = emo
                        if conf >= EMOTION_STORE_CONF:
                            t["emotion_history"].append((emo, conf))

            # ---------------- GENDER ----------------
            if self.gender_engine and face_size >= GENDER_MIN_FACE:
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    gender = self.gender_engine.predict(roi)
                    if gender != "Unknown":
                        t["gender"] = gender

            # ---------------- AGE ----------------
            if self.age_engine and face_size >= GENDER_MIN_FACE:
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    age = self.age_engine.predict(roi)
                    if age != "Unknown":
                        t["age"] = age

            # ---------------- WEAPON (UPPER BODY ROI) ----------------
            if self.theft_engine:
                expanded_y2 = min(frame.shape[0], y + h * 3)
                weapon_roi = frame[y:expanded_y2, x:x+w]

                if weapon_roi.size > 0:
                    try:
                        label = self.theft_engine.predict(weapon_roi)
                        if label != "nothing":
                            weapon_status = label
                    except Exception:
                        pass

            # ---------------- DRAW FACE BOX ----------------
            color = (0, 255, 0) if t["matched"] else (0, 0, 255)

            label_text = (
                f"{t['name']} | "
                f"{t.get('expression_now','--')} | "
                f"{t.get('gender','--')} | "
                f"{t.get('age','--')}"
            )

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            active.append(t)

        self.trackers = active

        # ---------------- FACE DETECTION ----------------
        if self.frame_count % DETECT_INTERVAL == 0:
            for face in self.app.get(frame):
                x1, y1, x2, y2 = map(int, face.bbox)
                w, h = x2 - x1, y2 - y1

                if any(self.iou((x1, y1, w, h), t["bbox"]) > IOU_THRESHOLD for t in self.trackers):
                    continue

                score, name = self.best_match(face.embedding)

                matched = True if self.quick_verify_mode else score >= self.adaptive_threshold(min(w, h))
                if self.quick_verify_mode:
                    name = "QuickPerson"

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
                    "emotion_counter": 0,
                    "gender": "--",
                    "age": "--"
                })

        # ---------------- WEAPON DISPLAY (BOTTOM LEFT ONLY) ----------------
        cv2.putText(frame,
                    f"Weapon: {weapon_status}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2)

        return frame
