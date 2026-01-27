import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from face.face_matcher import cosine_similarity

TRACKER_TTL = 60
DETECT_INTERVAL = 5
IOU_THRESHOLD = 0.3


def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("No tracker available")


class VideoFinder:
    def __init__(self, person_db):
        self.person_db = person_db
        self.frame_count = 0
        self.trackers = []

        providers = ort.get_available_providers()
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

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

    def detect_frame(self, frame):
        self.frame_count += 1

        active = []
        for t in self.trackers:
            ok, bbox = t["tracker"].update(frame)
            t["ttl"] -= 1
            if not ok or t["ttl"] <= 0:
                continue

            x, y, w, h = map(int, bbox)
            t["bbox"] = (x, y, w, h)

            color = (0, 255, 0) if t["matched"] else (0, 0, 255)
            label = f"{t['name']} {t['score']:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            active.append(t)

        self.trackers = active

        if self.frame_count % DETECT_INTERVAL != 0:
            return frame

        faces = self.app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            w, h = x2 - x1, y2 - y1
            size = min(w, h)

            threshold = self.adaptive_threshold(size)
            score, name = self.best_match(face.embedding)
            matched = score >= threshold

            if any(self.iou((x1, y1, w, h), t["bbox"]) > IOU_THRESHOLD
                   for t in self.trackers):
                continue

            tracker = create_tracker()
            tracker.init(frame, (x1, y1, w, h))

            self.trackers.append({
                "tracker": tracker,
                "bbox": (x1, y1, w, h),
                "ttl": TRACKER_TTL,
                "matched": matched,
                "name": name if matched else "Unknown",
                "score": score
            })

        return frame
