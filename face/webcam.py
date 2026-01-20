import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from face.face_matcher import cosine_similarity


# --------------------------------------------------
# TRACKER FACTORY
# --------------------------------------------------
def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("No OpenCV tracker available")


# --------------------------------------------------
# VIDEO FINDER
# --------------------------------------------------
class VideoFinder:
    def __init__(
        self,
        person_db,                  # dict[name -> embedding]
        threshold=0.35,
        detect_interval=5,
        iou_threshold=0.3,
    ):
        self.person_db = person_db
        self.threshold = threshold
        self.detect_interval = detect_interval
        self.iou_threshold = iou_threshold

        providers = ort.get_available_providers()
        self.ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))

        self.frame_count = 0
        self.trackers = []  # {tracker, bbox, score, matched, name}

    # --------------------------------------------------
    # BEST MATCH (FIXED)
    # --------------------------------------------------
    def _best_match(self, embedding):
        best_score = 0.0
        best_name = None

        for name, ref_emb in self.person_db.items():
            score = cosine_similarity(embedding, ref_emb)
            if score > best_score:
                best_score = score
                best_name = name

        return best_score, best_name

    # --------------------------------------------------
    # IOU
    # --------------------------------------------------
    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax + aw, bx + bw)
        inter_y2 = min(ay + ah, by + bh)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union = aw * ah + bw * bh - inter
        return inter / union

    # --------------------------------------------------
    # MAIN FRAME LOGIC
    # --------------------------------------------------
    def detect_frame(self, frame):
        self.frame_count += 1

        # ---------- UPDATE TRACKERS ----------
        active = []
        for t in self.trackers:
            ok, bbox = t["tracker"].update(frame)
            if not ok:
                continue

            x, y, bw, bh = map(int, bbox)
            t["bbox"] = (x, y, bw, bh)

            if t["matched"]:
                color = (0, 255, 0)
                label = f"{t['name']} {t['score']:.2f}"
            else:
                color = (0, 0, 255)
                label = f"NOT FOUND {t['score']:.2f}"

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            active.append(t)

        self.trackers = active

        # ---------- FACE DETECTION ----------
        if self.frame_count % self.detect_interval != 0:
            return frame

        faces = self.app.get(frame)

        for face in faces:
            score, name = self._best_match(face.embedding)

            x1, y1, x2, y2 = map(int, face.bbox)
            bbox = (x1, y1, x2 - x1, y2 - y1)

            # Avoid duplicate tracking
            if any(
                self._iou(bbox, t["bbox"]) > self.iou_threshold
                for t in self.trackers
            ):
                continue

            matched = score >= self.threshold

            tracker = create_tracker()
            tracker.init(frame, bbox)

            self.trackers.append({
                "tracker": tracker,
                "bbox": bbox,
                "score": score,
                "matched": matched,
                "name": name if matched else "Unknown",
            })

            # Draw immediately
            color = (0, 255, 0) if matched else (0, 0, 255)
            label = f"{name} {score:.2f}" if matched else f"NOT FOUND {score:.2f}"

            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                color,
                2,
            )
            cv2.putText(
                frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return frame
