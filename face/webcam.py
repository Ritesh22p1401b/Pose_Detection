import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from face.face_matcher import cosine_similarity


def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()
    if hasattr(cv2, "legacy"):
        if hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
        if hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
    raise RuntimeError("No OpenCV tracker available")


class VideoFinder:
    def __init__(
        self,
        reference_embeddings,
        threshold=0.35,
        detect_interval=5,
        iou_threshold=0.3,
    ):
        self.reference_embeddings = reference_embeddings
        self.threshold = threshold
        self.detect_interval = detect_interval
        self.iou_threshold = iou_threshold

        # -------- AUTO CPU / CUDA --------
        providers = ort.get_available_providers()
        self.ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))

        self.frame_count = 0
        self.trackers = []  # {tracker, bbox, score}

    # ---------- Utility ----------
    def _best_similarity(self, embedding):
        return max(
            cosine_similarity(embedding, ref)
            for ref in self.reference_embeddings
        )

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

    def _clamp_bbox(self, bbox, w, h):
        x, y, bw, bh = bbox
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        return x, y, bw, bh

    # ---------- Main ----------
    def detect_frame(self, frame):
        self.frame_count += 1
        found_any = False
        best_score = 0.0
        h, w = frame.shape[:2]

        # ---- Update trackers ----
        active = []
        for t in self.trackers:
            ok, bbox = t["tracker"].update(frame)
            if not ok:
                continue

            x, y, bw, bh = self._clamp_bbox(
                tuple(map(int, bbox)), w, h
            )

            t["bbox"] = (x, y, bw, bh)
            color = (0, 255, 0) if t["score"] >= self.threshold else (0, 0, 255)

            cv2.rectangle(
                frame, (x, y), (x + bw, y + bh), color, 2
            )

            found_any = True
            best_score = max(best_score, t["score"])
            active.append(t)

        self.trackers = active

        # ---- Detect faces ----
        if self.frame_count % self.detect_interval == 0:
            faces = self.app.get(frame)

            for face in faces:
                score = self._best_similarity(face.embedding)

                x1, y1, x2, y2 = map(int, face.bbox)
                new_bbox = self._clamp_bbox(
                    (x1, y1, x2 - x1, y2 - y1), w, h
                )

                # Draw RED box for unmatched face
                if score < self.threshold:
                    cv2.rectangle(
                        frame,
                        (new_bbox[0], new_bbox[1]),
                        (new_bbox[0] + new_bbox[2], new_bbox[1] + new_bbox[3]),
                        (0, 0, 255),
                        2,
                    )
                    continue

                # Avoid duplicates
                if any(
                    self._iou(new_bbox, t["bbox"]) > self.iou_threshold
                    for t in self.trackers
                ):
                    continue

                tracker = create_tracker()
                tracker.init(frame, new_bbox)

                self.trackers.append({
                    "tracker": tracker,
                    "bbox": new_bbox,
                    "score": score,
                })

        return frame, found_any, best_score
