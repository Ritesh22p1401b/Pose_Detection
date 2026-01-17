import cv2
from insightface.app import FaceAnalysis
from face_matcher import cosine_similarity
import uuid


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
    raise RuntimeError("No tracker available")


class VideoFinder:
    def __init__(
        self,
        reference_embeddings,
        video_source=0,
        threshold=0.35,
        detect_interval=5,
        iou_threshold=0.3,
    ):
        self.reference_embeddings = reference_embeddings
        self.video_source = video_source
        self.threshold = threshold
        self.detect_interval = detect_interval
        self.iou_threshold = iou_threshold

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        self.frame_count = 0
        self.trackers = []  # list of dicts: {tracker, bbox, score}

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

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = aw * ah + bw * bh - inter_area
        return inter_area / union_area

    # ---------- Main ----------
    def detect_frame(self, frame):
        self.frame_count += 1
        found_any = False
        best_score = 0.0

        # ---- Update existing trackers ----
        updated_trackers = []
        for t in self.trackers:
            success, bbox = t["tracker"].update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                t["bbox"] = (x, y, w, h)

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                found_any = True
                best_score = max(best_score, t["score"])
                updated_trackers.append(t)

        self.trackers = updated_trackers

        # ---- Detection (every N frames) ----
        if self.frame_count % self.detect_interval == 0:
            faces = self.app.get(frame)

            for face in faces:
                score = self._best_similarity(face.embedding)
                if score < self.threshold:
                    continue

                x1, y1, x2, y2 = map(int, face.bbox)
                new_bbox = (x1, y1, x2 - x1, y2 - y1)

                # Check overlap with existing trackers
                is_duplicate = False
                for t in self.trackers:
                    if self._iou(new_bbox, t["bbox"]) > self.iou_threshold:
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                # Create new tracker
                tracker = create_tracker()
                tracker.init(frame, new_bbox)

                self.trackers.append({
                    "tracker": tracker,
                    "bbox": new_bbox,
                    "score": score,
                })

        return frame, found_any, best_score

