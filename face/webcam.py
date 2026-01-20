
# import cv2
# import onnxruntime as ort
# from insightface.app import FaceAnalysis
# from face.face_matcher import cosine_similarity

# TRACKER_TTL = 30  # frames


# def create_tracker():
#     if hasattr(cv2, "TrackerCSRT_create"):
#         return cv2.TrackerCSRT_create()
#     if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
#         return cv2.legacy.TrackerCSRT_create()
#     raise RuntimeError("No tracker available")


# class VideoFinder:
#     def __init__(self, person_db, threshold=0.35, detect_interval=5):
#         self.person_db = person_db
#         self.threshold = threshold
#         self.detect_interval = detect_interval

#         providers = ort.get_available_providers()
#         ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

#         self.app = FaceAnalysis(name="buffalo_l")
#         self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

#         self.frame_count = 0
#         self.trackers = []

#     def _best_match(self, emb):
#         best_score, best_name = 0.0, None
#         for name, ref in self.person_db.items():
#             s = cosine_similarity(emb, ref)
#             if s > best_score:
#                 best_score, best_name = s, name
#         return best_score, best_name

#     def detect_frame(self, frame):
#         self.frame_count += 1
#         active = []

#         for t in self.trackers:
#             ok, bbox = t["tracker"].update(frame)
#             t["ttl"] -= 1
#             if not ok or t["ttl"] <= 0:
#                 continue

#             x, y, w, h = map(int, bbox)
#             color = (0, 255, 0) if t["matched"] else (0, 0, 255)
#             label = f"{t['name']} {t['score']:.2f}" if t["matched"] else f"NOT FOUND {t['score']:.2f}"

#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             self._draw_confidence_bar(frame, x, y + h + 5, w, t["score"])

#             cv2.putText(frame, label, (x, y - 8),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#             active.append(t)

#         self.trackers = active

#         if self.frame_count % self.detect_interval != 0:
#             return frame

#         for face in self.app.get(frame):
#             score, name = self._best_match(face.embedding)
#             x1, y1, x2, y2 = map(int, face.bbox)

#             tracker = create_tracker()
#             tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

#             self.trackers.append({
#                 "tracker": tracker,
#                 "score": score,
#                 "matched": score >= self.threshold,
#                 "name": name if score >= self.threshold else "Unknown",
#                 "ttl": TRACKER_TTL
#             })

#         return frame

#     def _draw_confidence_bar(self, frame, x, y, w, score):
#         bar_w = int(w * min(score / self.threshold, 1.0))
#         cv2.rectangle(frame, (x, y), (x + w, y + 6), (50, 50, 50), -1)
#         cv2.rectangle(frame, (x, y), (x + bar_w, y + 6), (0, 255, 255), -1)


import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from face.face_matcher import cosine_similarity

TRACKER_TTL = 30  # frames


def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("No OpenCV tracker available")


class VideoFinder:
    def __init__(
        self,
        person_db,          # dict: name -> embedding
        threshold=0.35,
        detect_interval=5,
        iou_threshold=0.3,
    ):
        self.person_db = person_db
        self.threshold = threshold
        self.detect_interval = detect_interval
        self.iou_threshold = iou_threshold

        providers = ort.get_available_providers()
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        self.frame_count = 0
        self.trackers = []  # {tracker, bbox, score, matched, name, ttl}

    # --------------------------------------------------
    # MATCHING
    # --------------------------------------------------
    def _best_match(self, embedding):
        best_score = 0.0
        best_name = None
        for name, ref in self.person_db.items():
            s = cosine_similarity(embedding, ref)
            if s > best_score:
                best_score = s
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

    def _overlaps_any_tracker(self, bbox):
        return any(self._iou(bbox, t["bbox"]) > self.iou_threshold for t in self.trackers)

    def _overlaps_matched_tracker(self, bbox):
        return any(
            t["matched"] and self._iou(bbox, t["bbox"]) > self.iou_threshold
            for t in self.trackers
        )

    # --------------------------------------------------
    # MAIN
    # --------------------------------------------------
    def detect_frame(self, frame):
        self.frame_count += 1

        # ---------- UPDATE TRACKERS ----------
        active = []
        for t in self.trackers:
            ok, bbox = t["tracker"].update(frame)
            t["ttl"] -= 1
            if not ok or t["ttl"] <= 0:
                continue

            x, y, w, h = map(int, bbox)
            t["bbox"] = (x, y, w, h)

            if t["matched"]:
                color = (0, 255, 0)
                label = f"{t['name']} {t['score']:.2f}"
            else:
                color = (0, 0, 255)
                label = f"NOT FOUND {t['score']:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
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

            # If overlaps ANY tracker → skip
            if self._overlaps_any_tracker(bbox):
                continue

            # If overlaps matched tracker → never create red box
            if self._overlaps_matched_tracker(bbox):
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
                "ttl": TRACKER_TTL,
            })

        return frame
