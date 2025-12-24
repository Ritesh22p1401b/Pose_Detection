from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model = YOLO(model_path)

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            conf=0.4,
            classes=[0],   # person only
            verbose=False
        )
        return results[0]
