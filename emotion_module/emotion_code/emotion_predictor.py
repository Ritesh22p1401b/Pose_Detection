import sys
import json
import cv2
from emotion_model import EmotionModel

if len(sys.argv) < 2:
    print(json.dumps({"error": "Image path required"}))
    sys.exit(1)

image = cv2.imread(sys.argv[1])
model = EmotionModel()
emotion, confidence = model.predict(image)

print(json.dumps({
    "emotion": emotion,
    "confidence": round(confidence, 4)
}))
