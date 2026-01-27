import sys
import json
import cv2
from emotion_model import EmotionModel


def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Image path not provided"
        }))
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(json.dumps({
            "error": "Failed to read image"
        }))
        return

    model = EmotionModel()
    emotion, confidence = model.predict(image)

    print(json.dumps({
        "emotion": emotion,
        "confidence": round(confidence, 4)
    }))


if __name__ == "__main__":
    main()
