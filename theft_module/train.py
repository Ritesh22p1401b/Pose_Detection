import os
import torch
from ultralytics import YOLO

PROJECT_DIR = "models"
MODEL_NAME = "gun_knife_detector"
WEIGHTS_DIR = f"{PROJECT_DIR}/{MODEL_NAME}/weights"
LAST_WEIGHTS = f"{WEIGHTS_DIR}/last.pt"

def main():
    # 🔹 Auto device selection
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {'GPU' if device == 0 else 'CPU'}")

    # 🔹 Ensure directories exist
    os.makedirs(PROJECT_DIR, exist_ok=True)

    # 🔹 Resume if training was interrupted
    if os.path.exists(LAST_WEIGHTS):
        print("[INFO] Resuming training from last checkpoint...")
        model = YOLO(LAST_WEIGHTS)
        resume = True
    else:
        print("[INFO] Starting fresh training...")
        model = YOLO("yolov8s.pt")
        resume = False

    # 🔹 Train (accuracy-focused config)
    model.train(
        data="data.yaml",
        epochs=100,                 # 🔑 train longer
        imgsz=768,                  # 🔑 better for knives
        batch=8 if device != "cpu" else 4,
        device=device,
        workers=4,
        optimizer="AdamW",
        lr0=0.0008,
        patience=20,
        save=True,
        save_period=1,              # save every epoch
        resume=resume,
        project=PROJECT_DIR,
        name=MODEL_NAME
    )

if __name__ == "__main__":
    main()
