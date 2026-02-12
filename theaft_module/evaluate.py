from ultralytics import YOLO
import torch

MODEL_PATH = "models/best.pt"

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data="data.yaml",
        imgsz=768,
        batch=8 if device != "cpu" else 4,
        device=device
    )

    print("\n========= EVALUATION RESULTS =========")
    print(f"Precision      : {metrics.box.mp:.4f}")
    print(f"Recall         : {metrics.box.mr:.4f}")
    print(f"mAP@0.50       : {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95  : {metrics.box.map:.4f}")
    print("=====================================\n")

if __name__ == "__main__":
    main()
