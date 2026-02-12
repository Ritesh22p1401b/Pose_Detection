import os
import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from data.dataset import AgeDataset
from trainer.transforms import train_transforms, val_transforms
from models.mobilenetv3_age import AgeMobileNetV3


# ================= CONFIG =================
DATASETS_DIR = "datasets"

IMAGES_ROOT = os.path.join(
    DATASETS_DIR,
    "imdb-clean-1024",
    "imdb-clean-1024"
)

TRAIN_CSV = os.path.join(DATASETS_DIR, "imdb_train_cleaned_1024.csv")
VAL_CSV   = os.path.join(DATASETS_DIR, "imdb_valid_cleaned_1024.csv")

CHECKPOINT_DIR = "checkpoints"

EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-4
# =========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("[BOOT]")
print("Python exec :", sys.executable)
print("CWD         :", os.getcwd())
print("Device      :", DEVICE)
print("TRAIN CSV   :", os.path.exists(TRAIN_CSV))
print("IMG ROOT    :", os.path.exists(IMAGES_ROOT))
print("=" * 60, flush=True)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def main():
    print("[STEP] Loading datasets...", flush=True)

    train_ds = AgeDataset(TRAIN_CSV, IMAGES_ROOT, train_transforms)
    val_ds   = AgeDataset(VAL_CSV, IMAGES_ROOT, val_transforms)

    print(f"[STEP] Train samples: {len(train_ds)}", flush=True)
    print(f"[STEP] Val samples  : {len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      # Windows-safe
        pin_memory=True
    )

    model = AgeMobileNetV3().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"[Epoch {epoch+1}] Avg Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "age_mobilenetv3_final.pth")
    print("[DONE] Training finished")


if __name__ == "__main__":
    main()
