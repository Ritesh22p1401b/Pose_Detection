import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from data.dataset import AgeDataset
from trainer.transforms import train_transforms, val_transforms
from models.mobilenetv3_age import AgeMobileNetV3

# ================= CONFIG =================
DATASETS_DIR = "datasets"

IMAGES_ROOT = os.path.join(DATASETS_DIR, "imdb-clean-1024")

TRAIN_CSV = os.path.join(DATASETS_DIR, "imdb_train_cleaned_1024.csv")
VAL_CSV   = os.path.join(DATASETS_DIR, "imdb_valid_cleaned_1024.csv")

CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-4
# =========================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(epoch, model, optimizer, loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth"),
    )


def load_latest_checkpoint(model, optimizer):
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if not ckpts:
        return 0

    ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest = ckpts[-1]

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, latest), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    print(f"[RESUME] Loaded {latest}")
    return ckpt["epoch"] + 1


def main():
    train_ds = AgeDataset(TRAIN_CSV, IMAGES_ROOT, train_transforms)
    val_ds   = AgeDataset(VAL_CSV, IMAGES_ROOT, val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    model = AgeMobileNetV3().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    start_epoch = load_latest_checkpoint(model, optimizer)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

        save_checkpoint(epoch, model, optimizer, avg_loss)

    torch.save(model.state_dict(), "age_mobilenetv3_final.pth")
    print("[DONE] Saved age_mobilenetv3_final.pth")


if __name__ == "__main__":
    main()
