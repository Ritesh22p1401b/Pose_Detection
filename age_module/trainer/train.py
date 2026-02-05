import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data.dataset import AgeDataset
from trainer.transforms import train_transforms, val_transforms
from models.mobilenetv3_age import AgeMobileNetV3

# ================= CONFIG =================
CSV_PATH = "datasets/imdb_wiki/clean_imdb.csv"
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-4

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# =========================================


def save_checkpoint(epoch, model, optimizer, loss):
    path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss
    }, path)
    print(f"[CHECKPOINT] Saved: {path}")


def load_latest_checkpoint(model, optimizer):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if not checkpoints:
        return 0  # start from epoch 0

    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest_ckpt = checkpoints[-1]
    ckpt_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"[RESUME] Loaded checkpoint: {latest_ckpt}")
    print(f"[RESUME] Resuming from epoch {start_epoch}")

    return start_epoch


def main():
    # -------- Split CSV once --------
    if not os.path.exists("train.csv"):
        train_rows, val_rows = train_test_split(
            open(CSV_PATH).readlines()[1:], test_size=0.1
        )

        with open("train.csv", "w") as f:
            f.write("path,age\n" + "".join(train_rows))

        with open("val.csv", "w") as f:
            f.write("path,age\n" + "".join(val_rows))

    # -------- Datasets --------
    train_ds = AgeDataset("train.csv", train_transforms)
    val_ds = AgeDataset("val.csv", val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # -------- Model --------
    model = AgeMobileNetV3().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # -------- Resume if possible --------
    start_epoch = load_latest_checkpoint(model, optimizer)

    # -------- Training Loop --------
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

    # -------- Final model --------
    torch.save(model.state_dict(), "age_mobilenetv3_final.pth")
    print("[DONE] Final model saved: age_mobilenetv3_final.pth")


if __name__ == "__main__":
    main()
