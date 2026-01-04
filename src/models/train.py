import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from dataset import SkeletonDataset
from model import GaitModel

# ---------------- CONFIG ----------------
SKELETON_DIR = "datasets/skeleton_train_norm/train"
CHECKPOINT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "gait_checkpoint.pth")

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-3
SEQ_LEN = 100
# --------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = SkeletonDataset(
        data_dir=SKELETON_DIR,
        seq_len=SEQ_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    sample_x, _, _ = dataset[0]
    num_joints = sample_x.shape[1]
    num_people = len(dataset.person_ids)

    model = GaitModel(num_joints, num_people).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0

    # -------- RESUME IF CHECKPOINT EXISTS --------
    if os.path.exists(CKPT_PATH):
        print("Resuming from checkpoint...")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # -------- TRAINING LOOP --------
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, pid, _ in pbar:
            x = x.to(device)
            pid = pid.to(device)

            _, id_logits, _ = model(x)
            loss = loss_fn(id_logits, pid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} | Avg Loss: {epoch_loss / len(loader):.4f}")

        # -------- SAVE CHECKPOINT --------
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "num_joints": num_joints,
                "num_people": num_people,
            },
            CKPT_PATH
        )

        print(f"Checkpoint saved at epoch {epoch+1}")

    # -------- FINAL MODEL SAVE --------
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "gait_model.pth"))
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    main()
