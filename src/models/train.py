import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Fix imports when running directly
sys.path.append(os.path.dirname(__file__))

from dataset import SkeletonDataset
from model import GaitModel

# ---------------- CONFIG ----------------
SKELETON_DIR = "datasets/skeleton_train_norm/train"
GENDER_CSV = "datasets/gender_classification.csv"

CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "gait_model.pth")

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-3
SEQ_LEN = 100
GENDER_LOSS_WEIGHT = 0.3
# --------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Dataset
    dataset = SkeletonDataset(
        data_dir=SKELETON_DIR,
        gender_csv=GENDER_CSV,
        seq_len=SEQ_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # Infer dataset properties
    sample_x, _, _ = dataset[0]
    num_joints = sample_x.shape[1]
    num_people = len(dataset.person_ids)

    print(f"Skeleton joints: {num_joints}")
    print(f"Person classes: {num_people}")

    # Model
    model = GaitModel(
        num_joints=num_joints,
        num_people=num_people
    ).to(device)

    # Losses
    id_loss_fn = nn.CrossEntropyLoss()
    gender_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, pid, gender in pbar:
            x = x.to(device)
            pid = pid.to(device)
            gender = gender.to(device)

            _, id_logits, gender_logits = model(x)

            loss_id = id_loss_fn(id_logits, pid)
            loss_gender = gender_loss_fn(gender_logits, gender)

            # loss = loss_id + GENDER_LOSS_WEIGHT * loss_gender
            loss = loss_id

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} | Avg Loss: {epoch_loss / len(loader):.4f}")

    # Save model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_joints": num_joints,
            "num_people": num_people,
            "seq_len": SEQ_LEN,
        },
        MODEL_PATH
    )

    print("\nTraining complete.")
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()
