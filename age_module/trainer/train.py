import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data.dataset import AgeDataset
from trainer.transforms import train_transforms, val_transforms
from models.mobilenetv3_age import AgeMobileNetV3

CSV_PATH = "datasets/imdb_wiki/clean_imdb.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 64

def main():
    train_csv, val_csv = train_test_split(
        open(CSV_PATH).readlines()[1:], test_size=0.1
    )

    with open("train.csv", "w") as f:
        f.write("path,age\n" + "".join(train_csv))

    with open("val.csv", "w") as f:
        f.write("path,age\n" + "".join(val_csv))

    train_ds = AgeDataset("train.csv", train_transforms)
    val_ds = AgeDataset("val.csv", val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = AgeMobileNetV3().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "age_mobilenetv3.pth")
    print("[INFO] Model saved: age_mobilenetv3.pth")

if __name__ == "__main__":
    main()
