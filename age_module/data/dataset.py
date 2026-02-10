import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from utils.age_bins import age_to_class


class AgeDataset(Dataset):
    def __init__(self, csv_file, images_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_root = images_root
        self.transform = transform

        # Your CSV uses `filename`
        if "filename" not in self.data.columns:
            raise ValueError(
                f"'filename' column not found. Columns: {self.data.columns.tolist()}"
            )

        if "age" not in self.data.columns:
            raise ValueError("'age' column not found in CSV")

        print("[DATASET] Using image column: 'filename'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.images_root, row["filename"])
        age = int(row["age"])

        label = age_to_class(age)

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
