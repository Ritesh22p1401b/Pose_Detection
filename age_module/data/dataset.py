import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from utils.age_bins import age_to_class


class AgeDataset(Dataset):
    def __init__(self, csv_file, images_root, transform=None):
        self.transform = transform
        df = pd.read_csv(csv_file)

        # Detect image column
        if "path" in df.columns:
            img_col = "path"
        elif "filename" in df.columns:
            img_col = "filename"
        else:
            raise ValueError(f"No image column found: {df.columns.tolist()}")

        if "age" not in df.columns:
            raise ValueError("'age' column missing in CSV")

        valid_rows = []

        for _, row in df.iterrows():
            rel_path = str(row[img_col]).lstrip("/\\")
            img_path = os.path.join(images_root, rel_path)

            if os.path.exists(img_path):
                r = row.copy()
                r["__img_path__"] = img_path
                valid_rows.append(r)

        if not valid_rows:
            raise RuntimeError(
                f"No valid images found.\n"
                f"images_root = {images_root}\n"
                f"example CSV path = {df.iloc[0][img_col]}"
            )

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

        print(f"[DATASET] CSV rows     : {len(df)}")
        print(f"[DATASET] Valid images : {len(self.data)}")
        print(f"[DATASET] Dropped      : {len(df) - len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image = Image.open(row["__img_path__"]).convert("RGB")
        image = np.array(image)

        age = int(row["age"])
        label = age_to_class(age)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
