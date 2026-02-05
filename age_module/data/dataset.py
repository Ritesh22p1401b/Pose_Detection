import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from utils.age_bins import age_to_class


class AgeDataset(Dataset):
    def __init__(self, csv_file: str, images_root: str, transform=None):
        """
        csv_file    : datasets/imdb_train_new_1024.csv
        images_root : datasets/imdb-clean-1024
        """
        self.data = pd.read_csv(csv_file)
        self.images_root = images_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # CSV contains: 00/xxxx.jpg
        img_path = os.path.join(self.images_root, row["path"])
        age = int(row["age"])

        label = age_to_class(age)

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
