import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.age_bins import age_to_class

class AgeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["path"]
        age = int(self.data.iloc[idx]["age"])
        label = age_to_class(age)

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
