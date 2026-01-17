import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    def __init__(self, data_dir, seq_len=100):
        self.data_dir = data_dir
        self.seq_len = seq_len

        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        )

        # Closed-set person IDs
        self.person_ids = {fname: idx for idx, fname in enumerate(self.files)}

    def __len__(self):
        return len(self.files)

    def _pad_or_truncate(self, x):
        T, J, C = x.shape

        if T > self.seq_len:
            return x[: self.seq_len]

        if T < self.seq_len:
            pad = np.zeros((self.seq_len - T, J, C), dtype=np.float32)
            return np.concatenate([x, pad], axis=0)

        return x

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.data_dir, fname)

        x = np.load(path).astype(np.float32)
        x = self._pad_or_truncate(x)
        x = torch.from_numpy(x)

        person_label = torch.tensor(
            self.person_ids[fname], dtype=torch.long
        )

        # Gender temporarily disabled (no valid mapping)
        gender_label = torch.tensor(0, dtype=torch.long)

        return x, person_label, gender_label
