import numpy as np
from scipy.signal import savgol_filter

class GaitEncoder:
    def encode(self, skeletons):
        if skeletons is None or len(skeletons) < 20:
            return None

        skeletons = savgol_filter(skeletons, 7, 2, axis=0)

        mean = skeletons.mean(axis=0)
        std = skeletons.std(axis=0)

        emb = np.concatenate([mean, std])

        if emb.shape[0] < 128:
            emb = np.pad(emb, (0, 128 - emb.shape[0]))
        else:
            emb = emb[:128]

        return emb
