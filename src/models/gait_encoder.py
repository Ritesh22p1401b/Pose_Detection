import numpy as np
from scipy.signal import savgol_filter

class GaitEncoder:
    def encode(self, skeletons):
        """
        skeletons: (T, D) where D = joints*2
        returns: 128-D embedding
        """
        if skeletons is None or len(skeletons) < 10:
            return None

        # Smooth temporal noise
        skeletons = savgol_filter(skeletons, 7, 2, axis=0)

        # Normalize (hip centered)
        hip_center = skeletons[:, 0:2].mean(axis=0)
        skeletons -= np.tile(hip_center, skeletons.shape[1] // 2)

        # Temporal aggregation
        emb = np.concatenate([
            skeletons.mean(axis=0),
            skeletons.std(axis=0)
        ])

        # Pad / truncate to 128
        if emb.shape[0] < 128:
            emb = np.pad(emb, (0, 128 - emb.shape[0]))
        else:
            emb = emb[:128]

        return emb
