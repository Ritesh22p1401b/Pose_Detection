import numpy as np

class GaitEncoder:
    def encode(self, skeleton_sequence):
        """
        skeleton_sequence: (T, D)
        returns: 128-D embedding
        """
        if len(skeleton_sequence) == 0:
            return None

        # Normalize
        skeleton_sequence -= skeleton_sequence.mean(axis=0)

        # Simple temporal aggregation (CPU-friendly)
        embedding = np.mean(skeleton_sequence, axis=0)

        # Pad / truncate to 128D
        if embedding.shape[0] < 128:
            embedding = np.pad(
                embedding, (0, 128 - embedding.shape[0])
            )
        else:
            embedding = embedding[:128]

        return embedding
