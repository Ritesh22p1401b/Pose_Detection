# import numpy as np

# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     """
#     Compute cosine similarity between two face embeddings.
#     Returns a float between -1 and 1 (higher is more similar).
#     """
#     if a.shape != b.shape:
#         raise ValueError("Embeddings must have the same shape")

#     similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#     return float(similarity)


import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Embeddings must have the same shape")

    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(similarity)
