import numpy as np

def cosine_similarity(a, b):
    a = np.array([a])
    b = np.array([b])
    return (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
