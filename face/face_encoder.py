import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEncoder:
    def __init__(self):
        # CPU-safe (AMD compatible)
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def encode(self, image_path: str) -> np.ndarray:
        """
        Encode a single reference image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        faces = self.app.get(image)
        if not faces:
            raise ValueError(f"No face detected in image: {image_path}")

        return faces[0].embedding

    def encode_images(self, image_paths):
        """
        Encode multiple reference images.
        Returns a list of embeddings.
        """
        embeddings = []

        for path in image_paths:
            image = cv2.imread(path)
            if image is None:
                print(f"[WARN] Cannot read image: {path}")
                continue

            faces = self.app.get(image)
            if not faces:
                print(f"[WARN] No face detected in image: {path}")
                continue

            embeddings.append(faces[0].embedding)

        if not embeddings:
            raise ValueError("No valid faces found in reference images.")

        return embeddings
