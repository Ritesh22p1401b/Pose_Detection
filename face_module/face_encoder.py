import cv2
import numpy as np
import os
import onnxruntime as ort
from insightface.app import FaceAnalysis
from PIL import Image


# ✅ Centralized extension support
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


class FaceEncoder:
    def __init__(self):
        providers = ort.get_available_providers()
        self.ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))

    # --------------------------------------------------
    # 🔥 SAFE IMAGE LOADER (MAIN FIX)
    # --------------------------------------------------
    def load_image_safe(self, path):
        """
        Loads image using OpenCV, fallback to PIL for JPEG issues
        """
        image = cv2.imread(path)

        if image is None:
            try:
                image = np.array(Image.open(path).convert("RGB"))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception:
                return None

        return image

    # --------------------------------------------------
    # QUICK VERIFY ENCODER (UPDATED)
    # --------------------------------------------------
    def encode_images(self, image_paths):
        """
        Used by Quick Verify.
        Encodes uploaded images without saving them.
        Returns a single averaged embedding.
        """
        embeddings = []

        for path in image_paths:
            if not path.lower().endswith(VALID_EXTENSIONS):
                continue

            image = self.load_image_safe(path)
            if image is None:
                continue

            faces = self.app.get(image)
            if not faces:
                continue

            embeddings.append(faces[0].embedding)

        if not embeddings:
            raise RuntimeError("No face detected in selected images")

        return np.mean(np.vstack(embeddings), axis=0)

    # --------------------------------------------------
    # REFERENCE DIRECTORY ENCODER (UPDATED)
    # --------------------------------------------------
    def encode_reference_directory(self, base_dir, selected_persons=None):
        person_db = {}

        if not os.path.exists(base_dir):
            return person_db

        for person_name in os.listdir(base_dir):
            if selected_persons and person_name not in selected_persons:
                continue

            person_dir = os.path.join(base_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            embeddings = []

            for img in os.listdir(person_dir):
                # ✅ NOW SUPPORTS JPEG
                if not img.lower().endswith(VALID_EXTENSIONS):
                    continue

                img_path = os.path.join(person_dir, img)

                image = self.load_image_safe(img_path)
                if image is None:
                    continue

                faces = self.app.get(image)
                if not faces:
                    continue

                embeddings.append(faces[0].embedding)

            if embeddings:
                person_db[person_name] = np.mean(
                    np.vstack(embeddings), axis=0
                )

        return person_db