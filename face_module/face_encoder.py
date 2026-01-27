import cv2
import numpy as np
import os
import onnxruntime as ort
from insightface.app import FaceAnalysis


class FaceEncoder:
    def __init__(self):
        providers = ort.get_available_providers()
        self.ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))

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
                if not img.lower().endswith((".jpg", ".png")):
                    continue

                img_path = os.path.join(person_dir, img)
                image = cv2.imread(img_path)
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
