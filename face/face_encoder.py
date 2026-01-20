import cv2
import numpy as np
import os
import pickle
import onnxruntime as ort
from insightface.app import FaceAnalysis


class FaceEncoder:
    def __init__(self):
        """
        Automatically selects CUDA if available, otherwise CPU.
        """

        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in available_providers:
            print("[INFO] CUDA available → Using GPU")
            self.ctx_id = 0
        else:
            print("[INFO] CUDA not available → Using CPU")
            self.ctx_id = -1

        # Initialize InsightFace
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))

    # ------------------------------------------------------------------
    # EXISTING METHODS (UNCHANGED)
    # ------------------------------------------------------------------

    def encode(self, image_path: str) -> np.ndarray:
        """
        Encode a single reference image.
        Returns a face embedding.
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

    # ------------------------------------------------------------------
    # NEW METHODS (PROJECT EXTENSIONS)
    # ------------------------------------------------------------------

    def _is_blurry(self, image, threshold=80.0):
        """
        Check image blur using Laplacian variance.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < threshold

    def encode_reference_directory(self, base_dir):
        """
        Encode reference faces stored as:
        base_dir/
            Person_1/
                img1.jpg
                img2.jpg
            Person_2/
                img1.jpg

        Returns:
            dict: { person_name: mean_embedding }
        """
        person_db = {}

        if not os.path.exists(base_dir):
            print(f"[WARN] Reference directory not found: {base_dir}")
            return person_db

        for person_name in os.listdir(base_dir):
            person_dir = os.path.join(base_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            embeddings = []

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # ---- Quality checks ----
                if self._is_blurry(image):
                    continue

                faces = self.app.get(image)
                if not faces:
                    continue

                face = faces[0]

                # ---- Pose filtering (yaw / pitch) ----
                yaw, pitch, _ = face.pose
                if abs(yaw) > 60 or abs(pitch) > 45:
                    continue

                embeddings.append(face.embedding)

            if embeddings:
                person_db[person_name] = np.mean(
                    np.vstack(embeddings), axis=0
                )
                print(f"[INFO] Loaded {person_name} ({len(embeddings)} images)")
            else:
                print(f"[WARN] No valid faces for {person_name}")

        return person_db

    # ------------------------------------------------------------------
    # EXPORT / IMPORT (DATABASE)
    # ------------------------------------------------------------------

    def export_database(self, person_db, file_path):
        """
        Save reference embeddings to disk.
        """
        with open(file_path, "wb") as f:
            pickle.dump(person_db, f)
        print(f"[INFO] Database exported to {file_path}")

    def import_database(self, file_path):
        """
        Load reference embeddings from disk.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        with open(file_path, "rb") as f:
            person_db = pickle.load(f)

        print(f"[INFO] Database imported from {file_path}")
        return person_db
