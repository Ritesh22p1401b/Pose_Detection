import os
import cv2
import numpy as np
from collections import deque

from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox, QApplication
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PIL import Image

from gait.inference.skeleton_extractor import SkeletonExtractor
from gait.inference.gait_matcher import GaitMatcher
from gait.inference.gender_features import extract_gender_features
import joblib


DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

GAIT_MODEL_PATH = "checkpoints/gait_model.pth"
GAIT_REFERENCE_PATH = "data/profiles/gait_reference.npy"
GENDER_MODEL_PATH = "gait/checkpoints/gender_model.pkl"

GAIT_WINDOW = 20
SPIKE_CONFIRM = 4
SPIKE_RESET = 5


class GaitWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gait Verification")
        self.resize(900, 650)

        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)

        self.train_btn = QPushButton("Add Training Videos")
        self.build_btn = QPushButton("Build Reference")
        self.live_btn = QPushButton("Start Live Camera")
        self.video_btn = QPushButton("Verify Video")
        self.stop_btn = QPushButton("Stop")

        layout = QVBoxLayout()
        for w in [
            self.video_label, self.train_btn, self.build_btn,
            self.live_btn, self.video_btn, self.stop_btn
        ]:
            layout.addWidget(w)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.extractor = SkeletonExtractor()
        self.matcher = None

        self.training_embeddings = []
        self.reference_ready = False

        self.skeleton_buffer = deque(maxlen=GAIT_WINDOW)
        self.found_spikes = 0
        self.not_found_spikes = 0
        self.last_found = False
        self.last_gender = "Unknown"

        self.gender_model = joblib.load(GENDER_MODEL_PATH) \
            if os.path.exists(GENDER_MODEL_PATH) else None

        self.cap = None

        self.train_btn.clicked.connect(self.add_training_videos)
        self.build_btn.clicked.connect(self.build_reference)
        self.live_btn.clicked.connect(self.start_live)
        self.video_btn.clicked.connect(self.verify_video)
        self.stop_btn.clicked.connect(self.stop_video)

    # ---------------- TRAINING ----------------
    def add_training_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Training Videos", "", "Video Files (*.mp4 *.avi *.mov)"
        )

        if len(files) < 3:
            QMessageBox.warning(self, "Error", "Select at least 3 videos")
            return

        if self.matcher is None:
            self.matcher = GaitMatcher(GAIT_MODEL_PATH, num_joints=12)

        added = 0
        for path in files:
            seq = self.extractor.extract_from_video(path, max_frames=120)
            if len(seq) == 0:
                continue

            emb = self.matcher.embed(seq)
            self.training_embeddings.append(emb)
            added += 1

        QMessageBox.information(self, "Training", f"Added {added} videos")

    def build_reference(self):
        if len(self.training_embeddings) < 3:
            QMessageBox.warning(self, "Error", "Not enough training data")
            return

        ref = self.matcher.build_reference(self.training_embeddings)
        os.makedirs(os.path.dirname(GAIT_REFERENCE_PATH), exist_ok=True)
        np.save(GAIT_REFERENCE_PATH, ref)

        self.reference_ready = True
        QMessageBox.information(self, "Reference", "Reference built successfully")

    # ---------------- VIDEO ----------------
    def start_live(self):
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        self.run_loop()

    def verify_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        self.stop_video()
        self.cap = cv2.VideoCapture(path)
        self.run_loop()

    def stop_video(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        self.video_label.setText("Stopped")

    def run_loop(self):
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.process_frame(frame)
            QApplication.processEvents()

    # ---------------- CORE ----------------
    def process_frame(self, frame):
        joints = self.extractor.extract_from_frame(frame)
        if joints is not None:
            self.skeleton_buffer.append(joints)

        if self.reference_ready and len(self.skeleton_buffer) == GAIT_WINDOW:
            seq = np.array(self.skeleton_buffer, dtype=np.float32)
            ref = np.load(GAIT_REFERENCE_PATH)

            emb = self.matcher.embed(seq)
            found, score = self.matcher.match(emb, ref)

            if found:
                self.found_spikes += 1
                self.not_found_spikes = 0
            else:
                self.not_found_spikes += 1
                self.found_spikes = 0

            if self.found_spikes >= SPIKE_CONFIRM:
                self.last_found = True
            elif self.not_found_spikes >= SPIKE_RESET:
                self.last_found = False

            if self.gender_model:
                g = self.gender_model.predict(
                    extract_gender_features(seq)
                )[0]
                self.last_gender = "Male" if g == 0 else "Female"

        label = "FOUND" if self.last_found else "NOT FOUND"
        color = (0, 255, 0) if self.last_found else (0, 0, 255)

        cv2.putText(
            frame, f"{label} | {self.last_gender}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )

        self.display(frame)

    def display(self, frame):
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))