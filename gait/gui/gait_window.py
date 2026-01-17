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

# ---- GAIT IMPORTS (your existing modules) ----
from gait.inference.skeleton_extractor import SkeletonExtractor
from gait.inference.gait_matcher import GaitMatcher
from gait.inference.gender_features import extract_gender_features

import joblib


# ---------------- CONFIG ----------------
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

GAIT_MODEL_PATH = "gait/checkpoints/gait_model.pth"
GAIT_REFERENCE_PATH = "profiles/gait_reference.npy"
GENDER_MODEL_PATH = "checkpoints/gender_model.pkl"

GAIT_WINDOW = 20           # frames per window
SPIKE_CONFIRM = 4          # required FOUND spikes
SPIKE_RESET = 5            # reset after NOT FOUND streak
# ---------------------------------------


class GaitWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gait Recognition")
        self.resize(900, 650)

        # ---------------- UI ----------------
        self.video_label = QLabel("Gait Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)

        self.train_btn = QPushButton("Add Training Videos (3–6)")
        self.build_btn = QPushButton("Build Reference")
        self.live_btn = QPushButton("Start Live Camera")
        self.video_btn = QPushButton("Verify Recorded Video")
        self.stop_btn = QPushButton("Stop")
        self.back_btn = QPushButton("Back")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.build_btn)
        layout.addWidget(self.live_btn)
        layout.addWidget(self.video_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.back_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # ---------------- STATE ----------------
        self.cap = None

        self.extractor = SkeletonExtractor()
        self.matcher = None

        self.training_embeddings = []
        self.reference_ready = False

        self.skeleton_buffer = []
        self.found_spikes = 0
        self.not_found_spikes = 0
        self.last_found = False
        self.last_gender = "Unknown"

        # Gender model (safe load)
        self.gender_model = None
        if os.path.exists(GENDER_MODEL_PATH):
            self.gender_model = joblib.load(GENDER_MODEL_PATH)
            print("[INFO] Gender model loaded")
        else:
            print("[INFO] Gender model not found. Gender disabled.")

        # ---------------- SIGNALS ----------------
        self.train_btn.clicked.connect(self.add_training_videos)
        self.build_btn.clicked.connect(self.build_reference)
        self.live_btn.clicked.connect(self.start_live)
        self.video_btn.clicked.connect(self.verify_video)
        self.stop_btn.clicked.connect(self.stop_video)
        self.back_btn.clicked.connect(self.go_back)

    # =================================================
    # TRAINING
    # =================================================
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
            cap = cv2.VideoCapture(path)
            seq = []

            while cap.isOpened() and len(seq) < 120:
                ret, frame = cap.read()
                if not ret:
                    break
                joints = self.extractor.extract_from_frame(frame)
                if joints is not None:
                    seq.append(joints)

            cap.release()

            if len(seq) == 0:
                continue

            seq = np.array(seq, dtype=np.float32)
            emb = self.matcher.embed(seq)
            self.training_embeddings.append(emb)
            added += 1

        QMessageBox.information(self, "Training", f"Added {added} training videos")

    def build_reference(self):
        if self.matcher is None or len(self.training_embeddings) < 3:
            QMessageBox.warning(self, "Error", "Add more training videos first")
            return

        ref = self.matcher.build_reference(self.training_embeddings)
        os.makedirs(os.path.dirname(GAIT_REFERENCE_PATH), exist_ok=True)
        np.save(GAIT_REFERENCE_PATH, ref)

        self.reference_ready = True
        QMessageBox.information(self, "Reference", "Gait reference built")

    # =================================================
    # VIDEO CONTROL
    # =================================================
    def start_live(self):
        self.stop_video()
        self.cap = cv2.VideoCapture(0)
        self.run_video_loop()

    def verify_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not path:
            return

        self.stop_video()
        self.cap = cv2.VideoCapture(path)
        self.run_video_loop()

    def stop_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.setText("Video Stopped")

    def run_video_loop(self):
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.process_frame(frame)
            QApplication.processEvents()

    # =================================================
    # CORE GAIT PIPELINE
    # =================================================
    def process_frame(self, frame):
        joints = self.extractor.extract_from_frame(frame)
        if joints is not None:
            self.skeleton_buffer.append(joints)
            if len(self.skeleton_buffer) > GAIT_WINDOW:
                self.skeleton_buffer.pop(0)

            if self.reference_ready and len(self.skeleton_buffer) >= GAIT_WINDOW:
                seq = np.array(self.skeleton_buffer, dtype=np.float32)
                ref = np.load(GAIT_REFERENCE_PATH)

                emb = self.matcher.embed(seq)
                found, score = self.matcher.match(emb, ref)

                # ---- Spike-based decision ----
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

                # ---- Gender (optional) ----
                if self.gender_model is not None:
                    features = extract_gender_features(seq)
                    g = self.gender_model.predict(features)[0]
                    self.last_gender = "Male" if g == 0 else "Female"
                else:
                    self.last_gender = "Unknown"

        # ---- Overlay ----
        color = (0, 255, 0) if self.last_found else (0, 0, 255)
        label = "FOUND" if self.last_found else "NOT FOUND"

        cv2.putText(
            frame,
            f"{label} | {self.last_gender}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        self.display_frame(frame)

    # =================================================
    # DISPLAY
    # =================================================
    def display_frame(self, frame):
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        qimg = QImage(
            img.tobytes(), img.width, img.height, QImage.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # =================================================
    # NAVIGATION
    # =================================================
    def go_back(self):
        self.close()
        if self.parent():
            self.parent().show()
