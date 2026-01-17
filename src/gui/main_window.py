from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox, QApplication
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

import cv2
import numpy as np
from PIL import Image
import joblib
import os

from src.inference.skeleton_extractor import SkeletonExtractor
from src.inference.gait_matcher import GaitMatcher
from src.inference.gender_features import extract_gender_features
from src.gui.video_thread import VideoThread
from collections import deque


# ---------------- CONFIG ----------------
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

GAIT_MODEL_PATH = "checkpoints/gait_model.pth"
GENDER_MODEL_PATH = "checkpoints/gender_model.pkl"
REFERENCE_PATH = "profiles/reference_embedding.npy"
# ---------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gait Recognition System")
        self.resize(900, 600)

        # ---------------- UI ----------------
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)

        self.train_btn = QPushButton("Add Training Videos (3–6)")
        self.build_btn = QPushButton("Build Reference Profile")
        self.live_btn = QPushButton("Start Live Camera")
        self.video_btn = QPushButton("Verify Recorded Video")
        self.stop_btn = QPushButton("Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.build_btn)
        layout.addWidget(self.live_btn)
        layout.addWidget(self.video_btn)
        layout.addWidget(self.stop_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # ------------- Core State -------------
        self.extractor = SkeletonExtractor()
        self.gender_model = joblib.load(GENDER_MODEL_PATH)

        self.matcher = None
        self.reference_embeddings = []
        self.reference_ready = False

        self.thread = None
        self.cap = None


        self.found_spikes = 0
        self.not_found_spikes = 0

        self.SPIKE_CONFIRM = 4      # REQUIRED FOUND spikes
        self.SPIKE_RESET = 10      # RESET after NOT FOUND streak

        self.skeleton_buffer = []
        self.GENDER_WINDOW = 30  # frames

        self.match_votes = deque(maxlen=15)


        self.last_found = False
        self.last_score = 0.0
        self.last_gender = "Unknown"

        # ------------- Signals ----------------
        self.train_btn.clicked.connect(self.add_training_videos)
        self.build_btn.clicked.connect(self.build_reference)
        self.live_btn.clicked.connect(self.start_live)
        self.video_btn.clicked.connect(self.verify_video)
        self.stop_btn.clicked.connect(self.stop_video)

    # ---------------------------------------
    # ADD TRAINING VIDEOS
    # ---------------------------------------
    def add_training_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select 3–6 Training Videos",
            "",
            "Video Files (*.mp4 *.avi *.mov)"
        )

        if not files:
            QMessageBox.warning(self, "Cancelled", "No videos selected.")
            return

        if len(files) < 3:
            QMessageBox.warning(
                self, "Error",
                "Please select at least 3 training videos."
            )
            return

        added = 0
        for path in files:
            skeleton = self.extractor.extract_from_video(path)
            if len(skeleton) == 0:
                continue

            if self.matcher is None:
                self.matcher = GaitMatcher(
                    GAIT_MODEL_PATH,
                    num_joints=12
                )


            emb = self.matcher.embed(skeleton)
            self.reference_embeddings.append(emb)
            added += 1

        if added == 0:
            QMessageBox.critical(
                self,
                "Failed",
                "No valid person detected in selected videos."
            )
        else:
            QMessageBox.information(
                self,
                "Success",
                f"{added} training video(s) uploaded successfully."
            )

    # ---------------------------------------
    # BUILD REFERENCE PROFILE
    # ---------------------------------------
    def build_reference(self):
        if len(self.reference_embeddings) < 3:
            QMessageBox.warning(
                self,
                "Error",
                "At least 3 valid training videos are required."
            )
            return

        os.makedirs("profiles", exist_ok=True)

        reference = self.matcher.build_reference(self.reference_embeddings)
        np.save(REFERENCE_PATH, reference)

        self.reference_ready = True

        QMessageBox.information(
            self,
            "Reference Created",
            "Reference profile created successfully.\n"
            "You can now start live camera or verify a video."
        )

    # ---------------------------------------
    # LIVE CAMERA
    # ---------------------------------------
    def start_live(self):
        if not self.reference_ready:
            QMessageBox.warning(
                self,
                "Error",
                "Please build the reference profile first."
            )
            return

        self.stop_video()

        self.thread = VideoThread(0)
        self.thread.frame_signal.connect(self.process_frame)
        self.thread.start()

        QMessageBox.information(
            self,
            "Live Camera",
            "Live camera started successfully."
        )

    # ---------------------------------------
    # RECORDED VIDEO
    # ---------------------------------------
    def verify_video(self):
        if not self.reference_ready:
            QMessageBox.warning(
                self,
                "Error",
                "Please build the reference profile first."
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov)"
        )

        if not path:
            QMessageBox.warning(self, "Cancelled", "No video selected.")
            return

        self.stop_video()
        self.cap = cv2.VideoCapture(path)

        QMessageBox.information(
            self,
            "Video Loaded",
            "Recorded video loaded successfully.\n"
            "Starting verification."
        )

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.process_frame(frame)
            QApplication.processEvents()

        self.cap.release()
        self.cap = None

    # ---------------------------------------
    # STOP VIDEO
    # ---------------------------------------
    def stop_video(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.setText("Video Stopped")

        # QMessageBox.information(
        #     self,
        #     "Stopped",
        #     "Video processing stopped successfully."
        # )

    # ---------------------------------------
    # FRAME PROCESSING
    # ---------------------------------------
    def process_frame(self, frame):
        joints = self.extractor.extract_from_frame(frame)

        if joints is not None:
            # Build temporal buffer
            self.skeleton_buffer.append(joints)

            # Keep buffer size bounded
            if len(self.skeleton_buffer) > self.GENDER_WINDOW:
                self.skeleton_buffer.pop(0)

            # ---------------- PERSON IDENTIFICATION ----------------
            if len(self.skeleton_buffer) >= 10:
                seq = np.array(self.skeleton_buffer, dtype=np.float32)

                ref = np.load(REFERENCE_PATH)
                emb = self.matcher.embed(seq)

                found, score = self.matcher.match(emb, ref)
                if found:
                    self.found_spikes += 1
                    self.not_found_spikes = 0
                else:
                    self.not_found_spikes += 1
                    self.found_spikes = 0

                # ---- FINAL DECISION ----
                if self.found_spikes >= self.SPIKE_CONFIRM:
                    self.last_found = True
                elif self.not_found_spikes >= self.SPIKE_RESET:
                    self.last_found = False

                self.last_score = score


            # ---------------- GENDER CLASSIFICATION ----------------
            if len(self.skeleton_buffer) >= self.GENDER_WINDOW:
                seq = np.array(self.skeleton_buffer, dtype=np.float32)
                features = extract_gender_features(seq)

                g = self.gender_model.predict(features)[0]
                self.last_gender = "Male" if g == 0 else "Female"

        frame = self.draw_box_and_label(frame)
        self.display_frame(frame)


    # ---------------------------------------
    # DRAW GREEN / RED BOX
    # ---------------------------------------
    def draw_box_and_label(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.extractor.pose.process(rgb)

        if res.pose_landmarks:
            xs = [lm.x for lm in res.pose_landmarks.landmark]
            ys = [lm.y for lm in res.pose_landmarks.landmark]

            x1, y1 = int(min(xs) * w), int(min(ys) * h)
            x2, y2 = int(max(xs) * w), int(max(ys) * h)

            color = (0, 255, 0) if self.last_found else (0, 0, 255)
            label = (
                f"FOUND | {self.last_gender}"
                if self.last_found
                else "NOT FOUND"
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        return frame

    # ---------------------------------------
    # DISPLAY FRAME
    # ---------------------------------------
    def display_frame(self, frame):
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(rgb)
        qimg = QImage(
            img.tobytes(),
            img.width,
            img.height,
            QImage.Format_RGB888
        )

        self.video_label.setPixmap(QPixmap.fromImage(qimg))
