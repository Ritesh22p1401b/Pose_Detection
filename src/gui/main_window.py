from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox, QApplication
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

import cv2
import numpy as np
from PIL import Image

from src.engine.trainer import Trainer
from src.engine.verifier import Verifier
from src.gui.video_thread import VideoThread

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
GAIT_WINDOW = 30   # frames per verification window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gait Recognition System")
        self.resize(900, 600)

        # ---------------- UI ----------------
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)

        self.train_btn = QPushButton("Train (5–6 Videos)")
        self.live_btn = QPushButton("Start Live Camera")
        self.video_btn = QPushButton("Verify Recorded Video")
        self.stop_btn = QPushButton("Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.live_btn)
        layout.addWidget(self.video_btn)
        layout.addWidget(self.stop_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # ------------- Core State -------------
        self.trainer = Trainer()
        self.verifier = None
        self.thread = None
        self.cap = None

        self.frames = []
        self.last_result = False
        self.last_score = 0.0

        # ------------- Signals ----------------
        self.train_btn.clicked.connect(self.train)
        self.live_btn.clicked.connect(self.start_live)
        self.video_btn.clicked.connect(self.verify_video)
        self.stop_btn.clicked.connect(self.stop_video)

    # ---------------------------------------
    # TRAIN PERSON
    # ---------------------------------------
    def train(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select 5–6 Training Videos",
            "",
            "Video Files (*.mp4 *.avi *.mov)"
        )

        if len(files) < 3:
            QMessageBox.warning(self, "Error", "Select at least 3 videos")
            return

        profile_path = self.trainer.build_profile(files, "person_01")
        self.verifier = Verifier(profile_path)

        QMessageBox.information(
            self, "Training Done",
            f"Training completed with {len(files)} videos"
        )

    # ---------------------------------------
    # LIVE CAMERA
    # ---------------------------------------
    def start_live(self):
        if not self.verifier:
            QMessageBox.warning(self, "Error", "Train a person first")
            return

        self.stop_video()
        self.thread = VideoThread(0)
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()

    # ---------------------------------------
    # RECORDED VIDEO
    # ---------------------------------------
    def verify_video(self):
        if not self.verifier:
            QMessageBox.warning(self, "Error", "Train a person first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not path:
            return

        self.stop_video()
        self.cap = cv2.VideoCapture(path)
        self.frames.clear()

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

        self.frames.clear()
        self.video_label.setText("Video Stopped")

    # ---------------------------------------
    # COMMON FRAME PIPELINE
    # ---------------------------------------
    def update_frame(self, frame):
        self.process_frame(frame)

    def process_frame(self, frame):
        self.frames.append(frame)

        if len(self.frames) >= GAIT_WINDOW and self.verifier:
            self.last_score, self.last_result = self.verifier.verify_frames(
                self.frames
            )
            self.frames.clear()

        # Draw bounding box + label
        frame = self.draw_box_and_label(frame)
        self.display_frame(frame)

    # ---------------------------------------
    # DRAW GREEN / RED BOX USING POSE
    # ---------------------------------------
    def draw_box_and_label(self, frame):
        h, w, _ = frame.shape

        # Use pose extractor directly
        pose = self.verifier.pose.pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            xs = [lm.x for lm in result.pose_landmarks.landmark]
            ys = [lm.y for lm in result.pose_landmarks.landmark]

            x1 = int(min(xs) * w)
            y1 = int(min(ys) * h)
            x2 = int(max(xs) * w)
            y2 = int(max(ys) * h)

            color = (0, 255, 0) if self.last_result else (0, 0, 255)
            label = "FOUND" if self.last_result else "NOT FOUND"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ({self.last_score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        return frame

    # ---------------------------------------
    # DISPLAY
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
