from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2
from PIL import Image
import numpy as np

from src.gui.video_thread import VideoThread
from src.engine.trainer import Trainer
from src.engine.verifier import Verifier
from src.utils.draw import draw_text

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Verification System")
        self.resize(900, 600)

        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.train_btn = QPushButton("Upload Training Video")
        self.live_btn = QPushButton("Start Live Camera")
        self.verify_btn = QPushButton("Upload Test Video")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.live_btn)
        layout.addWidget(self.verify_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.train_btn.clicked.connect(self.train)
        self.live_btn.clicked.connect(self.start_live)
        self.verify_btn.clicked.connect(self.verify_video)

        self.trainer = Trainer()
        self.verifier = None
        self.thread = None

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        qimg = QImage(
            img.tobytes(),
            img.width,
            img.height,
            QImage.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def train(self):
        path, _ = QFileDialog.getOpenFileName(self, "Training Video")
        if not path:
            return
        profile = self.trainer.build_profile(path)
        self.verifier = Verifier(profile)
        self.video_label.setText("Training Completed")

    def start_live(self):
        if self.thread:
            self.thread.stop()
        self.thread = VideoThread(0)
        self.thread.frame_signal.connect(self.display_frame)
        self.thread.start()

    def verify_video(self):
        if not self.verifier:
            self.video_label.setText("Train first")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Test Video")
        if not path:
            return

        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        score, same = self.verifier.verify_frames(frames)
        result = f"SAME PERSON ({score:.2f})" if same else f"DIFFERENT ({score:.2f})"
        self.video_label.setText(result)
