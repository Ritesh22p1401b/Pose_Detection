from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

import cv2
import numpy as np
from PIL import Image

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

DETECTION_INTERVAL = 15   # detect person every 15 frames
GAIT_WINDOW = 30          # frames per gait check

from src.gui.video_thread import VideoThread
from src.engine.trainer import Trainer
from src.engine.verifier import Verifier


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gait Verification System")
        self.resize(900, 600)

        # ---------------- UI ----------------
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        # self.video_label.setMinimumHeight(400)
        self.video_label.setFixedSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)


        self.train_btn = QPushButton("Upload Training Video")
        self.live_btn = QPushButton("Start Live Camera")
        self.verify_btn = QPushButton("Upload Test Video")
        self.stop_btn = QPushButton("Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.live_btn)
        layout.addWidget(self.verify_btn)
        layout.addWidget(self.stop_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # ------------- Core State -------------
        self.trainer = Trainer()
        self.verifier = None
        self.thread = None

        self.active_person_id = None
        self.buffer_frames = []

        # -------- Person Detector (HOG) --------
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )

        self.tracked_box = None

        # ------------- Signals ----------------
        self.train_btn.clicked.connect(self.train)
        self.live_btn.clicked.connect(self.start_live)
        self.verify_btn.clicked.connect(self.verify_video)
        self.stop_btn.clicked.connect(self.stop_video)

    # ---------------------------------------
    # TRAIN PERSON (REFERENCE SLOT)
    # ---------------------------------------
    def train(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Video"
        )
        if not path:
            return

        person_id = "person_01"  # can be user input later

        profile_path = self.trainer.build_profile(path, person_id)
        self.verifier = Verifier(profile_path)
        self.active_person_id = person_id

        self.video_label.setText(
            f"Training completed\nActive person: {person_id}"
        )

    # ---------------------------------------
    # START LIVE CAMERA
    # ---------------------------------------
    def start_live(self):
        if self.thread:
            self.stop_video()

        self.thread = VideoThread(0)
        self.thread.frame_signal.connect(self.display_frame)
        self.thread.start()

    # ---------------------------------------
    # VERIFY UPLOADED VIDEO (WITH BOX)
    # ---------------------------------------
    def verify_video(self):
        if not self.verifier:
            self.video_label.setText("Train a person first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Video"
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.display_frame(frame)

            # allow GUI refresh
            cv2.waitKey(1)

        cap.release()

    # ---------------------------------------
    # STOP VIDEO PROCESSING (NOT APP)
    # ---------------------------------------
    def stop_video(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None

        self.buffer_frames.clear()
        self.tracked_box = None

        self.video_label.setText("Video stopped")

    # ---------------------------------------
    # DISPLAY FRAME WITH TRACKING + BOX
    # ---------------------------------------
    def display_frame(self, frame):
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------- DETECT PERSON ----------
        if self.tracked_box is None:
            boxes, _ = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05
            )

            if len(boxes) > 0:
                self.tracked_box = boxes[0]

        # ---------- TRACK & VERIFY ----------
        same = False
        score = 0.0

        if self.tracked_box is not None:
            x, y, w, h = self.tracked_box
            person_crop = frame[y:y+h, x:x+w]

            if person_crop.size > 0:
                self.buffer_frames.append(person_crop)

            if self.verifier and len(self.buffer_frames) >= 30:
                score, same = self.verifier.verify_frames(
                    self.buffer_frames
                )
                self.buffer_frames.clear()

            # ---------- DRAW BOX ----------
            color = (0, 255, 0) if same else (0, 0, 255)
            label = "SAME PERSON" if same else "DIFFERENT PERSON"

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                display,
                f"{label} ({score:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # ---------- SHOW FRAME ----------
        # display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        display = cv2.resize(
            display,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(display)

        qimg = QImage(
            img.tobytes(),
            img.width,
            img.height,
            QImage.Format_RGB888
        )

        self.video_label.setPixmap(QPixmap.fromImage(qimg))
