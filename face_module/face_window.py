import os
import sys
import io

# ------------------------------
# FIX Windows stdout crash
# ------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="ignore")

# ------------------------------
# Silence TensorFlow INFO logs
# ------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

from webcam import VideoFinder
from face_encoder import FaceEncoder
from reference_manager import ReferenceManager
from camera.auto_camera import AutoCamera


VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480


def resize_with_aspect_ratio(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


class FaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification System")
        self.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT + 260)

        self.video_label = QLabel()
        self.video_label.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        self.manage_btn = QPushButton("Manage Reference Faces")
        self.load_btn = QPushButton("Load Selected Profiles")
        self.quick_btn = QPushButton("Quick Verify (No Save)")
        self.video_btn = QPushButton("Verify Video")
        self.live_btn = QPushButton("Live Camera")
        self.stop_btn = QPushButton("Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.manage_btn)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.quick_btn)
        layout.addWidget(self.video_btn)
        layout.addWidget(self.live_btn)
        layout.addWidget(self.stop_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.encoder = FaceEncoder()
        self.face_finder = None

        self.cap = None
        self.camera = None

        self.quick_mode = False
        self.selected_persons = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.manage_btn.clicked.connect(self.open_reference_manager)
        self.load_btn.clicked.connect(self.load_selected_profiles)
        self.quick_btn.clicked.connect(self.quick_verify)
        self.video_btn.clicked.connect(self.verify_video)
        self.live_btn.clicked.connect(self.start_live)
        self.stop_btn.clicked.connect(self.stop)

    def open_reference_manager(self):
        self.ref_manager = ReferenceManager()
        self.ref_manager.persons_selected.connect(self.set_selected_persons)
        self.ref_manager.show()

    def set_selected_persons(self, persons):
        self.selected_persons = persons
        self.quick_mode = False

    def load_selected_profiles(self):
        from reference_manager import REFERENCE_DIR

        if not self.selected_persons:
            QMessageBox.warning(self, "Error", "No profiles selected")
            return

        person_db = self.encoder.encode_reference_directory(
            REFERENCE_DIR,
            selected_persons=self.selected_persons
        )

        if not person_db:
            QMessageBox.warning(self, "Error", "No valid face images found")
            return

        self.face_finder = VideoFinder(person_db)
        self.quick_mode = False

        # ✅ FIXED: show selected profile names instead of count
        QMessageBox.information(
            self,
            "Profiles Loaded",
            "The following profile(s) are loaded for verification:\n\n"
            + ", ".join(self.selected_persons)
        )

    def quick_verify(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Quick Verify", "", "Images (*.jpg *.png)"
        )
        if not paths:
            return

        ref = self.encoder.encode_images(paths)
        ref = ref / np.linalg.norm(ref)

        self.face_finder = VideoFinder({"QuickPerson": ref})
        self.quick_mode = True
        self.selected_persons = []

        QMessageBox.information(self, "Ready", "Quick verification loaded successfully.")

    def verify_video(self):
        if not self.face_finder:
            QMessageBox.warning(self, "Error", "Load profiles first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)"
        )
        if not path:
            return

        self.stop()
        self.cap = cv2.VideoCapture(path)
        self.timer.start(30)

    def start_live(self):
        if not self.face_finder:
            QMessageBox.warning(self, "Error", "Load profiles first")
            return

        self.stop()
        self.camera = AutoCamera(index=1)
        self.cap = None
        self.timer.start(30)

    def update_frame(self):
        if self.camera:
            frame = self.camera.read()
            if frame is None:
                return
        elif self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                return
        else:
            return

        frame = self.face_finder.detect_frame(frame)
        frame = resize_with_aspect_ratio(frame, VIDEO_WIDTH, VIDEO_HEIGHT)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        self.video_label.setPixmap(
            QPixmap.fromImage(
                QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            )
        )

    def stop(self):
        self.timer.stop()

        if self.camera:
            self.camera.release()
            self.camera = None

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.clear()


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = FaceWindow()
    win.show()
    sys.exit(app.exec())
