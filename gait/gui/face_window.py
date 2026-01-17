import cv2
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox, QApplication
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

from face.webcam import VideoFinder
from face.face_encoder import FaceEncoder


class FaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition")
        self.resize(900, 650)

        # ---------------- UI ----------------
        self.video_label = QLabel("Face Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)

        self.image_btn = QPushButton("Upload Reference Image")
        self.video_btn = QPushButton("Upload Video")
        self.stop_btn = QPushButton("Stop Verification")
        self.back_btn = QPushButton("Back")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.image_btn)
        layout.addWidget(self.video_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.back_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # ---------------- State ----------------
        self.cap = None
        self.face_finder = None
        self.reference_embeddings = []

        # Initialize encoder ONCE (important)
        self.encoder = FaceEncoder()

        # ---------------- Signals ----------------
        self.image_btn.clicked.connect(self.load_image)
        self.video_btn.clicked.connect(self.verify_video)
        self.stop_btn.clicked.connect(self.stop)
        self.back_btn.clicked.connect(self.go_back)

    # -------- IMAGE UPLOAD --------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Face Image", "", "Images (*.jpg *.png)"
        )
        if not path:
            return

        try:
            # Encode reference face (returns single embedding)
            embedding = self.encoder.encode(path)

            # VideoFinder expects a LIST of embeddings
            self.reference_embeddings = [embedding]
            self.face_finder = VideoFinder(self.reference_embeddings)

            QMessageBox.information(
                self, "Face", "Reference image loaded successfully"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # -------- VIDEO VERIFY --------
    def verify_video(self):
        if self.face_finder is None:
            QMessageBox.warning(self, "Error", "Upload face image first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi)"
        )
        if not path:
            return

        self.cap = cv2.VideoCapture(path)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, found, score = self.face_finder.detect_frame(frame)
            self.display(frame)
            QApplication.processEvents()

        self.cap.release()

    # -------- STOP VIDEO ONLY --------
    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.setText("Stopped")

    # -------- BACK --------
    def go_back(self):
        self.close()
        if self.parent():
            self.parent().show()

    # -------- DISPLAY --------
    def display(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
