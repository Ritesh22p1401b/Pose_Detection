import cv2
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

from face.webcam import VideoFinder
from face.face_encoder import FaceEncoder

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480


def resize_with_aspect_ratio(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


class FaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Person Face Verification")
        self.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT + 160)

        # -------- VIDEO DISPLAY --------
        self.video_label = QLabel()
        self.video_label.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        # -------- BUTTONS --------
        self.image_btn = QPushButton("Upload Images")
        self.video_btn = QPushButton("Verify Video")
        self.live_btn = QPushButton("Live Camera")
        self.stop_btn = QPushButton("Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.image_btn)
        layout.addWidget(self.video_btn)
        layout.addWidget(self.live_btn)
        layout.addWidget(self.stop_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # -------- STATE --------
        self.encoder = FaceEncoder()
        self.face_finder = None
        self.cap = None
        self.is_live = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # -------- SIGNALS --------
        self.image_btn.clicked.connect(self.load_images)
        self.video_btn.clicked.connect(self.verify_video)
        self.live_btn.clicked.connect(self.start_live)
        self.stop_btn.clicked.connect(self.stop)

    # -------- MULTIPLE IMAGE UPLOAD --------
    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Face Images", "", "Images (*.jpg *.png)"
        )
        if not paths:
            return

        embeddings = self.encoder.encode_images(paths)
        self.face_finder = VideoFinder(embeddings)

        QMessageBox.information(
            self,
            "Success",
            f"{len(embeddings)} reference faces loaded"
        )

    # -------- RECORDED VIDEO --------
    def verify_video(self):
        if self.face_finder is None:
            QMessageBox.warning(self, "Error", "Upload face images first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Videos (*.mp4 *.avi *.mov)"
        )
        if not path:
            return

        self.stop()  # ensure clean state
        self.cap = cv2.VideoCapture(path)
        self.is_live = False
        self.timer.start(30)

    # -------- LIVE CAMERA --------
    def start_live(self):
        if self.face_finder is None:
            QMessageBox.warning(self, "Error", "Upload face images first")
            return

        self.stop()  # stop any running feed
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open webcam")
            return

        self.is_live = True
        self.timer.start(30)

    # -------- FRAME UPDATE --------
    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        frame = self.face_finder.detect_frame(frame)
        frame = resize_with_aspect_ratio(frame, VIDEO_WIDTH, VIDEO_HEIGHT)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # -------- STOP ONLY FEED --------
    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
