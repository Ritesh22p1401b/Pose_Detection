# import cv2
# from PySide6.QtWidgets import (
#     QMainWindow, QLabel, QPushButton, QFileDialog,
#     QVBoxLayout, QWidget, QMessageBox
# )
# from PySide6.QtGui import QPixmap, QImage
# from PySide6.QtCore import Qt, QTimer

# from face.webcam import VideoFinder
# from face.face_encoder import FaceEncoder


# # -------- FIXED GUI + VIDEO RESOLUTION --------
# VIDEO_WIDTH = 640
# VIDEO_HEIGHT = 480


# def resize_with_aspect_ratio(frame, target_w, target_h):
#     h, w = frame.shape[:2]
#     scale = min(target_w / w, target_h / h)
#     new_w = int(w * scale)
#     new_h = int(h * scale)
#     return cv2.resize(frame, (new_w, new_h))


# class FaceWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Face Verification")
#         self.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)

#         # -------- VIDEO DISPLAY --------
#         self.video_label = QLabel()
#         self.video_label.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
#         self.video_label.setAlignment(Qt.AlignCenter)
#         self.video_label.setStyleSheet("background-color: black;")

#         # -------- BUTTONS --------
#         self.image_btn = QPushButton("Upload Image")
#         self.video_btn = QPushButton("Verify Video")
#         self.stop_btn = QPushButton("Stop")

#         layout = QVBoxLayout()
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.addWidget(self.video_label)
#         layout.addWidget(self.image_btn)
#         layout.addWidget(self.video_btn)
#         layout.addWidget(self.stop_btn)

#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)

#         # -------- STATE --------
#         self.cap = None
#         self.face_finder = None
#         self.encoder = FaceEncoder()

#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)

#         # -------- SIGNALS --------
#         self.image_btn.clicked.connect(self.load_image)
#         self.video_btn.clicked.connect(self.verify_video)
#         self.stop_btn.clicked.connect(self.stop)

#     # -------- LOAD REFERENCE IMAGE --------
#     def load_image(self):
#         path, _ = QFileDialog.getOpenFileName(
#             self, "Select Face Image", "", "Images (*.jpg *.png)"
#         )
#         if not path:
#             return

#         embedding = self.encoder.encode(path)
#         self.face_finder = VideoFinder([embedding])

#         QMessageBox.information(self, "Face", "Reference image loaded")

#     # -------- START VIDEO VERIFICATION --------
#     def verify_video(self):
#         if self.face_finder is None:
#             QMessageBox.warning(self, "Error", "Upload reference image first")
#             return

#         path, _ = QFileDialog.getOpenFileName(
#             self, "Select Video", "", "Videos (*.mp4 *.avi)"
#         )
#         if not path:
#             return

#         self.cap = cv2.VideoCapture(path)
#         self.timer.start(30)  # ~30 FPS

#     # -------- FRAME UPDATE --------
#     def update_frame(self):
#         if not self.cap or not self.cap.isOpened():
#             return

#         ret, frame = self.cap.read()
#         if not ret:
#             self.stop()
#             return

#         # Face detection + green/red boxes
#         frame, _, _ = self.face_finder.detect_frame(frame)

#         # ---- BEST FIT (NO STRETCH) ----
#         frame = resize_with_aspect_ratio(
#             frame, VIDEO_WIDTH, VIDEO_HEIGHT
#         )

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb.shape
#         qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

#         self.video_label.setPixmap(QPixmap.fromImage(qimg))

#     # -------- STOP VIDEO ONLY --------
#     def stop(self):
#         self.timer.stop()
#         if self.cap:
#             self.cap.release()
#             self.cap = None
#         self.video_label.clear()


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
        self.setWindowTitle("Multi-Person Face Recognition")
        self.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT + 120)

        self.video_label = QLabel()
        self.video_label.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        self.image_btn = QPushButton("Upload Images")
        self.video_btn = QPushButton("Verify Video")
        self.stop_btn = QPushButton("Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.image_btn)
        layout.addWidget(self.video_btn)
        layout.addWidget(self.stop_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.encoder = FaceEncoder()
        self.face_finder = None
        self.cap = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.image_btn.clicked.connect(self.load_images)
        self.video_btn.clicked.connect(self.verify_video)
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

    # -------- VIDEO --------
    def verify_video(self):
        if self.face_finder is None:
            QMessageBox.warning(self, "Error", "Upload face images first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi)"
        )
        if not path:
            return

        self.cap = cv2.VideoCapture(path)
        self.timer.start(30)

    def update_frame(self):
        if not self.cap:
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

    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
