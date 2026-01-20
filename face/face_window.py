import cv2
import os
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

from face.webcam import VideoFinder
from face.face_encoder import FaceEncoder
from face.reference_manager import ReferenceManager

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
REFERENCE_DIR = "face/reference_faces"


def resize_with_aspect_ratio(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


# --------------------------------------------------
# MAIN WINDOW
# --------------------------------------------------
class FaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification System")
        self.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT + 260)

        # -------- VIDEO DISPLAY --------
        self.video_label = QLabel()
        self.video_label.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        # -------- BUTTONS --------
        self.manage_btn = QPushButton("Manage Reference Faces")
        self.load_btn = QPushButton("Load Reference Database")
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

        # -------- STATE --------
        self.encoder = FaceEncoder()
        self.face_finder = None
        self.cap = None
        self.quick_mode = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # -------- SIGNALS --------
        self.manage_btn.clicked.connect(self.open_reference_manager)
        self.load_btn.clicked.connect(self.load_reference_db)
        self.quick_btn.clicked.connect(self.quick_verify)
        self.video_btn.clicked.connect(self.verify_video)
        self.live_btn.clicked.connect(self.start_live)
        self.stop_btn.clicked.connect(self.stop)

    # --------------------------------------------------
    # QUICK VERIFY (TEMPORARY / IN-MEMORY)
    # --------------------------------------------------
    def quick_verify(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Face Images (Quick Verify)",
            "",
            "Images (*.jpg *.png)"
        )
        if not paths:
            return

        try:
            embeddings = self.encoder.encode_images(paths)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # Temporary person DB (no persistence)
        temp_db = {f"QuickPerson_{i}": emb for i, emb in enumerate(embeddings)}

        self.face_finder = VideoFinder(temp_db)
        self.quick_mode = True

        QMessageBox.information(
            self,
            "Quick Verify Ready",
            f"{len(embeddings)} images loaded (temporary)"
        )

    # --------------------------------------------------
    # MANAGED MODE (FOLDER-BASED)
    # --------------------------------------------------
    def open_reference_manager(self):
        self.ref_manager = ReferenceManager()
        self.ref_manager.show()

    def load_reference_db(self):
        if not os.path.exists(REFERENCE_DIR):
            QMessageBox.warning(self, "Error", "No reference faces found")
            return

        person_db = self.encoder.encode_reference_directory(REFERENCE_DIR)
        if not person_db:
            QMessageBox.warning(self, "Error", "No valid faces found")
            return

        self.face_finder = VideoFinder(person_db)
        self.quick_mode = False

        QMessageBox.information(
            self,
            "Database Loaded",
            f"{len(person_db)} persons loaded"
        )

    # --------------------------------------------------
    # RECORDED VIDEO VERIFICATION
    # --------------------------------------------------
    def verify_video(self):
        if self.face_finder is None:
            QMessageBox.warning(self, "Error", "Load database or use Quick Verify first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Videos (*.mp4 *.avi *.mov)"
        )
        if not path:
            return

        self.stop()
        self.cap = cv2.VideoCapture(path)
        self.timer.start(30)

    # --------------------------------------------------
    # LIVE CAMERA VERIFICATION
    # --------------------------------------------------
    def start_live(self):
        if self.face_finder is None:
            QMessageBox.warning(self, "Error", "Load database or use Quick Verify first")
            return

        self.stop()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open webcam")
            return

        self.timer.start(30)

    # --------------------------------------------------
    # FRAME UPDATE LOOP
    # --------------------------------------------------
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

    # --------------------------------------------------
    # STOP CURRENT FEED ONLY
    # --------------------------------------------------
    def stop(self):
        self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.clear()

        # Clear quick verify state safely
        if self.quick_mode:
            self.face_finder = None
            self.quick_mode = False
