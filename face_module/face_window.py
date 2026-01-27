import cv2
import os
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

from face_module.webcam import VideoFinder
from face_module.face_encoder import FaceEncoder
from face_module.reference_manager import ReferenceManager
from camera.auto_camera import AutoCamera


VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
REFERENCE_DIR = "face/reference_faces"


def resize_with_aspect_ratio(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


class FaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification System")
        self.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT + 260)

        # ---------------- VIDEO ----------------
        self.video_label = QLabel()
        self.video_label.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        # ---------------- BUTTONS ----------------
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

        # ---------------- STATE ----------------
        # ⚠ CUDA / CPU decision happens INSIDE these classes
        self.encoder = FaceEncoder()
        self.face_finder = None

        self.cap = None
        self.quick_mode = False
        self.selected_persons = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # ---------------- SIGNALS ----------------
        self.manage_btn.clicked.connect(self.open_reference_manager)
        self.load_btn.clicked.connect(self.load_selected_profiles)
        self.quick_btn.clicked.connect(self.quick_verify)
        self.video_btn.clicked.connect(self.verify_video)
        self.live_btn.clicked.connect(self.start_live)
        self.stop_btn.clicked.connect(self.stop)

    # --------------------------------------------------
    # QUICK VERIFY (TEMPORARY, NO SAVE)
    # --------------------------------------------------
    def quick_verify(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Quick Verify", "", "Images (*.jpg *.png)"
        )
        if not paths:
            return

        try:
            embeddings = self.encoder.encode_images(paths)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # VideoFinder decides CUDA / CPU internally
        self.face_finder = VideoFinder({
            "QuickPerson": embeddings
        })

        self.quick_mode = True
        self.selected_persons = []

        QMessageBox.information(self, "Ready", "Quick Verify enabled")

    # --------------------------------------------------
    # REFERENCE MANAGER
    # --------------------------------------------------
    def open_reference_manager(self):
        self.ref_manager = ReferenceManager()
        self.ref_manager.persons_selected.connect(self.set_selected_persons)
        self.ref_manager.show()

    def set_selected_persons(self, persons):
        self.selected_persons = persons
        self.quick_mode = False

        QMessageBox.information(
            self,
            "Profiles Selected",
            "Selected profiles:\n\n" + "\n".join(persons)
        )

    def load_selected_profiles(self):
        if not self.selected_persons:
            QMessageBox.warning(self, "Error", "No profiles selected")
            return

        person_db = self.encoder.encode_reference_directory(
            REFERENCE_DIR,
            selected_persons=self.selected_persons
        )

        if not person_db:
            QMessageBox.warning(
                self,
                "Error",
                "No valid face images found for selected profiles"
            )
            return

        # VideoFinder decides CUDA / CPU internally
        self.face_finder = VideoFinder(person_db)
        self.quick_mode = False

        QMessageBox.information(
            self,
            "Loaded",
            "Verification loaded for:\n\n" + "\n".join(person_db.keys())
        )

    # --------------------------------------------------
    # VIDEO / LIVE
    # --------------------------------------------------
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

    # def start_live(self):
    #     if not self.face_finder:
    #         QMessageBox.warning(self, "Error", "Load profiles first")
    #         return

    #     self.stop()
    #     self.cap = cv2.VideoCapture(0)

    #     if not self.cap.isOpened():
    #         QMessageBox.critical(self, "Error", "Cannot open webcam")
    #         return

    #     self.timer.start(30)
    def start_live(self):
        if not self.face_finder:
            QMessageBox.warning(self, "Error", "Load profiles first")
            return

        self.stop()

        self.camera = AutoCamera()

        try:
            self.cap = self.camera.open()
        except RuntimeError as e:
            QMessageBox.critical(self, "Camera Error", str(e))
            return

        print(f"[INFO] Live camera source: {self.camera.active_source}")

        self.timer.start(30)


    # --------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------
    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        frame = self.face_finder.detect_frame(frame)

        if self.quick_mode:
            label = "QUICK VERIFY"
        else:
            label = "PROFILES: " + ", ".join(self.selected_persons)

        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        frame = resize_with_aspect_ratio(frame, VIDEO_WIDTH, VIDEO_HEIGHT)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        self.video_label.setPixmap(
            QPixmap.fromImage(
                QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            )
        )

    # --------------------------------------------------
    # STOP (SAFE)
    # --------------------------------------------------
    # def stop(self):
    #     self.timer.stop()

    #     if self.cap:
    #         self.cap.release()
    #         self.cap = None

    #     self.video_label.clear()
    def stop(self):
        self.timer.stop()

        if hasattr(self, "camera") and self.camera:
            self.camera.release()
            self.camera = None

        self.cap = None
        self.video_label.clear()
