from PySide6.QtWidgets import *
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2
from PIL import Image

from src.engine.trainer import Trainer
from src.engine.verifier import Verifier
from src.gui.video_thread import VideoThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Recognition")

        self.label = QLabel("Video")
        self.label.setFixedSize(640, 480)

        self.train_btn = QPushButton("Train (5–6 Videos)")
        self.live_btn = QPushButton("Live Camera")
        self.stop_btn = QPushButton("Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.live_btn)
        layout.addWidget(self.stop_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.trainer = Trainer()
        self.verifier = None
        self.thread = None
        self.frames = []

        self.train_btn.clicked.connect(self.train)
        self.live_btn.clicked.connect(self.start_live)
        self.stop_btn.clicked.connect(self.stop)

    def train(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Videos")
        profile = self.trainer.build_profile(files, "person_01")
        self.verifier = Verifier(profile)

    def start_live(self):
        self.thread = VideoThread(0)
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()

    def stop(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.frames.clear()

    def update_frame(self, frame):
        self.frames.append(frame)
        if len(self.frames) >= 30 and self.verifier:
            score, same = self.verifier.verify_frames(self.frames)
            self.frames.clear()
            txt = "SAME" if same else "DIFFERENT"
            cv2.putText(frame, txt, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if same else (0, 0, 255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))
