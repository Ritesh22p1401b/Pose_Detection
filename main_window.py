from PySide6.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QWidget
)
from face_module.face_window import FaceWindow
from gait.gui.gait_window import GaitWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Person Identification System")
        self.resize(900, 650)

        self.face_btn = QPushButton("FACE Recognition")
        self.gait_btn = QPushButton("GAIT Recognition")

        layout = QVBoxLayout()
        layout.addWidget(self.face_btn)
        layout.addWidget(self.gait_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.face_btn.clicked.connect(self.open_face)
        self.gait_btn.clicked.connect(self.open_gait)

        self.face_window = None
        self.gait_window = None

    def open_face(self):
        if self.face_window is None:
            self.face_window = FaceWindow()
        self.face_window.show()
        self.hide()

    def open_gait(self):
        if self.gait_window is None:
            self.gait_window = GaitWindow()
        self.gait_window.show()
        self.hide()
