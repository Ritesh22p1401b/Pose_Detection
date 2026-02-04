import os
import subprocess
from PySide6.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Person Identification System")
        self.resize(900, 650)

        self.face_btn = QPushButton("FACE Recognition")

        layout = QVBoxLayout()
        layout.addWidget(self.face_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.face_btn.clicked.connect(self.open_face)

    def open_face(self):
        project_root = os.path.dirname(os.path.abspath(__file__))

        face_python = os.path.join(
            project_root,
            "face_module",
            "face",
            "Scripts",
            "python.exe"
        )

        face_entry = os.path.join(
            project_root,
            "face_module",
            "face_window.py"
        )

        if not os.path.isfile(face_python):
            QMessageBox.critical(
                self,
                "Face Launch Error",
                f"Face venv python not found:\n{face_python}"
            )
            return

        if not os.path.isfile(face_entry):
            QMessageBox.critical(
                self,
                "Face Launch Error",
                f"face_window.py not found:\n{face_entry}"
            )
            return

        try:
            subprocess.Popen(
                [face_python, face_entry],
                cwd=os.path.dirname(face_entry)
            )
        except Exception as e:
            QMessageBox.critical(self, "Launch Failed", str(e))
