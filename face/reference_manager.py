import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QListWidget, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QInputDialog
)

REFERENCE_DIR = "face/reference_faces"


class ReferenceManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reference Face Manager")
        self.setFixedSize(300, 400)

        os.makedirs(REFERENCE_DIR, exist_ok=True)

        self.person_list = QListWidget()

        self.create_btn = QPushButton("Create Person Folder")
        self.add_img_btn = QPushButton("Add Images")
        self.refresh_btn = QPushButton("Refresh List")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.create_btn)
        btn_layout.addWidget(self.add_img_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.person_list)
        layout.addLayout(btn_layout)
        layout.addWidget(self.refresh_btn)

        self.setLayout(layout)

        self.create_btn.clicked.connect(self.create_person)
        self.add_img_btn.clicked.connect(self.add_images)
        self.refresh_btn.clicked.connect(self.load_persons)

        self.load_persons()

    def load_persons(self):
        self.person_list.clear()
        for name in os.listdir(REFERENCE_DIR):
            path = os.path.join(REFERENCE_DIR, name)
            if os.path.isdir(path):
                self.person_list.addItem(name)

    def create_person(self):
        name, ok = QInputDialog.getText(
            self, "Create Person", "Enter person name:"
        )
        if not ok or not name.strip():
            return

        path = os.path.join(REFERENCE_DIR, name)
        os.makedirs(path, exist_ok=True)
        self.load_persons()

    def add_images(self):
        item = self.person_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Error", "Select a person first")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.jpg *.png)"
        )
        if not paths:
            return

        person_dir = os.path.join(REFERENCE_DIR, item.text())

        for p in paths:
            shutil.copy(p, person_dir)

        QMessageBox.information(
            self, "Done", f"{len(paths)} images added"
        )
