import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QListWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QInputDialog
)
from PySide6.QtGui import QPixmap

REFERENCE_DIR = "face/reference_faces"


class ReferenceManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reference Manager")
        self.setFixedSize(400, 500)

        os.makedirs(REFERENCE_DIR, exist_ok=True)

        self.list = QListWidget()
        self.preview = QLabel("Image Preview")
        self.preview.setFixedHeight(200)

        self.add_person = QPushButton("Create Person")
        self.rename_person = QPushButton("Rename")
        self.delete_person = QPushButton("Delete")
        self.add_images = QPushButton("Add Images")

        layout = QVBoxLayout()
        layout.addWidget(self.list)
        layout.addWidget(self.preview)

        btns = QHBoxLayout()
        btns.addWidget(self.add_person)
        btns.addWidget(self.rename_person)
        btns.addWidget(self.delete_person)

        layout.addLayout(btns)
        layout.addWidget(self.add_images)
        self.setLayout(layout)

        self.add_person.clicked.connect(self.create_person)
        self.rename_person.clicked.connect(self.rename)
        self.delete_person.clicked.connect(self.delete)
        self.add_images.clicked.connect(self.add_imgs)
        self.list.currentTextChanged.connect(self.preview_image)

        self.refresh()

    def refresh(self):
        self.list.clear()
        self.list.addItems(os.listdir(REFERENCE_DIR))

    def create_person(self):
        name, ok = QInputDialog.getText(self, "Name", "Person name:")
        if ok:
            os.makedirs(os.path.join(REFERENCE_DIR, name), exist_ok=True)
            self.refresh()

    def rename(self):
        old = self.list.currentItem().text()
        new, ok = QInputDialog.getText(self, "Rename", "New name:")
        if ok:
            os.rename(
                os.path.join(REFERENCE_DIR, old),
                os.path.join(REFERENCE_DIR, new),
            )
            self.refresh()

    def delete(self):
        person = self.list.currentItem().text()
        shutil.rmtree(os.path.join(REFERENCE_DIR, person))
        self.refresh()

    def add_imgs(self):
        person = self.list.currentItem().text()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Images", "", "Images (*.jpg *.png)"
        )
        for f in files:
            shutil.copy(f, os.path.join(REFERENCE_DIR, person))

    def preview_image(self, person):
        p_dir = os.path.join(REFERENCE_DIR, person)
        imgs = os.listdir(p_dir)
        if imgs:
            pix = QPixmap(os.path.join(p_dir, imgs[0]))
            self.preview.setPixmap(pix.scaledToWidth(200))
