import os
import shutil
import cv2
from PySide6.QtWidgets import (
    QWidget, QListWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QInputDialog
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal

REFERENCE_DIR = "reference_faces"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


class ReferenceManager(QWidget):
    persons_selected = Signal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reference Manager")
        self.setFixedSize(420, 550)

        os.makedirs(REFERENCE_DIR, exist_ok=True)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.MultiSelection)

        self.preview = QLabel("Image Preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedHeight(200)
        self.preview.setStyleSheet("border: 1px solid gray;")

        self.select_btn = QPushButton("Select Profiles")
        self.add_person = QPushButton("Create Person")
        self.rename_person = QPushButton("Rename")
        self.delete_person = QPushButton("Delete")
        self.add_images = QPushButton("Add Images")

        layout = QVBoxLayout(self)
        layout.addWidget(self.list)
        layout.addWidget(self.preview)
        layout.addWidget(self.select_btn)

        btns = QHBoxLayout()
        btns.addWidget(self.add_person)
        btns.addWidget(self.rename_person)
        btns.addWidget(self.delete_person)

        layout.addLayout(btns)
        layout.addWidget(self.add_images)

        self.select_btn.clicked.connect(self.select_profiles)
        self.add_person.clicked.connect(self.create_person)
        self.rename_person.clicked.connect(self.rename)
        self.delete_person.clicked.connect(self.delete)
        self.add_images.clicked.connect(self.add_imgs)
        self.list.currentItemChanged.connect(self.preview_image)

        self.refresh()

    def refresh(self):
        self.list.clear()
        for name in sorted(os.listdir(REFERENCE_DIR)):
            if os.path.isdir(os.path.join(REFERENCE_DIR, name)):
                self.list.addItem(name)

    def select_profiles(self):
        items = self.list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Error", "No profile selected")
            return

        selected = [i.text() for i in items]

        QMessageBox.information(
            self,
            "Selected Profiles",
            "Selected for verification:\n\n" + ", ".join(selected)
        )

        self.persons_selected.emit(selected)
        self.close()

    def create_person(self):
        name, ok = QInputDialog.getText(self, "Create Person", "Person name:")
        if ok and name.strip():
            os.makedirs(os.path.join(REFERENCE_DIR, name.strip()), exist_ok=True)
            self.refresh()

    def rename(self):
        item = self.list.currentItem()
        if not item:
            return

        new, ok = QInputDialog.getText(self, "Rename", "New name:")
        if ok and new.strip():
            os.rename(
                os.path.join(REFERENCE_DIR, item.text()),
                os.path.join(REFERENCE_DIR, new.strip())
            )
            self.refresh()

    def delete(self):
        item = self.list.currentItem()
        if not item:
            return

        shutil.rmtree(os.path.join(REFERENCE_DIR, item.text()))
        self.preview.clear()
        self.refresh()

    def is_valid_image(self, path):
        """Validate image using OpenCV"""
        try:
            img = cv2.imread(path)
            return img is not None
        except:
            return False

    def add_imgs(self):
        item = self.list.currentItem()
        if not item:
            QMessageBox.warning(self, "Error", "Select a person first")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.jpg *.jpeg *.png)"
        )

        if not files:
            return

        dest_dir = os.path.join(REFERENCE_DIR, item.text())
        added = 0
        skipped = 0

        for f in files:
            ext = os.path.splitext(f)[1].lower()

            if ext not in VALID_EXTENSIONS:
                skipped += 1
                continue

            if not self.is_valid_image(f):
                skipped += 1
                continue

            try:
                filename = os.path.basename(f)
                shutil.copy(f, os.path.join(dest_dir, filename))
                added += 1
            except Exception as e:
                print("Copy error:", e)
                skipped += 1

        QMessageBox.information(
            self,
            "Upload Result",
            f"Added: {added}\nSkipped: {skipped}"
        )

        self.preview_image(item)

    def preview_image(self, item):
        if not item:
            self.preview.clear()
            return

        person_dir = os.path.join(REFERENCE_DIR, item.text())

        images = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        if not images:
            self.preview.setText("No images")
            return

        img_path = os.path.join(person_dir, images[0])

        pix = QPixmap(img_path)

        # 🔥 Fix for PNG / corrupted preview
        if pix.isNull():
            self.preview.setText("Preview not available")
            return

        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )